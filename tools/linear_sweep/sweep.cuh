// Launch-shape variants of linear.cu's linear_kernel for tools/exl3_linear_sweep.py (measurement only; nothing in
// src/ calls them). Every variant keeps the shipped reduction: each warp's mma chain over the same K range that
// plan(K, N) gives it, warps added in warp order, then splits in split order, so the outputs are the same bytes.
//
//   orig_kernel<..., MINB, PFD>  linear_kernel (RT = 1) under __launch_bounds__(WK * 32, MINB), which caps the
//                                registers so MINB blocks fit an SM, and with PFD > 0 an L2 prefetch of the k step
//                                PFD ahead of the one being loaded.
//   kind 2: orig_kernel with the shipped kernel's grid_wait, to compare like with like.
//   wsplit_kernel<..., WKL, MINB> the same warps spread over WK / WKL blocks of WKL warps (more, smaller blocks):
//                                each warp stores its own partial sums to Z [SK * WK, M, N]; the last block of a
//                                column block adds them in the same order (warps, then splits). Windows of 1-16 rows.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "decode.cuh"
#include "pdl.cuh"

using namespace tf_exl3;

namespace sweep {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        constexpr int LPW = 8 / K2;
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

// linear_kernel with RT = 1, launch bounds (WK * 32, MINB) and an optional L2 prefetch PFD steps ahead.
template <int K2, int CB, int WK, int MINB, int PFD, int GW = 0>
__global__ void __launch_bounds__(WK * 32, MINB) orig_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype, long long ldy,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];
    __shared__ int last;
    const int RH = min(M, 8);

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s)
        if (s < per_warp)
            for (int w = 32 * lane; w < 8 * TW; w += 32 * 32)
                prefetch_l2(tiles + (size_t)(kt0 + s) * stride_k + w);
    if constexpr (GW) tf_exl3_pdl::grid_wait();      // the shipped kernel's griddepcontrol.wait (kind 2)

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int RP = min(16, M - m0);
        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
        const int r0 = m0 + (g < RP ? g : RP - 1), r1 = m0 + (g + 8 < RP ? g + 8 : RP - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PFD > 0) {
                if (i + 1 + PFD < per_warp)
                    for (int w = 32 * lane; w < 8 * TW; w += 32 * 32)
                        prefetch_l2(tile + (size_t)(i + 1 + PFD) * stride_k + w);
            }
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        const int R = RP, mt = m0;
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();
            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 q0 = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = q0.x; v[1] = q0.y; v[2] = q0.z; v[3] = q0.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(mt + rlo + r) * ldy + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + mt + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < RP; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * ldy + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}

// The WK warps of plan(K, N) spread over WK / WKL blocks of WKL warps; per-warp partials in Z [SK * WK, M, N].
template <int K2, int CB, int WKL, int MINB>
__global__ void __launch_bounds__(WKL * 32, MINB) wsplit_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype, long long ldy,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, int WK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    __shared__ int last;
    const int P = WK / WKL;
    const int nb = blockIdx.x, s = blockIdx.y / P, p = blockIdx.y % P;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int gw = p * WKL + warp;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = s * (per_warp * WK) + gw * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }
#pragma unroll
    for (int q = 0; q < 2; ++q)
        if (q < per_warp)
            for (int w = 32 * lane; w < 8 * TW; w += 32 * 32)
                prefetch_l2(tiles + (size_t)(kt0 + q) * stride_k + w);

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
    const int r0 = g < M ? g : M - 1, r1 = g + 8 < M ? g + 8 : M - 1;
    const half* x0 = xh + (size_t)r0 * K;
    const half* x1 = xh + (size_t)r1 * K;
    const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
    constexpr bool PF = K2 <= 12;
    constexpr int SR = step_regs<K2>();
    uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
    if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
    for (int i = 0; i < per_warp; ++i) {
        const int kt = kt0 + i;
        uint32_t a[4];
        a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
        a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
        a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
        a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
        if constexpr (PF) {
            if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t w[LW];
            if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
            else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w, lane, b0, b1);
            mma16816(acc[j][0], a, b0);
            mma16816(acc[j][1], a, b1);
        }
        if constexpr (PF) {
#pragma unroll
            for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
        }
    }

    float* zw = Z + (size_t)(s * WK + gw) * M * N + col0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            if (g < M) *reinterpret_cast<float2*>(zw + (size_t)g * N + col) = make_float2(acc[i][h][0], acc[i][h][1]);
            if (g + 8 < M)
                *reinterpret_cast<float2*>(zw + (size_t)(g + 8) * N + col) = make_float2(acc[i][h][2], acc[i][h][3]);
        }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counters + nb, 1) == SK * P - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const size_t MN = (size_t)M * N;
    for (int r = warp; r < M; r += WKL) {
        float4 tot = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int q = 0; q < SK; ++q) {
            const float* zr = Z + (size_t)q * WK * MN + (size_t)r * N + col0 + 4 * lane;
            float4 sum = __ldcg(reinterpret_cast<const float4*>(zr));
            for (int w = 1; w < WK; ++w) {
                const float4 u = __ldcg(reinterpret_cast<const float4*>(zr + (size_t)w * MN));
                sum.x += u.x; sum.y += u.y; sum.z += u.z; sum.w += u.w;
            }
            if (q == 0) {
                tot = sum;
            } else {
                tot.x += sum.x; tot.y += sum.y; tot.z += sum.z; tot.w += sum.w;
            }
        }
        float v[4] = {tot.x, tot.y, tot.z, tot.w};
        finish(v, lane, svh, bias, col0 + 4 * lane);
        store4(y, y_dtype, (size_t)r * ldy + col0 + 4 * lane, v);
    }
    if (threadIdx.x == 0) counters[nb] = 0;
}

// A streaming read of n16 16-byte words (the attainable-bandwidth reference): each thread xors its words.
template <int U>
__global__ void read_kernel(const uint4* __restrict__ p, long long n16, unsigned* __restrict__ out) {
    unsigned acc = 0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n16; i += U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = __ldcs(p + i + u * stride);
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n16; i += stride) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x * blockDim.x + threadIdx.x] = acc;   // keeps the loads live
}

inline int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace sweep
