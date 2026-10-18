// Flash Next's prompt indexer scores: attention._scores' bits from a row-tiled kernel.
//
// Ported from MiaAI-Lab's Qwen3.8 Flash Next recipe, patch 0002 (fn_qsa_scores.cu, MIT licence, Copyright (c) 2026
// MiaAI-Lab), itself built on the TensorFold contributors' Triton `_scores`. `_scores` gives one Triton program a
// (row, 64-block tile) and so reads every block's pooled key once per row. Here a CTA of 256 threads takes 32 rows x 64
// blocks, stages the tile's pooled keys and the rows' index queries in shared memory once, and each thread computes
// 2 rows x 4 blocks serially.
//
// The arithmetic is `_scores` as Triton 3.7 compiles it for sm_121 with 16-byte-aligned pointers (TTGIR layout
// sizePerThread [1, 8], threadsPerWarp [2, 16], warpsPerCTA [4, 1], so lane l of a row holds dims [8l, 8l + 8)):
//   lane partial  p_l = fma(k7,q7, ... fma(k2,q2, fma(k0,q0, k1*q1)))   (mul.f32, then fma.rn.f32.bf16)
//   lane tree     shfl.bfly 8, 4, 2, 1 with add.f32: a_l = p_l + p_{l+8}, b_l = a_l + a_{l+4}, c_l = b_l + b_{l+2},
//                 S = c_0 + c_1 (fp32 addition commutes, so every lane holds this sum)
//   heads         total = max(S_0, 0) + 0, then total = total + max(S_h, 0) for h = 1..3 (max.f32, add.f32)
//   store         div.full.f32 total / 0x413504F3 (sqrt 128 as Triton folds it)
// Each step below is the same PTX instruction on the same operands, so each score is the same fp32 word. Rows with
// TOP or fewer complete blocks and blocks past a row's complete ones are not stored, as in `_scores`.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int DI = 128;           // index head dim
constexpr int HI = 4;             // index heads
constexpr int TR = 2, TB = 4;     // rows and blocks a thread
constexpr int RT = 16 * TR;       // rows a CTA
constexpr int BT = 16 * TB;       // blocks a CTA (`_scores`' BB)
constexpr int THREADS = 256;

// a key row's 16 chunks of 8 dims stored at chunk ^ (row & 15): 16 rows' same chunk land in 16 different banks
__device__ __forceinline__ int kslot(int b, int c) { return b * DI + ((c ^ (b & 15)) * 8); }

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float add_rn(float a, float b) {
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float max0(float a) {
    float d;
    asm("max.f32 %0, %1, 0f00000000;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float div_sqrt_di(float a) {
    float d;
    asm("div.full.f32 %0, %1, 0f413504F3;" : "=f"(d) : "f"(a));
    return d;
}

struct F8 {
    float v[8];
};

// 8 bf16 as 8 fp32 (exact widenings, element 0 the low half of word 0)
__device__ __forceinline__ F8 widen(const uint4& w) {
    F8 f;
    f.v[0] = __uint_as_float(w.x << 16);
    f.v[1] = __uint_as_float(w.x & 0xFFFF0000u);
    f.v[2] = __uint_as_float(w.y << 16);
    f.v[3] = __uint_as_float(w.y & 0xFFFF0000u);
    f.v[4] = __uint_as_float(w.z << 16);
    f.v[5] = __uint_as_float(w.z & 0xFFFF0000u);
    f.v[6] = __uint_as_float(w.w << 16);
    f.v[7] = __uint_as_float(w.w & 0xFFFF0000u);
    return f;
}

// lane partial over 8 dims: k1 q1 first, then k0 q0, k2 q2 .. k7 q7 fused. A product of two bf16 values is exact in
// fp32, so fma.rn.f32 on the widened values gives fma.rn.f32.bf16's result.
__device__ __forceinline__ float lane(const F8& k, const F8& q) {
    float p = fma_rn(k.v[0], q.v[0], mul_rn(k.v[1], q.v[1]));
#pragma unroll
    for (int i = 2; i < 8; ++i) p = fma_rn(k.v[i], q.v[i], p);
    return p;
}

// IQ [rows, HI, DI] bf16, POOLED [blocks, DI] bf16, POS0 the first row's position, SC [rows, NB] fp32
__global__ void __launch_bounds__(THREADS)
qsa_scores_kernel(const uint16_t* __restrict__ IQ, const uint16_t* __restrict__ POOLED, const int* __restrict__ POS0,
                  float* __restrict__ SC, int NB, int rows, int ratio, int top) {
    __shared__ __align__(16) uint16_t ks[BT * DI];
    __shared__ __align__(16) uint16_t qs[RT * HI * DI];
    const int tid = threadIdx.x;
    const int r0 = blockIdx.x * RT;
    const int j0 = blockIdx.y * BT;
    const int pos0 = __ldg(POS0);
    const int last = min(r0 + RT, rows) - 1;
    const int reach = (pos0 + last + 1) / ratio;   // the tile's longest row's complete blocks
    if (reach <= top || j0 >= reach) return;
    // the keys of blocks [j0, j0 + BT) below `reach` (zeros past it), then the rows' queries
    for (int i = tid; i < BT * (DI / 8); i += THREADS) {
        const int b = i / (DI / 8), c = i % (DI / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (j0 + b < reach) v = __ldg(reinterpret_cast<const uint4*>(POOLED + (size_t)(j0 + b) * DI) + c);
        *reinterpret_cast<uint4*>(ks + kslot(b, c)) = v;
    }
    for (int i = tid; i < RT * HI * (DI / 8); i += THREADS) {
        const int rh = i / (DI / 8), c = i % (DI / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r0 + rh / HI < rows) v = __ldg(reinterpret_cast<const uint4*>(IQ + ((size_t)r0 * HI + rh) * DI) + c);
        *reinterpret_cast<uint4*>(qs + rh * DI + c * 8) = v;
    }
    __syncthreads();
    const int tr = tid / 16;   // rows tr * TR ..
    const int tb = tid % 16;   // blocks tb + 16 * jb
    float total[TR][TB];
#pragma unroll 1
    for (int h = 0; h < HI; ++h) {
        float c0[TR][TB], cc[TR][TB], bb[TR][TB], aa[TR][TB];
        // the 16 lanes in the tree's order: step bit 0 pairs l with l + 8, bit 1 with l + 4, bit 2 with l + 2, bit 3
        // takes lane 1's half after lane 0's
#pragma unroll 1
        for (int step = 0; step < 16; ++step) {
            const int half = step >> 3, q2 = (step >> 2) & 1, q4 = (step >> 1) & 1, q8 = step & 1;
            const int l = half + 2 * q2 + 4 * q4 + 8 * q8;
            F8 qf[TR];
#pragma unroll
            for (int ir = 0; ir < TR; ++ir)
                qf[ir] = widen(*reinterpret_cast<const uint4*>(qs + ((tr * TR + ir) * HI + h) * DI + l * 8));
#pragma unroll
            for (int jb = 0; jb < TB; ++jb) {
                const F8 kf = widen(*reinterpret_cast<const uint4*>(ks + kslot(tb + 16 * jb, l)));
#pragma unroll
                for (int ir = 0; ir < TR; ++ir) {
                    const float p = lane(kf, qf[ir]);
                    if (q8 == 0) {
                        aa[ir][jb] = p;
                        continue;
                    }
                    const float a = add_rn(aa[ir][jb], p);           // a_l = p_l + p_{l+8}
                    if (q4 == 0) {
                        bb[ir][jb] = a;
                        continue;
                    }
                    const float b = add_rn(bb[ir][jb], a);           // b_l = a_l + a_{l+4}
                    if (q2 == 0) {
                        cc[ir][jb] = b;
                        continue;
                    }
                    const float c = add_rn(cc[ir][jb], b);           // c_l = b_l + b_{l+2}
                    if (half == 0) {
                        c0[ir][jb] = c;
                        continue;
                    }
                    const float s = max0(add_rn(c0[ir][jb], c));     // relu(c_0 + c_1)
                    total[ir][jb] = h == 0 ? add_rn(s, 0.0f) : add_rn(total[ir][jb], s);
                }
            }
        }
    }
#pragma unroll
    for (int ir = 0; ir < TR; ++ir) {
        const int r = r0 + tr * TR + ir;
        if (r >= rows) continue;
        const int complete = (pos0 + r + 1) / ratio;
        if (complete <= top) continue;
#pragma unroll
        for (int jb = 0; jb < TB; ++jb) {
            const int bk = j0 + tb + 16 * jb;
            if (bk < complete) SC[(size_t)r * NB + bk] = div_sqrt_di(total[ir][jb]);
        }
    }
}

}  // namespace

void qsa_scores_cuda(const at::Tensor& iq, const at::Tensor& pooled, const at::Tensor& pos0, at::Tensor& scores,
                     int64_t rows, int64_t blocks, int64_t ratio, int64_t top) {
    if (rows <= 0 || blocks <= 0) return;
    const dim3 grid((unsigned)((rows + RT - 1) / RT), (unsigned)((blocks + BT - 1) / BT));
    qsa_scores_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint16_t*>(iq.data_ptr()), reinterpret_cast<const uint16_t*>(pooled.data_ptr()),
        pos0.data_ptr<int>(), scores.data_ptr<float>(), (int)scores.stride(0), (int)rows, (int)ratio, (int)top);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
