// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "experts_grouped.cuh"

namespace {

using tf_exl3x::HAD_SCALE;
using tf_exl3x::bf16r;
using tf_exl3x::fwht128;

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
// With ``items`` it also lists the work of grouped_kernel's listed layout: (place << 16 | row tile) for each tile of
// ``tile`` members an expert holds, places in id order, and their count in icount[0].
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

// Exclusive scan of v over the block's threads into the return value; tot[32] is shared scratch, total gets the sum.
__device__ __forceinline__ int block_scan(int v, int* tot, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += x;
    }
    if (lane == 31) tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int w = tot[lane];
        int s = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        tot[lane] = s - w;
        if (lane == 31) *total = s;
    }
    __syncthreads();
    return tot[warp] + inc - v;
}

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm,
                                                              int* __restrict__ items, int* __restrict__ icount,
                                                              int tile) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    __shared__ int item_tot[GROUP_THREADS / 32];
    tf_exl3_pdl::grid_wait();
    tf_exl3_pdl::grid_launch();
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    const int first = warp_tot[warp] + inc - used;
    int place = first;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
    if (items == nullptr) return;                         // (uniform: every thread takes the same branch)
    int tiles = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) tiles += (min(cnt[q], maxm) + tile - 1) / tile;
    int at = block_scan(tiles, item_tot, icount);
    place = first;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int nt = (min(cnt[q], maxm) + tile - 1) / tile;
        for (int t = 0; t < nt; ++t) items[at++] = (place << 16) | t;
        ++place;
    }
}

// group_kernel's outputs (the same uids, ucount, members, items and icount) by a count, a scan and a stable scatter
// (TF_EXL3_GROUP_SCAN). group_kernel's threads each scan every pick once an expert they hold: on a 1,024-row window
// of Flash Next (11,264 picks, 513 experts) 129 threads of one block ran 2.3 ms. Here every block counts all picks
// (warps merge equal experts with match_any) and scans the counts to the places in id order; block b then owns
// experts [b * per, (b + 1) * per): their ids, items and members. Members come out in pick order because the picks
// go in rounds of 1,024 in order: a pick's rank is the expert's count before the round, plus its count in earlier
// warps of the round, plus its lanes below in its warp. Expert e is thread e of each block, so E <= 1,024.
constexpr int SCAN_THREADS = 1024;
constexpr int SCAN_BLOCKS = 8;

__global__ void __launch_bounds__(SCAN_THREADS) group_scan_kernel(const int* __restrict__ pick,
                                                                   int* __restrict__ uids, int* __restrict__ ucount,
                                                                   int* __restrict__ members, int R, int slots, int E,
                                                                   int maxm, int* __restrict__ items,
                                                                   int* __restrict__ icount, int tile) {
    extern __shared__ int sh_scan[];
    int* cnt = sh_scan;                                               // [E] picks an expert
    int* place = sh_scan + E;                                         // [E] place, -1 if unused
    unsigned short* at = reinterpret_cast<unsigned short*>(sh_scan + 2 * E);   // [32][E] a round's warp counts/offsets
    __shared__ int used_tot[SCAN_THREADS / 32];
    __shared__ int item_tot[SCAN_THREADS / 32];
    __shared__ int totals[2];
    tf_exl3_pdl::grid_wait();
    tf_exl3_pdl::grid_launch();
    const int n = R * slots, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned below = (1u << lane) - 1u;
    if (tid < E) cnt[tid] = 0;
    for (int i = tid; i < 32 * E; i += SCAN_THREADS) at[i] = 0;
    __syncthreads();
    for (int i0 = 0; i0 < n; i0 += SCAN_THREADS) {                   // every lane takes every round (match_any)
        const int i = i0 + tid;
        const int v = i < n ? pick[i] : -1;
        const int key = v >= 0 && v < E ? v : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, key);
        if (key >= 0 && (peers & below) == 0) atomicAdd(&cnt[key], __popc(peers));
    }
    __syncthreads();
    const int e = tid;
    const int c = e < E ? cnt[e] : 0;
    const int mc = min(c, maxm);
    const int pl = block_scan(c > 0, used_tot, &totals[0]);
    const int it0 = block_scan(items != nullptr && c > 0 ? (mc + tile - 1) / tile : 0, item_tot, &totals[1]);
    if (e < E) place[e] = c > 0 ? pl : -1;
    const int per = (E + gridDim.x - 1) / gridDim.x, e0 = blockIdx.x * per, e1 = min(E, e0 + per);
    const bool own = e >= e0 && e < e1;
    if (own && c > 0) {
        uids[pl] = e;
        if (items != nullptr)
            for (int t = 0; t * tile < mc; ++t) items[it0 + t] = (pl << 16) | t;
    }
    if (blockIdx.x == 0 && tid == 0) {
        ucount[0] = totals[0];
        if (items != nullptr) icount[0] = totals[1];
    }
    __syncthreads();
    // the owned places' members past their count: -1
    for (int f = tid; f < (e1 - e0) * maxm; f += SCAN_THREADS) {
        const int q = e0 + f / maxm, j = f % maxm;
        if (place[q] >= 0 && j >= min(cnt[q], maxm)) members[(size_t)place[q] * maxm + j] = -1;
    }
    // the owned experts' picks, rank by rank
    int run = 0;                                                      // thread e: e's picks before this round
    for (int i0 = 0; i0 < n; i0 += SCAN_THREADS) {
        const int i = i0 + tid;
        const int v = i < n ? pick[i] : -1;
        const int key = v >= e0 && v < e1 ? v : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, key);
        const bool leader = key >= 0 && (peers & below) == 0;
        if (leader) at[warp * E + key] = (unsigned short)__popc(peers);
        __syncthreads();
        if (own) {                                                    // counts -> offsets, warps in order
            int r = run;
            for (int w = 0; w < 32; ++w) {
                const int k = at[w * E + e];
                if (k) at[w * E + e] = (unsigned short)r;             // a warp without e keeps its zero
                r += k;
            }
            run = r;
        }
        __syncthreads();
        if (key >= 0) {
            const int j = at[warp * E + key] + __popc(peers & below);
            if (j < maxm) members[(size_t)place[key] * maxm + j] = (i / slots) * 32 + (i % slots);
        }
        __syncwarp();
        if (leader) at[warp * E + key] = 0;                           // the next round's empty warps read zero
        __syncthreads();
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    tf_exl3_pdl::grid_wait();
    tf_exl3_pdl::grid_launch();
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    tf_exl3_pdl::grid_wait();
    tf_exl3_pdl::grid_launch();
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    // the arithmetic of tf_exl3x::gateup_tail (epilogue.cuh), kept inline: through the call this kernel's SASS
    // schedules two independent shuffle/add pairs in another order (the same operations, so the same bits)
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32, or rounded to
// bf16 once as a copy of the fp32 Y into a bf16 tensor rounds it (a prompt's MoE buffer, TF_FLASHNEXT_MOE_DIRECT).
template <typename TO>
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, TO* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    tf_exl3_pdl::grid_wait();
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    tf_exl3x::down_tail(v, lane, svh_d + (size_t)e * D + n, y + (size_t)p * D + n);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    tf_exl3_pdl::grid_wait();
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    tf_exl3_pdl::grid_wait();
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

namespace tf_exl3x {
extern template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void decode_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void decode_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void decode_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void dequant_launch<0>(const uint32_t*, half*, int, int, int, uint32_t, cudaStream_t);
extern template void dequant_launch<1>(const uint32_t*, half*, int, int, int, uint32_t, cudaStream_t);
extern template void dequant_launch<2>(const uint32_t*, half*, int, int, int, uint32_t, cudaStream_t);
}  // namespace tf_exl3x

void exl3x_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                        const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                        int64_t SK, int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo,
                        int64_t hi, int64_t rt, int64_t wmask, bool decode, bool pdl, const int* items,
                        const int* icount, int64_t nitems, int64_t epi, const void* s0, const void* s1,
                        const void* s2, void* out, int* arrivals, double limit, int64_t act, bool bf16,
                        bool staged) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    a.rt = (int)rt;
    a.wmask = (uint32_t)wmask;
    a.pdl = tf_exl3_pdl::enabled(pdl);
    a.items = items;
    a.icount = icount;
    a.nitems = (int)nitems;
    a.epi = (int)epi;
    a.ep.s0 = static_cast<const half*>(s0);
    a.ep.s1 = static_cast<const half*>(s1);
    a.ep.s2 = static_cast<const half*>(s2);
    a.ep.out = out;
    a.ep.arrivals = arrivals;
    a.ep.limit = (float)limit;
    a.ep.act = (int)act;
    a.ep.bf16 = bf16;
    a.staged = staged;
    TORCH_CHECK(!(decode && epi), "decode_kernel has no fused epilogue");
    TORCH_CHECK(!(decode && staged), "the staged path is grouped_kernel's");
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(cb >= 0 && cb <= 2, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    if (decode) {
        if (cb == 0) tf_exl3x::decode_launch<0>(a, stream);
        else if (cb == 1) tf_exl3x::decode_launch<1>(a, stream);
        else tf_exl3x::decode_launch<2>(a, stream);
    } else {
        if (cb == 0) tf_exl3x::grouped_launch<0>(a, stream);
        else if (cb == 1) tf_exl3x::grouped_launch<1>(a, stream);
        else tf_exl3x::grouped_launch<2>(a, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_dequant_cuda(const at::Tensor& T, at::Tensor& out, int64_t K, int64_t N, int64_t k2, int64_t cb,
                        int64_t wmask) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    auto o = reinterpret_cast<half*>(out.data_ptr());
    if (cb == 0) tf_exl3x::dequant_launch<0>(t, o, (int)K, (int)N, (int)k2, (uint32_t)wmask, stream);
    else if (cb == 1) tf_exl3x::dequant_launch<1>(t, o, (int)K, (int)N, (int)k2, (uint32_t)wmask, stream);
    else tf_exl3x::dequant_launch<2>(t, o, (int)K, (int)N, (int)k2, (uint32_t)wmask, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// A kernel's static shared memory, from the compiled kernel: the dynamic size plus this must stay under 48 KiB, or
// the launch needs the opt-in. Each caller keeps its kernel's value in a function-local static.
static size_t static_smem(const void* kernel) {
    cudaFuncAttributes attributes;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    return (size_t)attributes.sharedSizeBytes;
}

void exl3x_group_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members, int64_t R,
                      int64_t slots, int64_t E, bool pdl, int* items, int* icount, int64_t tile) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    const size_t smem = (size_t)R * slots * sizeof(int);
    static const size_t static_bytes = static_smem((const void*)group_kernel);    // warp_tot and item_tot
    if (smem + static_bytes > 48 * 1024) {
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, group_kernel));
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const size_t limit = device->sharedMemPerBlockOptin - attributes.sharedSizeBytes;
        TORCH_CHECK(smem <= limit, "EXL3 grouping needs ", smem, " dynamic shared-memory bytes; this GPU allows ",
                    limit, " after the kernel's static storage");
        if (smem > (size_t)attributes.maxDynamicSharedSizeBytes)
            C10_CUDA_CHECK(cudaFuncSetAttribute(group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)limit));
    }
    tf_exl3_pdl::launch(group_kernel, dim3(1), dim3(GROUP_THREADS), smem, at::cuda::getCurrentCUDAStream(),
                        tf_exl3_pdl::enabled(pdl), static_cast<const int*>(pick.data_ptr<int>()), uids.data_ptr<int>(),
                        ucount.data_ptr<int>(), members.data_ptr<int>(), (int)R, (int)slots, (int)E,
                        (int)members.size(1), items, icount, (int)tile);
}

// group_scan_kernel's dynamic shared memory: counts and places (int), and 32 warps' offsets (uint16) an expert.
static size_t scan_smem(int64_t E) { return (size_t)E * (2 * sizeof(int) + 32 * sizeof(unsigned short)); }

void exl3x_group_scan_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members,
                           int64_t R, int64_t slots, int64_t E, bool pdl, int* items, int* icount, int64_t tile) {
    TORCH_CHECK(E >= 1 && E <= SCAN_THREADS, "the scan grouping holds at most ", SCAN_THREADS, " experts");
    TORCH_CHECK(R * slots < 65536, "the scan grouping ranks at most 65,535 picks");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    const size_t smem = scan_smem(E);
    static const size_t static_bytes = static_smem((const void*)group_scan_kernel);    // 264: 680-682 experts opt in
    if (smem + static_bytes > 48 * 1024) {
        // raise only, to the device's ceiling (playbook 6.5): a smaller size never lowers what a larger one set
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, group_scan_kernel));
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const size_t limit = device->sharedMemPerBlockOptin - attributes.sharedSizeBytes;
        TORCH_CHECK(smem <= limit, "EXL3 scan grouping needs ", smem, " dynamic shared-memory bytes; this GPU allows ",
                    limit, " after the kernel's static storage");
        if (smem > (size_t)attributes.maxDynamicSharedSizeBytes)
            C10_CUDA_CHECK(cudaFuncSetAttribute(group_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                (int)limit));
    }
    const unsigned blocks = (unsigned)std::min<int64_t>(SCAN_BLOCKS, E);
    tf_exl3_pdl::launch(group_scan_kernel, dim3(blocks), dim3(SCAN_THREADS), smem, at::cuda::getCurrentCUDAStream(),
                        tf_exl3_pdl::enabled(pdl), static_cast<const int*>(pick.data_ptr<int>()), uids.data_ptr<int>(),
                        ucount.data_ptr<int>(), members.data_ptr<int>(), (int)R, (int)slots, (int)E,
                        (int)members.size(1), items, icount, (int)tile);
}

void exl3x_rot_in_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                       const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, int64_t rows, int64_t K,
                       int64_t slots, int64_t E, bool pdl) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool early = tf_exl3_pdl::enabled(pdl);
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    const int* pk = pick.data_ptr<int>();
    if (x.scalar_type() == at::kBFloat16)
        tf_exl3_pdl::launch(rot_in_kernel<__nv_bfloat16>, grid, dim3(32), 0, stream, early,
                            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x_stride, pk, s0, s1, o0, o1,
                            (int)K, (int)slots, (int)E);
    else
        tf_exl3_pdl::launch(rot_in_kernel<half>, grid, dim3(32), 0, stream, early,
                            reinterpret_cast<const half*>(x.data_ptr()), (int)x_stride, pk, s0, s1, o0, o1, (int)K,
                            (int)slots, (int)E);
}

void exl3x_gateup_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g,
                                const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd, int64_t rows,
                                int64_t P, int64_t N, int64_t SK, int64_t slots, int64_t E, double limit,
                                int64_t act_mode, bool pdl) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    tf_exl3_pdl::launch(gateup_epilogue_kernel, grid, dim3(32), 0, at::cuda::getCurrentCUDAStream(),
                        tf_exl3_pdl::enabled(pdl), static_cast<const float*>(Z.data_ptr<float>()),
                        static_cast<const int*>(pick.data_ptr<int>()), reinterpret_cast<const half*>(svh_g.data_ptr()),
                        reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
                        reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)SK, (int)E, (float)limit,
                        (int)act_mode);
}

void exl3x_down_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E, bool pdl) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(D / 128));
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool early = tf_exl3_pdl::enabled(pdl);
    const float* z = Z.data_ptr<float>();
    const int* pk = pick.data_ptr<int>();
    auto sv = reinterpret_cast<const half*>(svh_d.data_ptr());
    if (y.scalar_type() == at::kBFloat16)
        tf_exl3_pdl::launch(down_epilogue_kernel<__nv_bfloat16>, grid, dim3(32), 0, stream, early, z, pk, sv,
                            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), (int)P, (int)D, (int)SK, (int)E);
    else
        tf_exl3_pdl::launch(down_epilogue_kernel<float>, grid, dim3(32), 0, stream, early, z, pk, sv,
                            y.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E);
}

void exl3x_combine_cuda(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t D,
                        int64_t slots, bool pdl) {
    dim3 grid((unsigned)rows, (unsigned)((D + 255) / 256));
    tf_exl3_pdl::launch(combine_kernel, grid, dim3(256), 0, at::cuda::getCurrentCUDAStream(),
                        tf_exl3_pdl::enabled(pdl), static_cast<const float*>(y.data_ptr<float>()),
                        static_cast<const float*>(wts.data_ptr<float>()), out.data_ptr<float>(), (int)D, (int)slots);
}

void exl3x_down_combine_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                             const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t P, int64_t D, int64_t SK,
                             int64_t slots, int64_t E, bool pdl) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    tf_exl3_pdl::launch(down_combine_kernel, grid, dim3((unsigned)(32 * slots)), 0, at::cuda::getCurrentCUDAStream(),
                        tf_exl3_pdl::enabled(pdl), static_cast<const float*>(Z.data_ptr<float>()),
                        static_cast<const int*>(pick.data_ptr<int>()), reinterpret_cast<const half*>(svh_d.data_ptr()),
                        y.data_ptr<float>(), static_cast<const float*>(wts.data_ptr<float>()), out.data_ptr<float>(),
                        (int)P, (int)D, (int)SK, (int)E, (int)slots);
}
