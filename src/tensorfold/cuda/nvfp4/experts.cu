// Grouped NVFP4 experts on the shared plan: per 16-input block acc = fma(P, e4m3 scale, acc) in block order, one
// pair an mma row, so no pair affects another; each (expert, matrix) scale multiplies once before the epilogue.
//
// Two kernel families compute those bits. nvfp4_expert_kernel (decode, short prompts) gives a warp one 32-column block
// and 16 pairs at a time, reading weights and inputs from global memory. nvfp4_expert_tile_kernel (prompt chunks of
// 1,024+ rows, 64-pair items) shares an item's input rows among a CTA's four warps through shared memory and serves up
// to four 16-row tiles from each decoded weight fragment. The tile kernel is ported from MiaAI-Lab's Qwen3.8 Flash Next
// recipe, patch 0014 (fn_experts_tile.cu, MIT licence), itself built on the TensorFold contributors' kernels.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <torch/extension.h>

#include <map>
#include <mutex>
#include <utility>

#include "../experts.cuh"

namespace {

constexpr int BLOCK4 = 36;           // uint4 a (32 columns, 32 inputs) block: a lane's code words, then the scales

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// bf16x2 of the e2m1 nibbles at bits [s, s + 4) and [16 + s, 20 + s): fields into bf16's, times 2^126 (exact).
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

template <int M>
struct Stage {
  uint4 w[M];          // the lane's words: n8 tiles 0-3, 32 inputs each
  uint4 s[M];          // the quad's scales, bytes [block][tile][column]
  uint2 xa[2], xb[2];  // rows gq and gq + 8: inputs 4t .. 4t + 3 of each 16-input block
};

template <int M>
__device__ __forceinline__ void load_stage(Stage<M>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * BLOCK4 + lane);
    st.s[m] = ld_nc(b + m * BLOCK4 + 32 + t);
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
    st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
  }
}

template <int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st, bool hi) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
        float(&a)[4] = acc[m][0][j];
        a[0] = fmaf(p[0], s0, a[0]);
        a[1] = fmaf(p[1], s1, a[1]);
        if (hi) {
          a[2] = fmaf(p[2], s0, a[2]);
          a[3] = fmaf(p[3], s1, a[3]);
        }
      }
    }
}

template <int M>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  constexpr int D = 2;
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M>(acc, st[d], v1);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p; items of expert ``skip`` are left
// alone (the shared expert, run elsewhere), and an item's pairs go 16 at a time.
template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
      const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
      const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
      k_loop<M>(acc, blk, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
      }
      epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

template <int M, int EPI>
void launch(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale, int kg,
            int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out,
            int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_kernel<M, EPI, WARPS>, WARPS * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_kernel<M, EPI, WARPS><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// -- the shared-memory tile (prompt chunks) ---------------------------------------------------------------------
// Every stored element gets nvfp4_expert_kernel's arithmetic: for each 32-input group g and 16-input half h,
// p = mma(x, q) from a +0.0 accumulator, then acc = fmaf(p, e4m3 scale, acc) in (g, h) order, then acc *= the
// (expert, matrix) fp32 scale and the same epilogue. A row of an m16n8k16 is its own dot, so other pairs' rows, and
// stale shared-memory rows past an item's end that nobody stores, cannot change a stored value.
namespace tile {

constexpr int WARPS = 4;                       // a CTA: four 32-column blocks of one item
constexpr int XBYTES_PER_TILE = 2 * 32 * 16;   // a 16-row tile's A fragments for one K32 group: (half, lane) uint4s
constexpr int XBYTES = 4 * XBYTES_PER_TILE;    // room for four row tiles

template <int M, int S>
constexpr int smem_bytes() { return S * (XBYTES + WARPS * M * BLOCK4 * 16); }

__device__ __forceinline__ void cp4(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}

// experts.cuh's mma with a fresh +0.0 accumulator of its own (the same instruction)
__device__ __forceinline__ void mma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

template <int M>
struct WStage {
  uint4 w[M];
  uint4 s[M];
};

// One K32 group of a warp from shared memory, software pipelined: step f's mmas issue before step f - 1's fmas, and
// each acc element still gets its fmaf in (g, h) order.
template <int M, int RT>
__device__ __forceinline__ void compute(float (&acc)[M][RT][NTW][4], const WStage<M>& st, const char* xs, int lane) {
  uint4 xf[RT][2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      xf[r][h] = *reinterpret_cast<const uint4*>(xs + r * XBYTES_PER_TILE + (h * 32 + lane) * 16);
  constexpr int NF = M * 2 * NTW;
  float pp[2][RT][4];
  float sc[2][2];
#pragma unroll
  for (int f = 0; f <= NF; ++f) {
    if (f < NF) {
      const int m = f / (2 * NTW), h = (f / NTW) & 1, j = f % NTW;
      const uint32_t word = comp(st.w[m], j);
      const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
      const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
      const int sh = (j & 1) * 16;
      sc[f & 1][0] = e4m3f((sw >> sh) & 0xFFu);
      sc[f & 1][1] = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
      for (int r = 0; r < RT; ++r) mma0(pp[f & 1][r], xf[r][h].x, xf[r][h].y, xf[r][h].z, xf[r][h].w, b0, b1);
    }
    if (f > 0) {
      const int g = f - 1, m = g / (2 * NTW), j = g % NTW;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        float(&a)[4] = acc[m][r][j];
        a[0] = fmaf(pp[g & 1][r][0], sc[g & 1][0], a[0]);
        a[1] = fmaf(pp[g & 1][r][1], sc[g & 1][1], a[1]);
        a[2] = fmaf(pp[g & 1][r][2], sc[g & 1][0], a[2]);
        a[3] = fmaf(pp[g & 1][r][3], sc[g & 1][1], a[3]);
      }
    }
  }
}

// RT row tiles of an item, pairs r0 .. r0 + 16 * RT: X rows and each warp's weight blocks go through an S-slot ring of
// shared memory, cp.async filling S - 1 K32 groups ahead.
template <int M, int EPI, int RT, int S>
__device__ __forceinline__ void chunk(char* smem, const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                      const uint4* blk, bool active, const float* __restrict__ scale, int e, int KG,
                                      const int* __restrict__ members, int first, int cnt, int r0, void* out, int N,
                                      int cb, float limit, int tid, int lane, int gq, int t, int wp) {
  constexpr int NT = WARPS * 32, CP = (RT * 256 + NT - 1) / NT;
  constexpr int WSLAB = M * BLOCK4 * 16, STRIDE = XBYTES + WARPS * WSLAB;
  // thread -> (row rr, 4-byte word w of the K32 slice) of the RT * 16 rows, placed in mma A-fragment order: the
  // fragment (a0, a1, a2, a3) = (row gq inputs 4t, 4t + 1; row gq + 8 the same; row gq 4t + 2, 4t + 3; row gq + 8)
  const __nv_bfloat16* csrc[CP];
  int cdst[CP];
  bool cok[CP];
#pragma unroll
  for (int i = 0; i < CP; ++i) {
    const int idx = tid + i * NT, rr = idx >> 4, w = idx & 15;
    cok[i] = idx < RT * 256 && r0 + rr < cnt;
    const int pr = cok[i] ? __ldg(members + first + r0 + rr) : 0;
    csrc[i] = X + (size_t)(slots ? pr / slots : pr) * x_stride + 2 * w;
    const int lane2 = (rr & 7) * 4 + ((w & 7) >> 1);
    cdst[i] = (rr >> 4) * XBYTES_PER_TILE + ((w >> 3) * 32 + lane2) * 16 + ((w & 1) * 2 + ((rr >> 3) & 1)) * 4;
  }
  auto stage = [&](int g, int slot) {
    char* base = smem + slot * STRIDE;
#pragma unroll
    for (int i = 0; i < CP; ++i)
      if (cok[i]) cp4(base + cdst[i], csrc[i] + g * 32);
    if (active) {
      const uint4* src = blk + (size_t)g * (M * BLOCK4);
      char* dst = base + XBYTES + wp * WSLAB;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        cp16(dst + (m * BLOCK4 + lane) * 16, src + m * BLOCK4 + lane);
        if (lane < 4) cp16(dst + (m * BLOCK4 + 32 + lane) * 16, src + m * BLOCK4 + 32 + lane);
      }
    }
  };
  bool v0[RT], v1[RT];
  int pr0[RT], pr1[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const int b = r0 + 16 * r;
    v0[r] = b + gq < cnt;
    v1[r] = b + gq + 8 < cnt;
    pr0[r] = v0[r] ? __ldg(members + first + b + gq) : 0;
    pr1[r] = v1[r] ? __ldg(members + first + b + gq + 8) : 0;
  }
  float acc[M][RT][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    if (s < KG) stage(s, s);
    cp_commit();
  }
  for (int g = 0; g < KG; ++g) {
    cp_wait<S - 2>();
    __syncthreads();  // group g landed for every thread, and slot (g - 1) % S is free to refill
    if (g + S - 1 < KG) stage(g + S - 1, (g + S - 1) % S);
    cp_commit();
    if (active) {
      const char* base = smem + (g % S) * STRIDE;
      const char* wsm = base + XBYTES + wp * WSLAB;
      WStage<M> st;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        st.w[m] = *reinterpret_cast<const uint4*>(wsm + (m * BLOCK4 + lane) * 16);
        st.s[m] = *reinterpret_cast<const uint4*>(wsm + (m * BLOCK4 + 32 + t) * 16);
      }
      compute<M, RT>(acc, st, base, lane);
    }
  }
  cp_wait<0>();
  __syncthreads();  // the ring is free for the next chunk
  if (!active) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float gs = __ldg(scale + e * M + m);
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= gs;
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
    epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
}

}  // namespace tile

// A unit is (item, group of four 32-column blocks), one CTA; an item's pairs go RTMAX * 16 at a time, its tail in 3,
// 2 or 1 row tiles. Pair p reads X row p / slots when slots > 0, else row p; items of expert ``skip`` are left alone.
template <int M, int EPI, int RTMAX, int S>
__global__ void __launch_bounds__(tile::WARPS * 32)
    nvfp4_expert_tile_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                             const int* __restrict__ items, const int* __restrict__ counts,
                             const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  constexpr int WARPS = tile::WARPS;
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x, lane = tid & 31, gq = lane >> 2, t = lane & 3, wp = tid >> 5;
  const int groups = (NB + WARPS - 1) / WARPS;
  const int units = __ldg(counts) * groups;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;  // the whole CTA skips together: its barriers stay matched
    const int cb = cg * WARPS + wp;
    const bool active = cb < NB;  // a warp past the last block still stages X rows and meets every barrier
    const uint4* blk = W + ((size_t)e * NB + (active ? cb : 0)) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt;) {
      const int rt = (cnt - r0 + 15) >> 4;
      if (RTMAX >= 4 && rt >= 4) {
        if constexpr (RTMAX >= 4)
          tile::chunk<M, EPI, 4, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out,
                                    N, cb, limit, tid, lane, gq, t, wp);
        r0 += 64;
      } else if (RTMAX >= 3 && rt == 3) {
        if constexpr (RTMAX >= 3)
          tile::chunk<M, EPI, 3, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out,
                                    N, cb, limit, tid, lane, gq, t, wp);
        r0 += 48;
      } else if (RTMAX >= 2 && rt >= 2) {
        if constexpr (RTMAX >= 2)
          tile::chunk<M, EPI, 2, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out,
                                    N, cb, limit, tid, lane, gq, t, wp);
        r0 += 32;
      } else {
        tile::chunk<M, EPI, 1, S>(smem, X, x_stride, slots, blk, active, scale, e, KG, members, first, cnt, r0, out,
                                  N, cb, limit, tid, lane, gq, t, wp);
        r0 += 16;
      }
    }
  }
}

template <int M, int EPI, int RTMAX, int S>
void launch_tile(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale, int kg,
                 int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out,
                 int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = tile::WARPS;
  constexpr int SMEM = tile::smem_bytes<M, S>();
  // Both configs (34,816 and 19,200 bytes) launch under the default 48 KiB of dynamic shared memory, without the
  // cudaFuncSetAttribute opt-in. A config past it needs that opt-in, raised only, as 6f19c352 does for IQ2_XXS.
  static_assert(SMEM <= 48 * 1024, "nvfp4 expert tile: this config needs more than 48 KiB of shared memory");
  const auto kernel = nvfp4_expert_tile_kernel<M, EPI, RTMAX, S>;
  static std::mutex lock;
  static std::map<std::pair<int, const void*>, int> blocks_per_sm;
  const int device = x.get_device();
  const auto key = std::make_pair(device, reinterpret_cast<const void*>(kernel));
  int per_sm = 0;
  {
    const std::lock_guard<std::mutex> hold(lock);
    const auto hit = blocks_per_sm.find(key);
    if (hit != blocks_per_sm.end()) {
      per_sm = hit->second;
    } else {
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, WARPS * 32, SMEM));
      per_sm = per_sm < 1 ? 1 : per_sm;
      blocks_per_sm.emplace(key, per_sm);
    }
  }
  const int64_t cap = (int64_t)per_sm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = static_cast<int>(max_units < cap ? max_units : cap);
  if (grid < 1) return;
  kernel<<<grid, WARPS * 32, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void nvfp4_experts_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                        const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                        const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n, double limit,
                        int64_t skip, int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 2) launch<2, 2>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch<1, 0>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch<1, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else TORCH_CHECK(false, "nvfp4 experts: epilogue 0 (fp32 down), 2 (SwiGLU) or 3 (bf16 down), not ", epi);
}

// The tile family, one unit an (item, four column blocks): gate/up four row tiles and four stages (Mia's rt4_s4), down
// two row tiles and three stages (rt2_s3), her measured defaults.
void nvfp4_experts_tile_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                             const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                             const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                             double limit, int64_t skip, int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 2) launch_tile<2, 2, 4, 4>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch_tile<1, 0, 2, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch_tile<1, 3, 2, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else TORCH_CHECK(false, "nvfp4 expert tile: epilogue 0 (fp32 down), 2 (SwiGLU) or 3 (bf16 down), not ", epi);
}
