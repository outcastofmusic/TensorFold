// IQ2_XXS matmuls on ggml blocks as the checkpoint stores them: 66 bytes for 256 weights, fp16 d then 32 uint16 qs.
//
// Decoding follows llama.cpp's dequantize_row_iq2_xxs (ggml/src/ggml-quants.c:2489, llama.cpp 23f7aabc): a 32-weight
// sub-block is 8 bytes, aux0 holding four iq2xxs_grid indices and aux1 four 7-bit ksigns_iq2xs indices (bits 0..27)
// and a 4-bit scale s (bits 28..31); weight j of grid entry l is d * (0.5 + s) * 0.25 * grid[l][j], negated when the
// sign byte has bit j. Every grid byte is 8, 25 or 43, so a sub-block has three magnitudes; each is rounded to bf16
// once (the fp32 product is exact, as in the reference), and a pair of weights is one prmt from those three plus an
// xor of its two sign bits. iq2xxs.py builds the tables (a grid entry's four prmt selectors, a sign index's four
// pair masks) from the grid.
//
// Both kernels run mma.m16n8k16 with x rows as A and weight rows (output columns) as B, accumulating in fp32. The k
// slots of an mma need not be consecutive inputs, only the same inputs on both sides: lane (gq, t) owns sub-block t of
// each 128-input group, so it decodes that sub-block's scale once and a grid entry (8 weights) a lookup, and its x
// fragment is 8 contiguous bytes at the same inputs. An mma step s = 2 l + h of a group takes weights 8 l + 4 h ..
// + 3 of every lane's sub-block.
//
// decode: four warps a block, 16 columns of one expert, the K groups dealt to the warps in turn; the 16 weight rows are
//   staged whole in shared memory (16-byte loads, coalesced), and so are the item's x rows when they fit in 8 KB. A
//   grouped call has a block per row: the first row of each expert gathers every row with that expert, the others
//   exit, so an expert's weights are read once a call.
// prefill: 8 warps, 64 rows of one expert x 256 columns, a 128-input group a stage; x tiles go through shared memory
//   (cp.async, two stages, rows swizzled), each warp decodes its 32 columns straight from global, a group ahead.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

constexpr int QK = 256;            // weights a block
constexpr int BLOCK_U16 = 33;      // a block's 66 bytes as uint16: d, then qs[32]
constexpr int SEL_WORDS = 1024;    // selectors [256][4]
constexpr int SIGN_WORDS = 512;    // pair masks [128][4]
constexpr int TABLE_WORDS = SEL_WORDS + SIGN_WORDS;   // one variant; the second has its pairs in the order 2, 3, 0, 1

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// prmt without the nibble masking __byte_perm adds: the selectors hold byte indices 0..5 only
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t bf16_bits(float v) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(v)));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// One sub-block of one weight row: its two words and its three bf16 magnitudes as prmt's source bytes.
struct Sub {
  uint32_t aux0, aux1, lo, hi;
};

__device__ __forceinline__ Sub make_sub(uint32_t d16, uint32_t aux0, uint32_t aux1) {
  const float db = __half2float(__ushort_as_half(static_cast<unsigned short>(d16))) * (0.5f + (float)(aux1 >> 28)) *
                   0.25f;
  return {aux0, aux1, pack_bf16(db * 8.f, db * 25.f), bf16_bits(db * 43.f)};
}

// sub-block ib of the block at blk (uint16, 2-byte aligned), plain loads (shared memory or global)
__device__ __forceinline__ Sub load_sub(const uint16_t* blk, int ib) {
  const uint16_t* q = blk + 1 + 4 * ib;
  return make_sub(blk[0], static_cast<uint32_t>(q[0]) | static_cast<uint32_t>(q[1]) << 16,
                  static_cast<uint32_t>(q[2]) | static_cast<uint32_t>(q[3]) << 16);
}

// Grid entry l of a sub-block as four bf16 pairs, in the table variant's pair order.
__device__ __forceinline__ uint4 entry(const Sub& s, int l, const uint32_t* tab) {
  const uint4 p = reinterpret_cast<const uint4*>(tab)[(s.aux0 >> (8 * l)) & 255];
  const uint4 g = reinterpret_cast<const uint4*>(tab + SEL_WORDS)[(s.aux1 >> (7 * l)) & 127];
  return make_uint4(prmt(s.lo, s.hi, p.x) ^ g.x, prmt(s.lo, s.hi, p.y) ^ g.y, prmt(s.lo, s.hi, p.z) ^ g.z,
                    prmt(s.lo, s.hi, p.w) ^ g.w);
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<uint32_t*>(p) = pack_bf16(a, b);
}

// ---------------------------------------------------------------------------------------------------------------
// decode: blockIdx.x a 16-column tile, blockIdx.y an item (a grouped call: row y's expert; a linear call: one item).
// Warp w takes the 128-input groups w, w + 4, ...; warp 0 adds the four partial sums in warp order. Dynamic shared
// memory: the 16 weight rows, then DECODE_XS bytes for the item's x rows when they fit.

constexpr int DECODE_COLS = 16, DWARPS = 4, DECODE_XS = 8192, XPAD = 16;   // XPAD bf16 between staged x rows

// The decode loop over warp w's groups: x rows at xa / xb (shared memory when STAGED), 16 columns at col0 / col1.
template <int MT, bool STAGED>
__device__ __forceinline__ void decode_loop(float (&acc)[MT][2][4], const __nv_bfloat16* const (&xa)[MT],
                                            const __nv_bfloat16* const (&xb)[MT], int mts, const uint16_t* col0,
                                            const uint16_t* col1, int groups, int warp, int t, const uint32_t* tab) {
#pragma unroll 1
  for (int g = warp; g < groups; g += DWARPS) {
    const int sb = 4 * g + t, b = sb >> 3, ib = sb & 7;
    const Sub s0 = load_sub(col0 + b * BLOCK_U16, ib), s1 = load_sub(col1 + b * BLOCK_U16, ib);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint4 e0 = entry(s0, l, tab), e1 = entry(s1, l, tab);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int k = 128 * g + 8 * l + 4 * h;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m >= mts) break;
          uint2 xl, xh;
          if constexpr (STAGED) {
            xl = *reinterpret_cast<const uint2*>(xa[m] + k);
            xh = *reinterpret_cast<const uint2*>(xb[m] + k);
          } else {
            xl = __ldg(reinterpret_cast<const uint2*>(xa[m] + k));
            xh = __ldg(reinterpret_cast<const uint2*>(xb[m] + k));
          }
          mma(acc[m][0], xl.x, xh.x, xl.y, xh.y, h ? e0.z : e0.x, h ? e0.w : e0.y);
          mma(acc[m][1], xl.x, xh.x, xl.y, xh.y, h ? e1.z : e1.x, h ? e1.w : e1.y);
        }
      }
    }
  }
}

template <int MT>
__global__ void __launch_bounds__(DWARPS * 32)
    decode_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W, const int* __restrict__ ids,
                  int R, int N, int K, const uint32_t* __restrict__ gtab, __nv_bfloat16* __restrict__ out) {
  extern __shared__ uint4 dyn[];
  __shared__ __align__(16) uint32_t tab[TABLE_WORDS];
  __shared__ int rows[16 * MT];
  __shared__ int s_nrows;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  const int e = ids != nullptr ? __ldg(ids + blockIdx.y) : 0;
  if (warp == 0) {
    int nrows = R;
    if (ids != nullptr) {
      const int r0 = blockIdx.y;
      bool earlier = false;
      for (int r = lane; r < r0; r += 32) earlier |= __ldg(ids + r) == e;
      if (__any_sync(0xffffffffu, earlier)) {
        nrows = -1;                                                // an earlier row's block has this expert
      } else {
        nrows = 0;
        for (int base = 0; base < R; base += 32) {
          const int r = base + lane;
          const bool mine = r < R && __ldg(ids + r) == e;
          const unsigned votes = __ballot_sync(0xffffffffu, mine);
          if (mine) rows[nrows + __popc(votes & ((1u << lane) - 1u))] = r;
          nrows += __popc(votes);
        }
      }
    } else {
      for (int r = lane; r < R; r += 32) rows[r] = r;
    }
    if (lane == 0) s_nrows = nrows;
  }
  for (int i = tid; i < TABLE_WORDS / 4; i += DWARPS * 32)
    reinterpret_cast<uint4*>(tab)[i] = __ldg(reinterpret_cast<const uint4*>(gtab) + i);
  __syncthreads();
  const int nrows = s_nrows;
  if (nrows < 0) return;
  const int kb = K / QK, rb = kb * BLOCK_U16;                      // a weight row in uint16
  const int n0 = blockIdx.x * DECODE_COLS, ncols = min(DECODE_COLS, N - n0);
  const uint4* src = reinterpret_cast<const uint4*>(W + ((size_t)e * N + n0) * (size_t)(kb * 66));
  const int units = ncols * kb * 66 / 16;                          // ncols is 8 or 16: 528 kb bytes or twice that
  for (int i = tid; i < units; i += DWARPS * 32) cp16(dyn + i, src + i);
  uint4* xs = dyn + DECODE_COLS * kb * 66 / 16;
  const int xstride = K + XPAD;
  const bool staged = nrows * xstride * 2 <= DECODE_XS;
  if (staged) {
    const int per_row = K / 8;
    for (int i = tid; i < nrows * per_row; i += DWARPS * 32) {
      const int r = i / per_row, c = i - r * per_row;
      cp16(xs + r * (xstride / 8) + c, X + (size_t)rows[r] * K + 8 * c);
    }
  }
  cp_commit();
  // rows past nrows read row 0 (finite or not, their sums are never stored)
  const __nv_bfloat16* xa[MT];
  const __nv_bfloat16* xb[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int ra = 16 * m + gq < nrows ? 16 * m + gq : 0, rb8 = 16 * m + gq + 8 < nrows ? 16 * m + gq + 8 : 0;
    if (staged) {
      xa[m] = reinterpret_cast<const __nv_bfloat16*>(xs) + ra * xstride + 32 * t;
      xb[m] = reinterpret_cast<const __nv_bfloat16*>(xs) + rb8 * xstride + 32 * t;
    } else {
      xa[m] = X + (size_t)rows[ra] * K + 32 * t;
      xb[m] = X + (size_t)rows[rb8] * K + 32 * t;
    }
  }
  const int mts = (nrows + 15) >> 4;
  float acc[MT][2][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
  cp_wait<0>();
  __syncthreads();
  const uint16_t* col0 = reinterpret_cast<const uint16_t*>(dyn) + gq * rb;
  const uint16_t* col1 = ncols == DECODE_COLS ? col0 + 8 * rb : col0;   // an 8-column tail decodes column gq twice
  if (staged) {
    decode_loop<MT, true>(acc, xa, xb, mts, col0, col1, K / 128, warp, t, tab);
  } else {
    decode_loop<MT, false>(acc, xa, xb, mts, col0, col1, K / 128, warp, t, tab);
  }
  __syncthreads();                                                 // dyn now holds the partial sums
  float* part = reinterpret_cast<float*>(dyn);
  constexpr int PART = MT * 2 * 4;                                 // floats a lane
  if (warp > 0) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) part[((warp - 1) * PART + (m * 2 + j) * 4 + q) * 32 + lane] = acc[m][j][q];
  }
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m >= mts) break;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if (8 * j >= ncols) break;
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int w = 1; w < DWARPS; ++w) acc[m][j][q] += part[((w - 1) * PART + (m * 2 + j) * 4 + q) * 32 + lane];
      const int col = n0 + 8 * j + 2 * t, ra = 16 * m + gq;
      if (ra < nrows) store2(out + (size_t)rows[ra] * N + col, acc[m][j][0], acc[m][j][1]);
      if (ra + 8 < nrows) store2(out + (size_t)rows[ra + 8] * N + col, acc[m][j][2], acc[m][j][3]);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// prefill: blockIdx.x a 64-row tile of one expert, blockIdx.y 256 columns. A grouped call's rows are read through
// ``order`` (rows sorted by expert); tile_end[e] is the inclusive sum of tiles over experts 0..e, starts[e] the first
// sorted row of expert e (E + 1 entries). A linear call passes no order: one expert, tile i is rows 64 i ...
//
// x's stage row is 16 uint4 (128 inputs), chunk c stored at c ^ (row & 7). Lanes with t >= 2 read the table variant
// whose pairs come in the order 2, 3, 0, 1 and take the other half of each grid entry first, so the 32 lanes of an
// x load hit 16 distinct 8-byte bank pairs, two lanes each (the 256 bytes' minimum of two wavefronts).

constexpr int PM = 64, PWARPS = 8, PNT = 4, PCOLS = PWARPS * 8 * PNT, PSTAGES = 2;
constexpr int XROW_U4 = 16;
constexpr int XSTAGE_U4 = PM * XROW_U4;

struct Raw {
  uint32_t d, aux0, aux1;
};

__global__ void __launch_bounds__(PWARPS * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
                   const int* __restrict__ order, const int* __restrict__ tile_end, const int* __restrict__ starts,
                   int E, int R, int N, int K, const uint32_t* __restrict__ gtab, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) uint32_t tab[2 * TABLE_WORDS];
  __shared__ __align__(16) uint4 xs[PSTAGES * XSTAGE_U4];
  const int tile = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, t = lane & 3;
  int e = 0, first, cnt;
  if (order == nullptr) {
    first = tile * PM;
    cnt = min(PM, R - first);
    if (cnt <= 0) return;
  } else {
    if (tile >= __ldg(tile_end + E - 1)) return;
    int lo = 0, hi = E - 1;                                      // the first expert whose tile_end passes this tile
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (__ldg(tile_end + mid) > tile) hi = mid; else lo = mid + 1;
    }
    e = lo;
    const int s0 = __ldg(starts + e), count = __ldg(starts + e + 1) - s0;
    const int local = tile - (__ldg(tile_end + e) - (count + PM - 1) / PM);
    first = s0 + local * PM;
    cnt = min(PM, count - local * PM);
  }
  for (int i = tid; i < 2 * TABLE_WORDS / 4; i += PWARPS * 32)
    reinterpret_cast<uint4*>(tab)[i] = __ldg(reinterpret_cast<const uint4*>(gtab) + i);
  // x loads: thread tid copies chunk tid % 16 of rows tid / 16 + 16 i, i < 4, each stage
  const int xc = tid & 15, xr0 = tid >> 4;
  const __nv_bfloat16* xsrc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = xr0 + 16 * i;
    xsrc[i] = r < cnt ? X + (size_t)(order ? __ldg(order + first + r) : first + r) * K + 8 * xc : nullptr;
  }
  const int groups = K / 128;
  auto stage = [&](int s, int g) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = xr0 + 16 * i;
      if (xsrc[i]) cp16(xs + s * XSTAGE_U4 + r * XROW_U4 + (xc ^ (r & 7)), xsrc[i] + 128 * g);
    }
  };
  stage(0, 0);
  cp_commit();
  const int kb = K / QK, rb = kb * BLOCK_U16;
  const int c0 = blockIdx.y * PCOLS + warp * 8 * PNT;
  const uint16_t* wcol[PNT];
#pragma unroll
  for (int j = 0; j < PNT; ++j) {
    const int n = min(c0 + 8 * j + gq, N - 1);                  // a column past N decodes column N - 1, unstored
    wcol[j] = reinterpret_cast<const uint16_t*>(W) + ((size_t)e * N + n) * (size_t)rb;
  }
  const bool active = c0 < N;
  const int v = t >> 1;                                          // this lane's table variant and first half
  const uint32_t* mytab = tab + v * TABLE_WORDS;
  Raw raw[PNT];                                                  // the lane's sub-block of group g, a group ahead
  auto fetch = [&](int g) {
    const int sb = 4 * g + t, b = sb >> 3, ib = sb & 7;
#pragma unroll
    for (int j = 0; j < PNT; ++j) {
      const uint16_t* blk = wcol[j] + b * BLOCK_U16;
      const uint16_t* q = blk + 1 + 4 * ib;
      raw[j].d = __ldg(blk);
      raw[j].aux0 = static_cast<uint32_t>(__ldg(q)) | static_cast<uint32_t>(__ldg(q + 1)) << 16;
      raw[j].aux1 = static_cast<uint32_t>(__ldg(q + 2)) | static_cast<uint32_t>(__ldg(q + 3)) << 16;
    }
  };
  float acc[PM / 16][PNT][4];
#pragma unroll
  for (int m = 0; m < PM / 16; ++m)
#pragma unroll
    for (int j = 0; j < PNT; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
  const int mts = (cnt + 15) >> 4;
  if (active) fetch(0);
  for (int g = 0; g < groups; ++g) {
    cp_wait<0>();
    __syncthreads();                                             // stage g landed (and, at g = 0, the tables)
    if (g + 1 < groups) stage((g + 1) % PSTAGES, g + 1);
    cp_commit();
    if (active) {
      Sub sub[PNT];
#pragma unroll
      for (int j = 0; j < PNT; ++j) sub[j] = make_sub(raw[j].d, raw[j].aux0, raw[j].aux1);
      if (g + 1 < groups) fetch(g + 1);
      const uint4* xt = xs + (g % PSTAGES) * XSTAGE_U4;
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        uint4 ev[PNT];
#pragma unroll
        for (int j = 0; j < PNT; ++j) ev[j] = entry(sub[j], l, mytab);
        const int chunk = 4 * t + l;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int half = h ^ v;
#pragma unroll
          for (int m = 0; m < PM / 16; ++m) {
            if (m >= mts) break;
            const int r = 16 * m + gq;
            const uint2 xl = reinterpret_cast<const uint2*>(xt + r * XROW_U4 + (chunk ^ (r & 7)))[half];
            const uint2 xh = reinterpret_cast<const uint2*>(xt + (r + 8) * XROW_U4 + (chunk ^ (r & 7)))[half];
#pragma unroll
            for (int j = 0; j < PNT; ++j)
              mma(acc[m][j], xl.x, xh.x, xl.y, xh.y, h ? ev[j].z : ev[j].x, h ? ev[j].w : ev[j].y);
          }
        }
      }
    }
    __syncthreads();                                             // stage g is read before stage g + 2 overwrites it
  }
  if (!active) return;
#pragma unroll
  for (int m = 0; m < PM / 16; ++m) {
    if (m >= mts) break;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = 16 * m + gq + 8 * h;
      if (r >= cnt) continue;
      const int orow = order ? __ldg(order + first + r) : first + r;
#pragma unroll
      for (int j = 0; j < PNT; ++j) {
        const int col = c0 + 8 * j + 2 * t;
        if (c0 + 8 * j < N) store2(out + (size_t)orow * N + col, acc[m][j][2 * h], acc[m][j][2 * h + 1]);
      }
    }
  }
}

}  // namespace

void iq2xxs_decode_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor* ids, int64_t items, int64_t n,
                        int64_t mt, const at::Tensor& tab, at::Tensor& out) {
  const at::cuda::CUDAGuard guard(x.device());
  const int R = x.size(0), K = x.size(1), N = n;
  const dim3 grid((N + DECODE_COLS - 1) / DECODE_COLS, items);
  const size_t smem = std::max((size_t)DECODE_COLS * (K / QK) * 66 + DECODE_XS,
                               (size_t)(DWARPS - 1) * mt * 2 * 4 * 32 * sizeof(float));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* ip = ids ? ids->data_ptr<int>() : nullptr;
  const auto* tp = reinterpret_cast<const uint32_t*>(tab.data_ptr<int>());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto go = [&](auto kernel) {
    if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kernel<<<grid, DWARPS * 32, smem, stream>>>(xp, w.data_ptr<uint8_t>(), ip, R, N, K, tp, op);
  };
  switch (mt) {
    case 1: go(decode_kernel<1>); break;
    case 2: go(decode_kernel<2>); break;
    case 3: go(decode_kernel<3>); break;
    default: go(decode_kernel<4>); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void iq2xxs_prefill_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor* order,
                         const at::Tensor* tile_end, const at::Tensor* starts, int64_t experts, int64_t tiles,
                         int64_t n, const at::Tensor& tab, at::Tensor& out) {
  const at::cuda::CUDAGuard guard(x.device());
  const int R = x.size(0), K = x.size(1), N = n;
  const dim3 grid(tiles, (N + PCOLS - 1) / PCOLS);
  prefill_kernel<<<grid, PWARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(),
      order ? order->data_ptr<int>() : nullptr, tile_end ? tile_end->data_ptr<int>() : nullptr,
      starts ? starts->data_ptr<int>() : nullptr, experts, R, N, K,
      reinterpret_cast<const uint32_t*>(tab.data_ptr<int>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
