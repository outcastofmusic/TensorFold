// IQ2_XXS matmuls on ggml blocks as the checkpoint stores them: 66 bytes for 256 weights, fp16 d then 32 uint16 qs.
//
// Decoding follows llama.cpp's dequantize_row_iq2_xxs (ggml/src/ggml-quants.c:2489, llama.cpp 23f7aabc): a 32-weight
// sub-block is 8 bytes, aux0 holding four iq2xxs_grid indices and aux1 four 7-bit ksigns_iq2xs indices (bits 0..27)
// and a 4-bit scale s (bits 28..31); weight j of grid entry l is d * (0.5 + s) * 0.25 * grid[l][j], negated when the
// sign byte has bit j. Every grid byte is 8, 25 or 43, so a sub-block has three magnitudes; each is rounded to bf16
// once (the fp32 product is exact, as in the reference), and a pair of weights is one prmt from those three plus an
// xor of its two sign bits. iq2xxs.py builds the one table, a grid entry's four prmt selectors; the sign bits come
// from the 7-bit index and its parity by a multiply.
//
// Both kernels run mma.m16n8k16 with x rows as A and weight rows (output columns) as B, accumulating in fp32. The k
// slots of an mma need not be consecutive inputs, only the same inputs on both sides: lane (gq, t) owns sub-block t of
// each 128-input group, so it decodes that sub-block's scale once and a grid entry (8 weights) a lookup, and its x
// fragment is 8 contiguous bytes at the same inputs. An mma step s = 2 l + h of a group takes weights 8 l + 4 h ..
// + 3 of every lane's sub-block.
//
// decode: a persistent grid of four-warp blocks over tiles of 16 columns of one distinct expert, each tile's weight
//   rows staged in shared memory a tile ahead, K split between the warps; an expert's rows are gathered once,
//   so its weights are read once a call.
// prefill: 4 warps, 64 rows of one expert x 128 columns, a 128-input group a stage; x tiles go through shared memory
//   (cp.async, two stages, rows swizzled), each warp decodes its 32 columns straight from global, two groups ahead.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int QK = 256;            // weights a block
constexpr int BLOCK_U16 = 33;      // a block's 66 bytes as uint16: d, then qs[32]
constexpr int TABLE_WORDS = 512;   // selectors [256][2], one variant; the second has its pairs in the order 2, 3, 0, 1
constexpr uint32_t SIGN_BITS = 0x80008000u;

// Sign bits 2p, 2p + 1 of v at bits 15 and 31: v * spread(p) & SIGN_BITS (the two shifted copies of v never overlap)
__host__ __device__ constexpr uint32_t spread(int p) { return 1u << (15 - 2 * p) | 1u << (30 - 2 * p); }

// the four pairs' multipliers in table variant v's order
__device__ __forceinline__ uint4 spreads(int v) {
  return v ? make_uint4(spread(2), spread(3), spread(0), spread(1))
           : make_uint4(spread(0), spread(1), spread(2), spread(3));
}

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

// sub-block ib of the block at blk (uint16, 2-byte aligned), from shared memory
__device__ __forceinline__ Sub load_sub(const uint16_t* blk, int ib) {
  const uint16_t* q = blk + 1 + 4 * ib;
  return make_sub(blk[0], static_cast<uint32_t>(q[0]) | static_cast<uint32_t>(q[1]) << 16,
                  static_cast<uint32_t>(q[2]) | static_cast<uint32_t>(q[3]) << 16);
}

// Grid entry l of a sub-block as four bf16 pairs, in the order of the table variant ``tab`` (shared memory) and of
// ``k`` (spreads(variant)). The sign byte is ksigns_iq2xs: the 7 stored bits
// and their parity.
__device__ __forceinline__ uint4 entry(const Sub& s, int l, const uint32_t* tab, const uint4& k) {
  const uint2 p = reinterpret_cast<const uint2*>(tab)[(s.aux0 >> (8 * l)) & 255];
  const uint32_t s7 = (s.aux1 >> (7 * l)) & 127, v = s7 | (__popc(s7) & 1) << 7;
  return make_uint4(prmt(s.lo, s.hi, p.x) ^ (v * k.x & SIGN_BITS), prmt(s.lo, s.hi, p.x >> 16) ^ (v * k.y & SIGN_BITS),
                    prmt(s.lo, s.hi, p.y) ^ (v * k.z & SIGN_BITS), prmt(s.lo, s.hi, p.y >> 16) ^ (v * k.w & SIGN_BITS));
}

struct Raw {
  uint32_t d, aux0, aux1;
};

// Sub-block sb of a weight row in global memory. Its 8 bytes are 2-byte aligned: two or three aligned words and a
// funnel shift (the third word can end 2 bytes past the tensor; CUDA allocations come in 512-byte multiples).
__device__ __forceinline__ Raw fetch_raw(const uint8_t* row, int sb) {
  const uint8_t* blk = row + 66 * (sb >> 3);
  const uintptr_t at = reinterpret_cast<uintptr_t>(blk + 2 + 8 * (sb & 7));
  const uint32_t* w = reinterpret_cast<const uint32_t*>(at & ~uintptr_t(3));
  const uint32_t sh = (at & 3) * 8, w0 = __ldg(w), w1 = __ldg(w + 1), w2 = sh ? __ldg(w + 2) : 0u;
  return {__ldg(reinterpret_cast<const uint16_t*>(blk)), __funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh)};
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<uint32_t*>(p) = pack_bf16(a, b);
}

// ---------------------------------------------------------------------------------------------------------------
// decode: a persistent grid. A tile is 16 columns of one distinct expert of the call (a linear call: one expert);
// block b takes tiles b, b + G, ... (G blocks), so the grid reads the weights in order, which ran 37.6 us against
// 39.8 us for contiguous runs of tiles a block (N 2560, K 1024, 9 experts, 2026-10-16). A tile's 16 weight rows are
// contiguous in memory and come in by cp.async a tile ahead, with the expert's x rows when they fit in DECODE_XS
// bytes (read from global otherwise); four warps split K by 128-input group (warp w takes groups w, w + 4, ...) and
// warp 0 adds the partial sums in warp order. Two stages beat three and four (more blocks fit an SM).

constexpr int DECODE_COLS = 16, DWARPS = 4, DSTAGES = 2, DECODE_XS = 6144, XPAD = 16;   // XPAD bf16 between x rows
constexpr int MAX_ROWS = 64;

template <int MT, bool STAGED>
__device__ __forceinline__ void decode_loop(float (&acc)[MT][2][4], const __nv_bfloat16* const (&xa)[MT],
                                            const __nv_bfloat16* const (&xb)[MT], int mts, const uint16_t* col0,
                                            const uint16_t* col1, int groups, int warp, int t, const uint32_t* tab) {
  const uint4 k = spreads(0);
#pragma unroll 2
  for (int g = warp; g < groups; g += DWARPS) {
    const int sb = 4 * g + t, b = sb >> 3, ib = sb & 7;
    const Sub s0 = load_sub(col0 + b * BLOCK_U16, ib), s1 = load_sub(col1 + b * BLOCK_U16, ib);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint4 e0 = entry(s0, l, tab, k), e1 = entry(s1, l, tab, k);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int k = 128 * g + 8 * l + 4 * h;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m >= mts) break;
          uint2 xl = make_uint2(0u, 0u), xh = xl;                // a missing row is zeros, loaded by no one
          if constexpr (STAGED) {
            if (xa[m]) xl = *reinterpret_cast<const uint2*>(xa[m] + k);
            if (xb[m]) xh = *reinterpret_cast<const uint2*>(xb[m] + k);
          } else {
            if (xa[m]) xl = __ldg(reinterpret_cast<const uint2*>(xa[m] + k));
            if (xb[m]) xh = __ldg(reinterpret_cast<const uint2*>(xb[m] + k));
          }
          mma(acc[m][0], xl.x, xh.x, xl.y, xh.y, h ? e0.z : e0.x, h ? e0.w : e0.y);
          mma(acc[m][1], xl.x, xh.x, xl.y, xh.y, h ? e1.z : e1.x, h ? e1.w : e1.y);
        }
      }
    }
  }
}

// A grouped call of more than MAX_ROWS rows gets its plan from plan_kernel, in global memory: plan[0] the number of
// distinct experts U, then uexp[E] (U used), ustart[E + 1] (U + 1 used), then urow[R], the rows sorted by expert.
__global__ void __launch_bounds__(1024) plan_kernel(const int* __restrict__ ids, int R, int E, int* __restrict__ plan) {
  extern __shared__ int sm[];                                    // count[E], then the scan [E]
  int* count = sm;
  int* scan = sm + E;
  const int tid = threadIdx.x;
  for (int e = tid; e < E; e += blockDim.x) count[e] = 0;
  __syncthreads();
  for (int r = tid; r < R; r += blockDim.x) atomicAdd(count + __ldg(ids + r), 1);
  __syncthreads();
  // inclusive scans of the counts and of the used flags, packed as (rows << 11 | used), E <= 1024 < 2^11
  for (int e = tid; e < E; e += blockDim.x) scan[e] = count[e] << 11 | (count[e] > 0);
  __syncthreads();
  for (int d = 1; d < E; d <<= 1) {
    int v[2] = {0, 0};
    for (int i = 0, e = tid; e < E; e += blockDim.x, ++i) v[i] = scan[e] + (e >= d ? scan[e - d] : 0);
    __syncthreads();
    for (int i = 0, e = tid; e < E; e += blockDim.x, ++i) scan[e] = v[i];
    __syncthreads();
  }
  int* uexp = plan + 1;
  int* ustart = plan + 1 + E;
  int* urow = plan + 2 + 2 * E;
  for (int e = tid; e < E; e += blockDim.x) {
    const int incl = scan[e], used = incl & 2047, first = (incl >> 11) - count[e];
    if (count[e] > 0) {
      uexp[used - 1] = e;
      ustart[used - 1] = first;
    }
    count[e] = first;                                            // now each expert's next free slot
  }
  if (tid == 0) {
    const int u = scan[E - 1] & 2047;
    plan[0] = u;
    ustart[u] = R;
  }
  __syncthreads();
  // an expert's rows land in any order: a row's sum does not depend on where it sits
  for (int r = tid; r < R; r += blockDim.x) urow[atomicAdd(count + __ldg(ids + r), 1)] = r;
}

// Dynamic shared memory: DSTAGES stages (16 weight rows and DECODE_XS bytes of x rows), then the partial sums. A row's
// sum takes the same steps whatever the call's other rows, so a verify window's rows equal serial steps' bit for bit.
// An expert with more than 16 MT rows takes them 16 MT at a time from the same staged weight rows.
template <int MT>
__global__ void __launch_bounds__(DWARPS * 32)
    decode_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W, const int* __restrict__ ids,
                  const int* __restrict__ plan, int R, int E, int N, int K, const uint32_t* __restrict__ gtab,
                  __nv_bfloat16* __restrict__ out) {
  extern __shared__ uint4 dyn[];
  __shared__ __align__(16) uint32_t tab[TABLE_WORDS];
  __shared__ int s_row[MAX_ROWS], s_start[MAX_ROWS + 1], s_exp[MAX_ROWS];
  __shared__ int s_nuniq;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  for (int i = tid; i < TABLE_WORDS / 4; i += DWARPS * 32) cp16(reinterpret_cast<uint4*>(tab) + i, gtab + 4 * i);
  cp_commit();
  const int* urow = s_row;
  const int* ustart = s_start;
  const int* uexp = s_exp;
  if (plan != nullptr) {
    uexp = plan + 1, ustart = plan + 1 + E, urow = plan + 2 + 2 * E;
    if (tid == 0) s_nuniq = __ldg(plan);
  } else if (warp == 0) {                // the call's distinct experts in order of first row, and each one's rows
    int nuniq = 1, pos = R;
    if (ids == nullptr) {                // a linear call: one expert, rows in order
      if (lane == 0) s_exp[0] = 0;
    } else {
      // R <= 64: lane l holds ids[l] and ids[l + 32]; a row leads when no earlier row has its expert
      nuniq = 0, pos = 0;
      const int id0 = lane < R ? __ldg(ids + lane) : -1, id1 = lane + 32 < R ? __ldg(ids + lane + 32) : -1;
      for (int r0 = 0; r0 < R; ++r0) {
        const int e = __shfl_sync(0xffffffffu, r0 < 32 ? id0 : id1, r0 & 31);
        const unsigned lo = __ballot_sync(0xffffffffu, id0 == e), hi = __ballot_sync(0xffffffffu, id1 == e);
        const unsigned long long same = (unsigned long long)hi << 32 | lo;
        if (__ffsll(same) - 1 != r0) continue;         // an earlier row has this expert
        if (lane == 0) {
          s_exp[nuniq] = e;
          s_start[nuniq] = pos;
        }
        if (id0 == e) s_row[pos + __popc(lo & ((1u << lane) - 1u))] = lane;
        if (id1 == e) s_row[pos + __popc(lo) + __popc(hi & ((1u << lane) - 1u))] = lane + 32;
        pos += __popcll(same);
        ++nuniq;
      }
    }
    if (lane == 0) {
      s_start[0] = 0;
      s_start[nuniq] = pos;
      s_nuniq = nuniq;
    }
  }
  __syncthreads();
  const bool linear = ids == nullptr;    // rows in order, any count
  auto row = [&](int i) { return linear ? i : urow[i]; };
  // block b takes tiles b, b + G, b + 2 G, ... (G = gridDim.x): at any moment the grid reads consecutive tiles
  const int coltiles = (N + DECODE_COLS - 1) / DECODE_COLS, total = s_nuniq * coltiles, G = gridDim.x;
  if ((int)blockIdx.x >= total) return;
  const int kb = K / QK, rb = kb * BLOCK_U16, slab_u4 = DECODE_COLS * kb * 66 / 16;
  const int stage_u4 = slab_u4 + DECODE_XS / 16;                 // a stage: the tile's weight rows, then its x rows
  float* part = reinterpret_cast<float*>(dyn + DSTAGES * stage_u4);
  const size_t rowb = (size_t)kb * 66;
  const int xstride = K + XPAD;
  auto span = [&](int u, int& first, int& nrows) {
    first = linear ? 0 : ustart[u];
    nrows = linear ? R : ustart[u + 1] - first;
  };
  auto issue = [&](int tile, int s) {    // tile's 16 rows (8 at an N tail), and its expert's x rows when they fit
    const int u = tile / coltiles, n0 = (tile - u * coltiles) * DECODE_COLS;
    const uint4* src = reinterpret_cast<const uint4*>(W + ((size_t)(linear ? 0 : uexp[u]) * N + n0) * rowb);
    const int units = min(DECODE_COLS, N - n0) * kb * 66 / 16;
    uint4* dst = dyn + s * stage_u4;
    for (int i = tid; i < units; i += DWARPS * 32) cp16(dst + i, src + i);
    int first, nrows;
    span(u, first, nrows);
    const int per_row = K / 8;
    if (nrows * xstride * 2 > DECODE_XS) return;
    for (int q = tid; q < nrows * per_row; q += DWARPS * 32) {
      const int r = q / per_row, c = q - r * per_row;
      cp16(dst + slab_u4 + r * (xstride / 8) + c, X + (size_t)row(first + r) * K + 8 * c);
    }
  };
#pragma unroll
  for (int p = 0; p < DSTAGES - 1; ++p) {
    if ((int)blockIdx.x + p * G < total) issue(blockIdx.x + p * G, p);
    cp_commit();
  }
  constexpr int CHUNK = 16 * MT;
  constexpr int PART = MT * 2 * 4;       // floats a lane
  const __nv_bfloat16* xa[MT];
  const __nv_bfloat16* xb[MT];
  for (int tile = blockIdx.x, i = 0; tile < total; tile += G, ++i) {
    const int s = i % DSTAGES;
    if (tile + (DSTAGES - 1) * G < total) issue(tile + (DSTAGES - 1) * G, (i + DSTAGES - 1) % DSTAGES);
    cp_commit();
    const int u = tile / coltiles, n0 = (tile - u * coltiles) * DECODE_COLS;
    int first, nrows;
    span(u, first, nrows);
    const bool staged = nrows * xstride * 2 <= DECODE_XS;
    const __nv_bfloat16* xst = reinterpret_cast<const __nv_bfloat16*>(dyn + s * stage_u4 + slab_u4);
    const int ncols = min(DECODE_COLS, N - n0);
    const uint16_t* col0 = reinterpret_cast<const uint16_t*>(dyn + s * stage_u4) + gq * rb;
    const uint16_t* col1 = ncols == DECODE_COLS ? col0 + 8 * rb : col0;   // an 8-column tail decodes column gq twice
    cp_wait<DSTAGES - 1>();              // this tile's stage; the next ones may still be in flight
    __syncthreads();
    for (int c0 = 0; c0 < nrows; c0 += CHUNK) {
      const int crows = min(CHUNK, nrows - c0), mts = (crows + 15) >> 4;
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int ra = 16 * m + gq, rb8 = ra + 8;
        if (staged) {
          xa[m] = ra < crows ? xst + (c0 + ra) * xstride + 32 * t : nullptr;
          xb[m] = rb8 < crows ? xst + (c0 + rb8) * xstride + 32 * t : nullptr;
        } else {
          xa[m] = ra < crows ? X + (size_t)row(first + c0 + ra) * K + 32 * t : nullptr;
          xb[m] = rb8 < crows ? X + (size_t)row(first + c0 + rb8) * K + 32 * t : nullptr;
        }
      }
      float acc[MT][2][4];
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
      if (staged) {
        decode_loop<MT, true>(acc, xa, xb, mts, col0, col1, K / 128, warp, t, tab);
      } else {
        decode_loop<MT, false>(acc, xa, xb, mts, col0, col1, K / 128, warp, t, tab);
      }
      if (warp > 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) part[((warp - 1) * PART + (m * 2 + j) * 4 + q) * 32 + lane] = acc[m][j][q];
      }
      __syncthreads();
      if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m >= mts) break;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            if (8 * j >= ncols) break;
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int w = 1; w < DWARPS; ++w)
                acc[m][j][q] += part[((w - 1) * PART + (m * 2 + j) * 4 + q) * 32 + lane];
            const int col = n0 + 8 * j + 2 * t, ra = 16 * m + gq;
            if (ra < crows) store2(out + (size_t)row(first + c0 + ra) * N + col, acc[m][j][0], acc[m][j][1]);
            if (ra + 8 < crows)
              store2(out + (size_t)row(first + c0 + ra + 8) * N + col, acc[m][j][2], acc[m][j][3]);
          }
        }
      }
      __syncthreads();                   // the partial sums (and, after the last chunk, stage s) are free again
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// prefill: blockIdx.y a 64-row tile of one expert, blockIdx.x 128 columns. A grouped call's rows are read through
// ``order`` (rows sorted by expert); tile_end[e] is the inclusive sum of tiles over experts 0..e, starts[e] the first
// sorted row of expert e (E + 1 entries). A linear call passes no order: one expert, tile i is rows 64 i ...
//
// x's stage row is 16 uint4 (128 inputs), chunk c stored at c ^ (row & 7). Lanes with t >= 2 read the table variant
// whose pairs come in the order 2, 3, 0, 1 and take the other half of each grid entry first, so the 32 lanes of an
// x load hit 16 distinct 8-byte bank pairs, two lanes each (the 256 bytes' minimum of two wavefronts).

constexpr int PM = 64, PWARPS = 4, PNT = 4, PCOLS = PWARPS * 8 * PNT, PSTAGES = 2;
constexpr int XROW_U4 = 16;
constexpr int XSTAGE_U4 = PM * XROW_U4;

__global__ void __launch_bounds__(PWARPS * 32)
    prefill_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
                   const int* __restrict__ order, const int* __restrict__ tile_end, const int* __restrict__ starts,
                   int E, int R, int N, int K, const uint32_t* __restrict__ gtab, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) uint32_t tab[2 * TABLE_WORDS];
  __shared__ __align__(16) uint4 xs[PSTAGES * XSTAGE_U4];
  __shared__ int xrow[PM];                                       // the tile's x rows (-1 past its count)
  const int tile = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, t = lane & 3;
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
  for (int r = tid; r < PM; r += PWARPS * 32) xrow[r] = r < cnt ? (order ? __ldg(order + first + r) : first + r) : -1;
  __syncthreads();
  const int groups = K / 128;
  auto stage = [&](int s, int g) {                               // chunk q % 16 of row q / 16, q = tid + 128 i
#pragma unroll
    for (int i = 0; i < XSTAGE_U4 / (PWARPS * 32); ++i) {
      const int q = tid + PWARPS * 32 * i, r = q >> 4, c = q & 15, src = xrow[r];
      if (src >= 0) cp16(xs + s * XSTAGE_U4 + r * XROW_U4 + (c ^ (r & 7)), X + (size_t)src * K + 128 * g + 8 * c);
    }
  };
  stage(0, 0);
  cp_commit();
  const int kb = K / QK;
  const int c0 = blockIdx.x * PCOLS + warp * 8 * PNT;
  const uint8_t* wcol[PNT];
#pragma unroll
  for (int j = 0; j < PNT; ++j) {
    const int n = min(c0 + 8 * j + gq, N - 1);                  // a column past N decodes column N - 1, unstored
    wcol[j] = W + ((size_t)e * N + n) * (size_t)(kb * 66);
  }
  const bool active = c0 < N;
  const int v = t >> 1;                                          // this lane's table variant and first half
  const uint32_t* mytab = tab + v * TABLE_WORDS;
  const uint4 kv = spreads(v);
  Raw raw0[PNT], raw1[PNT];                                      // the lane's sub-blocks of even and odd groups
  auto fetch = [&](Raw (&raw)[PNT], int g) {
#pragma unroll
    for (int j = 0; j < PNT; ++j) raw[j] = fetch_raw(wcol[j], 4 * g + t);
  };
  float acc[PM / 16][PNT][4];
#pragma unroll
  for (int m = 0; m < PM / 16; ++m)
#pragma unroll
    for (int j = 0; j < PNT; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
  const int mts = (cnt + 15) >> 4;
  if (active) {                                                  // weights two groups ahead (groups is even)
    fetch(raw0, 0);
    fetch(raw1, 1);
  }
  auto step = [&](int g, Raw (&raw)[PNT]) {
    cp_wait<0>();
    __syncthreads();                                             // stage g landed (and, at g = 0, the tables)
    if (g + 1 < groups) stage((g + 1) % PSTAGES, g + 1);
    cp_commit();
    if (active) {
      Sub sub[PNT];
#pragma unroll
      for (int j = 0; j < PNT; ++j) sub[j] = make_sub(raw[j].d, raw[j].aux0, raw[j].aux1);
      if (g + 2 < groups) fetch(raw, g + 2);
      const uint4* xt = xs + (g % PSTAGES) * XSTAGE_U4;
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        uint4 ev[PNT];
#pragma unroll
        for (int j = 0; j < PNT; ++j) ev[j] = entry(sub[j], l, mytab, kv);
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
  };
  for (int g = 0; g < groups; g += 2) {
    step(g, raw0);
    step(g + 1, raw1);
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

void iq2xxs_decode_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor* ids, at::Tensor* plan, int64_t n,
                        int64_t mt, const at::Tensor& tab, at::Tensor& out) {
  const at::cuda::CUDAGuard guard(x.device());
  const int R = x.size(0), K = x.size(1), N = n, E = w.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (plan) {
    plan_kernel<<<1, 1024, 2 * E * sizeof(int), stream>>>(ids->data_ptr<int>(), R, E, plan->data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const size_t smem = (size_t)DSTAGES * (DECODE_COLS * (K / QK) * 66 + DECODE_XS) +
                      (size_t)(DWARPS - 1) * mt * 2 * 4 * 32 * sizeof(float);
  const int experts = ids ? std::min(R, E) : 1;
  const int tiles = experts * ((N + DECODE_COLS - 1) / DECODE_COLS);    // a bound: the call's distinct experts
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* ip = ids ? ids->data_ptr<int>() : nullptr;
  const auto* pp = plan ? plan->data_ptr<int>() : nullptr;
  const auto* tp = reinterpret_cast<const uint32_t*>(tab.data_ptr<int>());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  // the shared-memory opt-in and the occupancy it gives, asked of the driver once per (device, MT, smem): a call
  // otherwise spends two driver queries before its launch
  static std::mutex lock;
  static std::unordered_map<uint64_t, int> blocks_per_sm;
  auto go = [&](auto kernel) {
    const uint64_t key = (uint64_t)x.get_device() << 48 | (uint64_t)mt << 40 | (uint64_t)smem;
    int per_sm = 0;
    {
      const std::lock_guard<std::mutex> hold(lock);
      const auto hit = blocks_per_sm.find(key);
      if (hit != blocks_per_sm.end()) {
        per_sm = hit->second;
      } else {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));  // > 48 KB
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, DWARPS * 32, smem));
        blocks_per_sm.emplace(key, per_sm);
      }
    }
    const int blocks = std::max(1, std::min(tiles, sms * std::max(per_sm, 1)));
    kernel<<<blocks, DWARPS * 32, smem, stream>>>(xp, w.data_ptr<uint8_t>(), ip, pp, R, E, N, K, tp, op);
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
  const dim3 grid((N + PCOLS - 1) / PCOLS, tiles);           // a tile's column blocks run together: x stays in L2
  prefill_kernel<<<grid, PWARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(),
      order ? order->data_ptr<int>() : nullptr, tile_end ? tile_end->data_ptr<int>() : nullptr,
      starts ? starts->data_ptr<int>() : nullptr, experts, R, N, K,
      reinterpret_cast<const uint32_t*>(tab.data_ptr<int>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
