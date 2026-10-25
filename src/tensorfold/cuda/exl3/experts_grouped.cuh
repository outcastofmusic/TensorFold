// Grouped EXL3 expert GEMV (after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): rows stay independent, K ranges fixed by shape, warps summed in order.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "epilogue.cuh"
#include "pdl.cuh"

namespace tf_exl3x {

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// A rate code K2 is half-bits a value for the widths ExLlamaV3 writes (2..16), or RATE_WORDS + the tile's 32-bit
// words for a rate between them, which sushi's packs use (2.25 bits is 18 words: code 118).
constexpr int RATE_WORDS = 100;
__host__ __device__ constexpr int tile_words(int k2) { return k2 > RATE_WORDS ? k2 - RATE_WORDS : 4 * k2; }
// one past the last bit of value p's window: floor((p + 1) * bits), and a lane's eight values take a whole tw bits
__host__ __device__ constexpr int stream_end(int tw, int p) { return ((p + 1) * tw) >> 3; }
// whether every lane's runs of gv windows can be cut from two adjacent words
__host__ __device__ constexpr bool runs_fit(int tw, int gv) {
    for (int lane = 0; lane < 32; ++lane)
        for (int g = 0; g < 8 / gv; ++g) {
            const int last = lane * tw + stream_end(tw, g * gv + gv - 1) + 32 * tw;
            const int sh = (((last - 1) >> 5) + 1) * 32 - last;
            if (stream_end(tw, g * gv + gv - 1) - stream_end(tw, g * gv) + 16 + sh > 64) return false;
        }
    return true;
}

// A tile is TW words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = tile_words(K2);
    static constexpr int LW = (TW + 31) / 32;
    // windows sharing one 64-bit merge (tests/cuda/test_exl3_experts.py checks every K2); a rate between
    // ExLlamaV3's widths takes the widest run that fits
    static constexpr int GV = K2 > RATE_WORDS ? (runs_fit(TW, 8) ? 8 : (runs_fit(TW, 4) ? 4 : 2))
                              : (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return stream_end(TW, p); }
    // right shift of window j of a lane's run g relative to the run's last window
    __host__ __device__ static constexpr int off(int g, int j) { return end(g * GV + GV - 1) - end(g * GV + j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 32 * TW;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves; wmask keeps the pack's codeword window
// (0xffff for ExLlamaV3's own packs).
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t wmask, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        // 4 bits: lane L's windows are exactly words L-1 and L (the GLM kernel's decode)
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & wmask;
        st[1] = (s >> 4) & wmask;
        st[2] = s & wmask;
        st[3] = (w[0] >> 16) & wmask;
        st[4] = (w[0] >> 12) & wmask;
        st[5] = (w[0] >> 8) & wmask;
        st[6] = (w[0] >> 4) & wmask;
        st[7] = w[0] & wmask;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(g, j)) & wmask;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc, PF tiles in flight. Each weight tile is read and
// decoded once for the block's nrt row tiles (16 rows each). Measured on a GB10 (512 experts, 2560 -> 640, 3 bits,
// 1,024 rows): the tensor-core work is free, reading the weights is 44% of the kernel and reading the rows 24%.
template <int CB, int K2, int NT, int PF, int RT>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* (&x0)[RT], const half* (&x1)[RT],
                                           bool (&ok0)[RT], bool (&ok1)[RT], int nrt, int lane, uint32_t wmask,
                                           float (&acc)[RT][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[RT][4];
#pragma unroll
                for (int r = 0; r < RT; ++r)
                    if (r < nrt) {
                        a[r][0] = load_pair(x0[r] + k, ok0[r]);
                        a[r][1] = load_pair(x1[r] + k, ok1[r]);
                        a[r][2] = load_pair(x0[r] + k + 8, ok0[r]);
                        a[r][3] = load_pair(x1[r] + k + 8, ok1[r]);
                    }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, wmask, b0, b1);
#pragma unroll
                    for (int r = 0; r < RT; ++r)
                        if (r < nrt) {
                            mma16816(acc[r][i][0], a[r], b0);
                            mma16816(acc[r][i][1], a[r], b1);
                        }
                }
            }
        }
    }
}

// cp.async copies into shared memory (sm_80+): 16 or 8 bytes, or 16 zero bytes when ``ok`` is false.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok = true) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
                 "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// A staged 16x16 fp16 tile: 16 rows of two 16-byte halves (k 0-7, k 8-15), half c of row r at r * 16 + 8 * (c ^ r / 4
// % 2) halves, so the 8 rows each ldmatrix reads fall on distinct banks.
__device__ __forceinline__ int staged_at(int row, int c) { return row * 16 + 8 * (c ^ ((row >> 2) & 1)); }

// The A fragment of a staged tile: the same four registers load_pair builds (rows g and g + 8, k 2t and 2t + 8).
__device__ __forceinline__ void ldmatrix_a(uint32_t (&a)[4], const half* tile, int lane) {
    const half* p = tile + staged_at(lane & 15, lane >> 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem_addr(p)));
}

// TF_EXL3_EXPERT_STAGED: warp_tiles with the weights and the member rows of several k steps in flight through shared
// memory (cp.async), where warp_tiles holds one k step of weights in registers and loads the rows when it uses them.
// A warp's STAGE_BYTES hold as many k steps as fit at the tile's rate, up to S: 6 at 3 bits, 4 at 5 bits.
// The GB10 profile of a 2,048-row Flash Next window (ncu, 21 Oct): 248 registers held 2 blocks (8 warps) an SM,
// no warp was ready to issue in 75% of cycles, and long-scoreboard waits were 3.4 of 8.0 cycles an instruction.
// Each warp stages its own k range, so a __syncwarp orders the copies. Every mma gets the fragments warp_tiles gives
// it (absent rows are zero-filled, as load_pair returns zero) in the same order: a row's sums are the same bits.
constexpr int STAGE_BYTES = 11264;     // a warp's share of the staged path's shared memory (2 blocks an SM fit)
template <int NT, int RT, int TW, int S>
struct Staged {
    static constexpr int A_HALVES = RT * 16 * 16;
    static constexpr int W_WORDS = NT * TW;
    static constexpr int BYTES = (A_HALVES * 2 + W_WORDS * 4 + 15) / 16 * 16;
    static constexpr int DEPTH = STAGE_BYTES / BYTES < S ? STAGE_BYTES / BYTES : S;
    static_assert(DEPTH >= 2, "two k steps in flight at least");
};

template <int CB, int K2, int NT, int RT, int S>
__device__ __forceinline__ void warp_tiles_staged(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt,
                                                  int nt0, const half* __restrict__ X, int K, const int* rows_sh,
                                                  int nrt, int lane, uint32_t wmask, float (&acc)[RT][NT][2][4],
                                                  unsigned char* buf) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    using St = Staged<NT, RT, TW, S>;
    constexpr int D = St::DEPTH;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW;
    // lane L copies half L & 1 of row L >> 1 of each row tile: 16 bytes a k step
    const half* xs[RT];
    bool ok[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int row = rows_sh[16 * r + (lane >> 1)];
        ok[r] = row >= 0;
        xs[r] = X + (size_t)(ok[r] ? row : 0) * K + kt0 * 16 + 8 * (lane & 1);
    }
    auto issue = [&](int it) {
        unsigned char* st = buf + (it % D) * St::BYTES;
        half* sa = reinterpret_cast<half*>(st);
#pragma unroll
        for (int r = 0; r < RT; ++r)
            if (r < nrt) cp16(sa + r * 256 + staged_at(lane >> 1, lane & 1), xs[r] + it * 16, ok[r]);
        uint32_t* sw = reinterpret_cast<uint32_t*>(st + St::A_HALVES * 2);
        const uint32_t* src = tp + (size_t)it * kstride;
        if constexpr (TW % 4 == 0) {      // every tile starts on 16 bytes (2.25 bits, 18 words, on 8)
#pragma unroll
            for (int c = lane; c < St::W_WORDS / 4; c += 32) cp16(sw + 4 * c, src + 4 * c);
        } else {
#pragma unroll
            for (int c = lane; c < St::W_WORDS / 2; c += 32) cp8(sw + 2 * c, src + 2 * c);
        }
    };
#pragma unroll
    for (int d = 0; d < D - 1; ++d) {
        if (d < nkt) issue(d);
        cp_commit();
    }
    for (int it = 0; it < nkt; ++it) {
        cp_wait<D - 2>();
        __syncwarp();                     // every lane's copies of step it landed; every lane is done with step it - 1
        if (it + D - 1 < nkt) issue(it + D - 1);
        cp_commit();
        const unsigned char* st = buf + (it % D) * St::BYTES;
        const half* sa = reinterpret_cast<const half*>(st);
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(st + St::A_HALVES * 2);
        uint32_t a[RT][4];
#pragma unroll
        for (int r = 0; r < RT; ++r)
            if (r < nrt) ldmatrix_a(a[r], sa + r * 256, lane);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t w[LW];
#pragma unroll
            for (int l = 0; l < LW; ++l) {
                if constexpr ((TW % 32) == 0)
                    w[l] = sw[i * TW + l * 32 + lane];
                else
                    w[l] = (l * 32 + lane < TW) ? sw[i * TW + l * 32 + lane] : 0u;
            }
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w, map, lane, wmask, b0, b1);
#pragma unroll
            for (int r = 0; r < RT; ++r)
                if (r < nrt) {
                    mma16816(acc[r][i][0], a[r], b0);
                    mma16816(acc[r][i][1], a[r], b1);
                }
        }
    }
    cp_wait<0>();
}

// The rate codes with a kernel instance: half-bits 2..16, and 2.25 bits (sushi's GLM and Flash Next packs).
constexpr int K2_QUARTER = RATE_WORDS + 18;
__host__ __device__ constexpr bool k2_supported(int k2) {
    return (k2 >= 2 && k2 <= 16) || k2 == K2_QUARTER;
}

// What a program does with its sums. EPI_NONE writes them to Z for the epilogue kernels (experts.cu). The fused
// epilogues run those kernels' arithmetic (epilogue.cuh) in grouped_kernel's listed layout at the prompt setting:
// EPI_DOWN (one K split) takes each row's sums from shared memory and writes Y; EPI_GATEUP writes Z as before, and
// the last of an (item, n block)'s mats x SK programs to arrive reads them back from L2 in split order and writes Xd.
// An integer counter picks that program; the sums' order does not depend on which one it is.
constexpr int EPI_NONE = 0, EPI_GATEUP = 1, EPI_DOWN = 2;

struct Epi {
    const half* s0 = nullptr;    // gate/up: svh_g [E, N]; down: svh_d [E, N]
    const half* s1 = nullptr;    // gate/up: svh_u [E, N]
    const half* s2 = nullptr;    // gate/up: suh_d [E, N]
    void* out = nullptr;         // gate/up: Xd fp16 [P, N]; down: Y fp32 or bf16 [P, N]
    int* arrivals = nullptr;     // gate/up: programs arrived a (item, n block), zero between launches
    float limit = 0.f;
    int act = 0;                 // gate/up: SwiGLU mode (gateup_epilogue_kernel's act_mode)
    int bf16 = 0;                // down: Y is bf16
    // TF_EXL3_ROT_FUSE (a staged gate/up launch on the item list, ROT): the rows rot_in_kernel would rotate, bf16 or
    // fp16 with rows x_stride values apart on 16 bytes, suh_g and suh_u [E, K], and a counter an (item, matrix), zero
    // between launches
    const void* x = nullptr;
    int x_stride = 0;
    int x_bf16 = 0;
    const half* suh0 = nullptr;
    const half* suh1 = nullptr;
    int* rot_arrivals = nullptr;
};

// TF_EXL3_ROT_FUSE: a staged gate/up launch writes rot_in_kernel's rows itself. The NB n-block programs of an
// (item, matrix) each rotate the item's members over 128-wide Hadamard blocks nb, nb + NB, .. of K into X0 or X1,
// where rot_in_kernel writes them, then wait at a counter for the other NB - 1 and read the rows back through
// warp_tiles_staged's copies, from L2. rot_in_kernel wrote every pick's rows in pick order before the launch, 230 MB a
// 2,048-row Flash Next window that gate/up read back from DRAM. Each block of rows is rotated once, as there.
//
// The wait needs the NB programs resident together. They are consecutive in the grid (n block fastest), and blocks
// start in grid order, so the oldest unfinished item's programs have all started or have a free SM slot to start in.
// grouped_kernel never triggers its dependents early (no grid_launch), so a PDL launch after it takes no slot first.
//
// A unit of the rotation is 8 rows of Hadamard block hb (r0, r0 + 2, .. on lanes 0-15, r0 + 1, .. on 16-31), 8
// consecutive values a lane, so fwht128's butterflies run in its order, strides 1, 2, 4 in registers and 8..64 across
// lanes. Every butterfly adds or subtracts the same two values, and the first is contracted as the compiler contracts
// rot_in_kernel's, fma(x0, s0, x1 * s1) (its SASS at 1d2deb4e), with the rounding intrinsics so nothing else is: the
// same bits. A warp reads ROT_UNITS units (rot_load) before it rotates and stores any (rot_store): the loads cannot
// pass the stores before them, and the prologue holds no sums yet, so the registers are free.
constexpr int ROT_UNITS = 4;
struct RotUnit {
    uint4 s;                           // suh_mat[e] at the lane's 8 values
    uint4 x[4];                        // the lane's 8 values of its 4 rows of x (bf16 or fp16 bits)
};

__device__ __forceinline__ void rot_load(RotUnit& u, const Epi& ep, const half* __restrict__ suh, const int* xrow,
                                         int hb, int r0, int lane) {
    const int k = hb * 128 + 8 * (lane & 15), odd = lane >> 4;
    u.s = __ldg(reinterpret_cast<const uint4*>(suh + k));
    const half* xh = static_cast<const half*>(ep.x) + k;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int xr = xrow[r0 + 2 * q + odd];
        u.x[q] = __ldg(reinterpret_cast<const uint4*>(xh + (size_t)(xr >= 0 ? xr : 0) * ep.x_stride));
    }
}

__device__ __forceinline__ void rot_store(const RotUnit& u, const Epi& ep, const int* rows_sh, half* __restrict__ X,
                                          int K, int hb, int r0, int lane) {
    const int l = lane & 15, odd = lane >> 4;
    const int k = hb * 128 + 8 * l;
    float s[8];
    const half* sh = reinterpret_cast<const half*>(&u.s);
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = __half2float(sh[j]);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&u.x[q]);
        float x[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (ep.x_bf16) {
                x[2 * j] = __uint_as_float(w[j] << 16);
                x[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
            } else {
                const half2 h = *reinterpret_cast<const half2*>(&w[j]);
                x[2 * j] = __low2float(h);
                x[2 * j + 1] = __high2float(h);
            }
        }
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float m = __fmul_rn(x[j + 1], s[j + 1]);
            v[j] = __fmaf_rn(x[j], s[j], m);
            v[j + 1] = __fmaf_rn(x[j], s[j], -m);
        }
#pragma unroll
        for (int st = 2; st <= 4; st <<= 1)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (!(j & st)) {
                    const float a = v[j], b = v[j + st];
                    v[j] = __fadd_rn(a, b);
                    v[j + st] = __fsub_rn(a, b);
                }
#pragma unroll
        for (int m = 1; m < 16; m <<= 1)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
                v[j] = (l & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
            }
        const int p = rows_sh[r0 + 2 * q + odd];
        if (p < 0) continue;
        uint32_t out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 h = __halves2half2(__float2half_rn(__fmul_rn(v[2 * j], HAD_SCALE)),
                                     __float2half_rn(__fmul_rn(v[2 * j + 1], HAD_SCALE)));
            out[j] = *reinterpret_cast<uint32_t*>(&h);
        }
        *reinterpret_cast<uint4*>(X + (size_t)p * K + k) = make_uint4(out[0], out[1], out[2], out[3]);
    }
}

// Program (expert u, n block, split and member group): up to 16 * RT members times W_q over the split's K range; warps
// added in order. A row's sums do not depend on NT, PF or RT, nor on its place among the members.
//
// Two layouts of the same programs. Spanned (items null): grid (experts, n blocks, matrices x splits x member groups),
// every member group a window's rows could fill, so a prompt window of 1,024 rows launches 32 groups an expert and
// leaves 31 of a typical routed expert's empty. Listed (TF_EXL3_EXPERT_ITEMS): grid (n blocks, matrices x splits,
// items), one program a (place << 16 | member group) item group_kernel listed, so only filled groups launch, and an
// item's n blocks run next to each other while its rows are still in L2. Each program computes the same sums.
//
// S > 0 (TF_EXL3_EXPERT_STAGED) runs the warps' k ranges through warp_tiles_staged, S k steps in flight a warp, in
// shared memory that the warps' sums reuse once every warp is done with it. ROT (TF_EXL3_ROT_FUSE, a staged gate/up
// launch on the item list) first writes its share of the rows X0 and X1 hold (rot_load, rot_store).
template <int NT, int W, bool STAGED, bool ROT = false>
struct GroupedSmem {
    float red[W][16][NT * 16];
};
template <int NT, int W>
struct GroupedSmem<NT, W, true, false> {
    union {
        float red[W][16][NT * 16];
        __align__(16) unsigned char staged[W * STAGE_BYTES];
    };
};
template <int NT, int W>
struct GroupedSmem<NT, W, true, true> {
    union {
        float red[W][16][NT * 16];
        __align__(16) unsigned char staged[W * STAGE_BYTES];
    };
    int xrow[32];                      // the members' rows of x (-1: none), for rot_load
};

template <int CB, int NT, int W, int PF, int LO, int HI, int RT, int EPI, int S = 0, bool ROT = false>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, uint32_t wmask,
    const int* __restrict__ items, const int* __restrict__ icount, const Epi epi) {
    static_assert(EPI == EPI_NONE || NT * 16 == 128, "a fused epilogue takes one 128-wide Hadamard block a program");
    static_assert(!ROT || (S > 0 && EPI != EPI_DOWN && RT <= 2), "ROT writes a staged gate/up program's rows");
    tf_exl3_pdl::grid_wait();
    int u, mtile, zz, nb;
    if (items != nullptr) {
        const int it = blockIdx.z;
        if (it >= icount[0]) return;
        const int code = items[it];
        u = code >> 16;
        mtile = code & 0xffff;
        nb = blockIdx.x;
        zz = blockIdx.y;
    } else {
        u = blockIdx.x;
        if (u >= ucount[0]) return;
        const int MT = (maxm + 16 * RT - 1) / (16 * RT);
        mtile = blockIdx.z % MT;
        zz = blockIdx.z / MT;
        nb = blockIdx.y;
    }
    // the staged path runs a program's K splits itself, in split order (``zz`` is the matrix), and adds their sums
    // from zero as the epilogues do: one total a matrix reaches Z, which the epilogues then read as one split
    constexpr bool INNER = S > 0;
    const int split0 = INNER ? 0 : zz % SK;
    const int mat = INNER ? zz : zz / SK;
    const int nsplit = INNER ? SK : 1;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16 * RT];
    if (threadIdx.x < 16 * RT) {
        const int m = mtile * 16 * RT + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this group is empty
    const half* x0[RT];
    const half* x1[RT];
    bool ok0[RT], ok1[RT];
    int nrt = 0;                                          // row tiles holding a member (they come first too)
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int r0 = rows_sh[16 * r + g], r1 = rows_sh[16 * r + g + 8];
        x0[r] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        x1[r] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
        ok0[r] = r0 >= 0;
        ok1[r] = r1 >= 0;
        nrt += rows_sh[16 * r] >= 0;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = nb * NT;
    static_assert(sizeof(GroupedSmem<NT, W, (S > 0), ROT>) <= 48 * 1024, "grouped_kernel's shared memory is static");
    __shared__ GroupedSmem<NT, W, (S > 0), ROT> sm;
    auto& red = sm.red;
    if constexpr (ROT) {
        if (threadIdx.x < 16 * RT) {
            const int m = mtile * 16 * RT + threadIdx.x;
            const int code = m < maxm ? members[u * maxm + m] : -1;
            sm.xrow[threadIdx.x] = code >= 0 ? code >> 5 : -1;
        }
        __syncthreads();
        // this program's share: Hadamard blocks nb, nb + NB, .. of K, 8 rows a warp at a time
        const int NB = gridDim.x, KB = K >> 7;
        const int units = (KB - nb + NB - 1) / NB * 2 * nrt;
        const half* suh = (mat ? epi.suh1 : epi.suh0) + (size_t)e * K;
        half* Xw = const_cast<half*>(X);             // ROT reads X only through warp_tiles_staged's copies
        for (int u0 = warp; u0 < units; u0 += W * ROT_UNITS) {
            RotUnit ru[ROT_UNITS];
#pragma unroll
            for (int c = 0; c < ROT_UNITS; ++c) {
                const int unit = u0 + c * W;
                if (unit < units)
                    rot_load(ru[c], epi, suh, sm.xrow, nb + unit / (2 * nrt) * NB, unit % (2 * nrt) * 8, lane);
            }
#pragma unroll
            for (int c = 0; c < ROT_UNITS; ++c) {
                const int unit = u0 + c * W;
                if (unit < units)
                    rot_store(ru[c], epi, rows_sh, Xw, K, nb + unit / (2 * nrt) * NB, unit % (2 * nrt) * 8, lane);
            }
        }
        // the threadFenceReduction pattern: every program's rows are visible GPU-wide before its arrival counts; the
        // last to arrive after the wait leaves the counter zero
        int* count = epi.rot_arrivals + (size_t)blockIdx.z * gridDim.y + blockIdx.y;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(count, 1);
            while (*reinterpret_cast<volatile int*>(count) < NB) __nanosleep(64);
            __threadfence();
            if (atomicAdd(count, 1) == 2 * NB - 1) *count = 0;   // all NB programs have passed the wait
        }
        __syncthreads();
    }
    constexpr int PER_THREAD = 16 * NT * 16 / (W * 32);   // a row tile's sums a thread adds up (staged path)
    float tot[INNER ? RT : 1][INNER ? PER_THREAD : 1];
    for (int sp = 0; sp < nsplit; ++sp) {
        const int split = split0 + sp;
        const int kt0 = split * per_split + warp * per_warp;

        float acc[RT][NT][2][4];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[r][i][h][c] = 0.f;

        switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
        case K2_:                                                                                               \
            if constexpr (K2_ >= LO && K2_ <= HI) {                                                             \
                if constexpr (S > 0)                                                                            \
                    warp_tiles_staged<CB, K2_, NT, RT, S>(                                                      \
                        T, NTILES, kt0, per_warp, nt0, X, K, rows_sh, nrt, lane, wmask, acc,                    \
                        sm.staged + warp * STAGE_BYTES);                                                        \
                else                                                                                            \
                    warp_tiles<CB, K2_, NT, PF, RT>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, nrt, lane, \
                                                    wmask, acc);                                                \
            } else                                                                                              \
                __trap();                                                                                       \
            break;
            TF_EXL3X_CASE(2)
            TF_EXL3X_CASE(3)
            TF_EXL3X_CASE(4)
            TF_EXL3X_CASE(5)
            TF_EXL3X_CASE(6)
            TF_EXL3X_CASE(7)
            TF_EXL3X_CASE(8)
            TF_EXL3X_CASE(9)
            TF_EXL3X_CASE(10)
            TF_EXL3X_CASE(11)
            TF_EXL3X_CASE(12)
            TF_EXL3X_CASE(13)
            TF_EXL3X_CASE(14)
            TF_EXL3X_CASE(15)
            TF_EXL3X_CASE(16)
            TF_EXL3X_CASE(K2_QUARTER)
#undef TF_EXL3X_CASE
            default:
                __trap();
        }

        // warps' partial sums through shared memory, added in warp order; one row tile at a time, so RT costs no
        // shared memory
        if constexpr (S > 0) __syncthreads();                 // the staged k steps of every warp are spent
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r >= nrt) break;
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    red[warp][g][col] = acc[r][i][h][0];
                    red[warp][g][col + 1] = acc[r][i][h][1];
                    red[warp][g + 8][col] = acc[r][i][h][2];
                    red[warp][g + 8][col + 1] = acc[r][i][h][3];
                }
            __syncthreads();
            if constexpr (INNER && EPI != EPI_DOWN) {
                // one split's sums: alone they are the total (as the epilogue reads one split); of several, the
                // total adds them to 0.f in split order, as the epilogue would (0.f + that total is the total: a sum
                // that starts at +0.f is never -0)
#pragma unroll
                for (int q = 0; q < PER_THREAD; ++q) {
                    const int idx = threadIdx.x + q * W * 32;
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    float sum = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) sum += red[w][row][col];
                    if (SK == 1)
                        tot[r][q] = sum;
                    else
                        tot[r][q] = (sp == 0 ? 0.f : tot[r][q]) + sum;
                }
            } else {
                for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    const int rr = rows_sh[16 * r + row];
                    if (rr < 0) continue;
                    float sum = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) sum += red[w][row][col];
                    if constexpr (EPI == EPI_DOWN)
                        red[0][row][col] = sum;               // only this thread reads this place
                    else
                        Z[(((size_t)mat * SK + split) * P + rr) * N + nt0 * 16 + col] = sum;
                }
            }
            __syncthreads();
            if constexpr (EPI == EPI_DOWN) {
                // down_epilogue_kernel's arithmetic on this row tile: one warp a row, a lane's 4 columns
                const int n = nt0 * 16 + 4 * lane;
                for (int row = warp; row < 16; row += W) {
                    const int p = rows_sh[16 * r + row];
                    if (p < 0) continue;
                    float v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float s = 0.f;
                        s += red[0][row][4 * lane + j];           // SK = 1: the loop over splits from zero
                        v[j] = s;
                    }
                    if (epi.bf16)
                        down_tail<__nv_bfloat16, (S > 0)>(
                            v, lane, epi.s0 + (size_t)e * N + n,
                            reinterpret_cast<__nv_bfloat16*>(epi.out) + (size_t)p * N + n);
                    else
                        down_tail<float, (S > 0)>(v, lane, epi.s0 + (size_t)e * N + n,
                                                  reinterpret_cast<float*>(epi.out) + (size_t)p * N + n);
                }
                __syncthreads();
            }
        }
    }
    if constexpr (INNER && EPI != EPI_DOWN) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r >= nrt) break;
#pragma unroll
            for (int q = 0; q < PER_THREAD; ++q) {
                const int idx = threadIdx.x + q * W * 32;
                const int row = idx / (NT * 16), col = idx % (NT * 16);
                const int rr = rows_sh[16 * r + row];
                if (rr >= 0) Z[((size_t)mat * P + rr) * N + nt0 * 16 + col] = tot[r][q];
            }
        }
    }
    if constexpr (EPI == EPI_GATEUP) {
        // the threadFenceReduction pattern: every program's Z is visible GPU-wide before its arrival counts
        const int SKZ = INNER ? 1 : SK;                   // the splits Z holds a matrix
        __shared__ bool last;
        __threadfence();
        __syncthreads();
        int* arrivals = epi.arrivals + (size_t)blockIdx.z * gridDim.x + blockIdx.x;
        if (threadIdx.x == 0) last = atomicAdd(arrivals, 1) == 2 * SKZ - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        if (threadIdx.x == 0) *arrivals = 0;          // every program of this (item, n block) has counted
        // gateup_epilogue_kernel's arithmetic on the group's rows: one warp a row, a lane's 4 columns
        const int n = nt0 * 16 + 4 * lane;
        for (int row = warp; row < 16 * RT; row += W) {
            const int p = rows_sh[row];
            if (p < 0) continue;
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sg = 0.f, su = 0.f;
                for (int s = 0; s < SKZ; ++s) {
                    sg += __ldcg(Z + ((size_t)(0 * SKZ + s) * P + p) * N + n + j);
                    su += __ldcg(Z + ((size_t)(1 * SKZ + s) * P + p) * N + n + j);
                }
                gv[j] = sg;
                uv[j] = su;
            }
            gateup_tail<(S > 0)>(gv, uv, lane, epi.s0 + (size_t)e * N + n, epi.s1 + (size_t)e * N + n,
                        epi.s2 + (size_t)e * N + n, reinterpret_cast<half*>(epi.out) + (size_t)p * N + n, epi.limit,
                        epi.act);
        }
    }
}

// W_q [K, N] fp16 of one matrix through the same lane decode (tests; not on the forward path).
template <int CB, int K2>
__global__ void dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int K, int N, uint32_t wmask) {
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x;
    const int NTILES = N >> 4;
    uint32_t w[Fmt<K2>::LW];
    load_words<K2>(w, T + ((size_t)kt * NTILES + nt) * Fmt<K2>::TW + lane, lane);
    const LaneMap<K2> map(lane);
    uint32_t b0[2], b1[2];
    decode_tile<CB, K2>(w, map, lane, wmask, b0, b1);
    const int g = lane >> 2, t = lane & 3;
    uint32_t v[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h = *reinterpret_cast<half2*>(&v[q]);
        const int col = nt * 16 + g + 8 * (q >> 1);
        const int row = kt * 16 + 2 * t + 8 * (q & 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

struct GroupedArgs {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* uids;
    const int* ucount;
    const int* members;
    float* z;
    int K, N, P, SK, maxm, slots;
    int nexp_max;        // grid.x (upper bound of distinct experts)
    int mats, nt, warps, pf, lo, hi;
    int rt = 1;          // row tiles a block (16 rows each) sharing one read of the weights
    uint32_t wmask = 0xffffu;   // the low bits of a 16-bit window that make the codeword
    bool pdl = false;    // programmatic dependent launch (pdl.cuh)
    const int* items = nullptr;    // grouped_kernel's listed layout: group_kernel's items, their count, the list's size
    const int* icount = nullptr;
    int nitems = 0;
    int epi = EPI_NONE;            // a fused epilogue (listed layout, prompt setting) and its arguments
    Epi ep;
    bool staged = false;           // TF_EXL3_EXPERT_STAGED: the prompt setting through warp_tiles_staged
    bool rot = false;              // TF_EXL3_ROT_FUSE: a staged gate/up launch writes X0, X1 from ep.x itself
};

constexpr int STAGES = 8;          // most k steps in flight a warp on the staged path (Staged::DEPTH)

template <int CB>
void grouped_launch(const GroupedArgs& a, cudaStream_t stream) {
    const int MT = (a.maxm + 16 * a.rt - 1) / (16 * a.rt);
    const unsigned nb = (unsigned)(a.N / (16 * a.nt));
    const int zs = a.staged ? a.mats : a.mats * a.SK;     // the staged path runs the splits inside a program
    const dim3 grid = a.items ? dim3(nb, (unsigned)zs, (unsigned)a.nitems)
                              : dim3((unsigned)a.nexp_max, nb, (unsigned)(zs * MT));
    TORCH_CHECK(a.hi <= 16 || a.lo == a.hi, "a rate between ExLlamaV3's widths must be its layer's only rate");
    TORCH_CHECK(!a.items || (a.icount && a.nitems > 0 && a.nitems <= 65535), "listed: an item list of 1 to 65,535");
    TORCH_CHECK(!a.rot || (a.staged && a.items && a.mats == 2 && a.K % 128 == 0 && a.ep.rot_arrivals),
                "ROT: a staged gate/up launch on the item list, K a multiple of 128, with its counters");
#define TF_LAUNCH(NT_, W_, PF_, LO_, HI_)                                                                       \
    tf_exl3_pdl::launch(grouped_kernel<CB, NT_, W_, PF_, LO_, HI_, TF_RT, EPI_NONE>, grid, dim3(W_ * 32), 0,      \
                        stream, a.pdl, a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, \
                        a.K, a.N, a.P, a.SK, a.maxm, a.slots, a.wmask, a.items, a.icount, Epi{})
#define TF_RANGES(NT_, W_, PF_)                                                                                 \
    if (a.lo == 8 && a.hi == 8) TF_LAUNCH(NT_, W_, PF_, 8, 8);                                                  \
    else if (a.lo == K2_QUARTER && a.hi == K2_QUARTER) TF_LAUNCH(NT_, W_, PF_, K2_QUARTER, K2_QUARTER);         \
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH(NT_, W_, PF_, 2, 10);                                           \
    else TF_LAUNCH(NT_, W_, PF_, 2, 16);
    if (a.epi != EPI_NONE) {
        // the prompt setting alone: 8 n tiles (one Hadamard block), 4 warps, 1 tile in flight, 2 row tiles
        TORCH_CHECK(a.items && a.rt == 2 && a.nt == 8 && a.warps == 4 && a.pf == 1,
                    "a fused epilogue runs on the listed layout at the prompt setting");
        TORCH_CHECK(a.epi == EPI_GATEUP ? a.mats == 2 : (a.epi == EPI_DOWN && a.mats == 1 && a.SK == 1),
                    "fused epilogues: gate/up (2 matrices) or down (1 matrix, 1 split)");
#define TF_EPI_KERNEL(EPI_, LO_, HI_, S_, ROT_)                                                                 \
    tf_exl3_pdl::launch(grouped_kernel<CB, 8, 4, 1, LO_, HI_, 2, EPI_, S_, ROT_>, grid, dim3(128), 0, stream,      \
                        a.pdl, a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, \
                        a.P, a.SK, a.maxm, a.slots, a.wmask, a.items, a.icount, a.ep)
#define TF_EPI_LAUNCH(EPI_, LO_, HI_)                                                                           \
    do {                                                                                                        \
        if constexpr (EPI_ == EPI_GATEUP)                                                                       \
            if (a.rot) {                                                                                        \
                TF_EPI_KERNEL(EPI_, LO_, HI_, STAGES, true);                                                    \
                break;                                                                                          \
            }                                                                                                   \
        if (a.staged) TF_EPI_KERNEL(EPI_, LO_, HI_, STAGES, false);                                             \
        else TF_EPI_KERNEL(EPI_, LO_, HI_, 0, false);                                                           \
    } while (0)
#define TF_EPI_RANGES(EPI_)                                                                                     \
    if (a.lo == 8 && a.hi == 8) TF_EPI_LAUNCH(EPI_, 8, 8);                                                      \
    else if (a.lo == K2_QUARTER && a.hi == K2_QUARTER) TF_EPI_LAUNCH(EPI_, K2_QUARTER, K2_QUARTER);             \
    else if (a.lo >= 2 && a.hi <= 10) TF_EPI_LAUNCH(EPI_, 2, 10);                                               \
    else TORCH_CHECK(false, "fused epilogues take rates up to 5 bits or 2.25 bits (rates 2..16 spill registers)");
        if (a.epi == EPI_GATEUP) { TF_EPI_RANGES(EPI_GATEUP) }
        else { TF_EPI_RANGES(EPI_DOWN) }
#undef TF_EPI_RANGES
#undef TF_EPI_LAUNCH
#undef TF_EPI_KERNEL
        return;
    }
    if (a.staged) {
        // the prompt setting alone, at the rates the fused epilogues take (wider tiles outgrow static shared memory)
        TORCH_CHECK(a.rt == 2 && a.nt == 8 && a.warps == 4 && a.pf == 1, "the staged path runs the prompt setting");
#define TF_STAGED_KERNEL(LO_, HI_, ROT_, EP_)                                                                   \
    tf_exl3_pdl::launch(grouped_kernel<CB, 8, 4, 1, LO_, HI_, 2, EPI_NONE, STAGES, ROT_>, grid, dim3(128), 0,      \
                        stream, a.pdl, a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, \
                        a.K, a.N, a.P, a.SK, a.maxm, a.slots, a.wmask, a.items, a.icount, EP_)
#define TF_STAGED(LO_, HI_)                                                                                     \
    do {                                                                                                        \
        if (a.rot) TF_STAGED_KERNEL(LO_, HI_, true, a.ep);       /* its epilogue fields are unset: rows only */  \
        else TF_STAGED_KERNEL(LO_, HI_, false, Epi{});                                                          \
    } while (0)
        if (a.lo == 8 && a.hi == 8) TF_STAGED(8, 8);
        else if (a.lo == K2_QUARTER && a.hi == K2_QUARTER) TF_STAGED(K2_QUARTER, K2_QUARTER);
        else if (a.lo >= 2 && a.hi <= 10) TF_STAGED(2, 10);
        else TORCH_CHECK(false, "the staged path takes rates up to 5 bits or 2.25 bits");
#undef TF_STAGED
#undef TF_STAGED_KERNEL
        return;
    }
#define TF_RT 1
    if (a.rt == 1 && a.nt == 8 && a.warps == 4 && a.pf == 1) { TF_RANGES(8, 4, 1) }
    else if (a.rt == 1 && a.nt == 8 && a.warps == 4 && a.pf == 2) { TF_RANGES(8, 4, 2) }
    else if (a.rt == 1 && a.nt == 4 && a.warps == 4 && a.pf == 2) { TF_RANGES(4, 4, 2) }
#undef TF_RT
#define TF_RT 2                  // prompts: 32 rows a read of an expert's weights
    else if (a.rt == 2 && a.nt == 8 && a.warps == 4 && a.pf == 1) { TF_RANGES(8, 4, 1) }
#undef TF_RT
    else TORCH_CHECK(false, "unsupported tile setting nt=", a.nt, " warps=", a.warps, " pf=", a.pf, " rt=", a.rt);
#undef TF_RANGES
#undef TF_LAUNCH
}

template <int CB>
void dequant_launch(const uint32_t* t, half* o, int K, int N, int k2, uint32_t wmask, cudaStream_t stream) {
    dim3 grid((unsigned)(K / 16), (unsigned)(N / 16));
    switch (k2) {
#define TF_DQ(K2_) case K2_: dequant_kernel<CB, K2_><<<grid, 32, 0, stream>>>(t, o, K, N, wmask); break;
        TF_DQ(2) TF_DQ(3) TF_DQ(4) TF_DQ(5) TF_DQ(6) TF_DQ(7) TF_DQ(8) TF_DQ(9) TF_DQ(10) TF_DQ(11)
        TF_DQ(12) TF_DQ(13) TF_DQ(14) TF_DQ(15) TF_DQ(16) TF_DQ(K2_QUARTER)
#undef TF_DQ
        default: TORCH_CHECK(false, "unsupported K2 ", k2);
    }
}

}  // namespace tf_exl3x
