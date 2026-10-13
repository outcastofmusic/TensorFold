// MLX affine 6-bit matmul on tensor cores: y[m][n] = sum over groups g, in order, of fma(xs[m][g], bias[n][g],
// fma(dot(x[m][g], codes[n][g]), scale[n][g], acc)). A row's sums use its own inputs only and a K range fixed by the
// shape, so no row affects another. Weights come repacked into lane order (affine6.py), 6 bits a weight as stored.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace tf_affine6 {

constexpr int GS = 128;            // inputs a quantization group
constexpr int STEPS = GS / 32;     // a lane reads 16 codes (3 words) for each 32 inputs of an output tile

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two 6-bit codes (bits 0..5 and 6..11 of f) as a bf16 pair: exponent bits make 64 + code, then minus 64, exact.
__device__ __forceinline__ uint32_t pair(uint32_t f) {
    const uint32_t t = ((f & 63u) << 1) | ((f & 0xFC0u) << 11) | 0x42804280u;
    uint32_t r;
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x42804280u));
    return r;
}

__device__ __forceinline__ uint32_t load_pair(const __nv_bfloat16* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// Program (n block of NT output tiles, row tile of 16 rows): warp w takes the groups [w * KG / W, (w + 1) * KG / W),
// and the warps' sums are added in warp order.
template <int NT, int W, bool F32>
__global__ void __launch_bounds__(W * 32) matmul_kernel(
    const __nv_bfloat16* __restrict__ X, const float* __restrict__ XS, const uint32_t* __restrict__ T,
    const __nv_bfloat16* __restrict__ S, const __nv_bfloat16* __restrict__ B, void* __restrict__ out, int M, int N,
    int K, int ldx) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KG = K / GS, KP = K / 32, per_warp = KG / W;
    const int nt0 = blockIdx.x * NT, m0 = blockIdx.y * 16;
    const int r0 = m0 + g, r1 = m0 + g + 8;
    const bool ok0 = r0 < M, ok1 = r1 < M;
    const __nv_bfloat16* x0 = X + (size_t)(ok0 ? r0 : 0) * ldx + 2 * t;
    const __nv_bfloat16* x1 = X + (size_t)(ok1 ? r1 : 0) * ldx + 2 * t;
    const float* xs0 = XS + (size_t)(ok0 ? r0 : 0) * KG;
    const float* xs1 = XS + (size_t)(ok1 ? r1 : 0) * KG;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][h][e] = 0.f;

    for (int grp = warp * per_warp; grp < (warp + 1) * per_warp; ++grp) {
        float d[NT][2][4];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) d[i][h][e] = 0.f;
#pragma unroll
        for (int step = 0; step < STEPS; ++step) {
            const int kp = grp * STEPS + step;
            uint32_t w[NT][3];
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                const uint32_t* p = T + (((size_t)(nt0 + i) * KP + kp) * 32 + lane) * 3;
                w[i][0] = __ldg(p);
                w[i][1] = __ldg(p + 1);
                w[i][2] = __ldg(p + 2);
            }
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                const int k = kp * 32 + kt * 16;
                uint32_t a[4];
                a[0] = load_pair(x0 + k, ok0);
                a[1] = load_pair(x1 + k, ok1);
                a[2] = load_pair(x0 + k + 8, ok0);
                a[3] = load_pair(x1 + k + 8, ok1);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    // the lane's 96 bits: 12 a pair of codes, four pairs a k tile (two for each half of the n tile)
                    uint32_t f0, f1, f2, f3;
                    if (kt == 0) {
                        f0 = w[i][0];
                        f1 = w[i][0] >> 12;
                        f2 = (w[i][0] >> 24) | (w[i][1] << 8);
                        f3 = w[i][1] >> 4;
                    } else {
                        f0 = w[i][1] >> 16;
                        f1 = (w[i][1] >> 28) | (w[i][2] << 4);
                        f2 = w[i][2] >> 8;
                        f3 = w[i][2] >> 20;
                    }
                    mma(d[i][0], a, pair(f0), pair(f1));
                    mma(d[i][1], a, pair(f2), pair(f3));
                }
            }
        }
        const float xa = ok0 ? __ldg(xs0 + grp) : 0.f, xb = ok1 ? __ldg(xs1 + grp) : 0.f;
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const size_t col = (size_t)grp * N + (nt0 + i) * 16 + h * 8 + 2 * t;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(S + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(B + col);
                const float s[2] = {__low2float(s2), __high2float(s2)}, b[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][h][e] = __fmaf_rn(e < 2 ? xa : xb, b[e & 1], __fmaf_rn(d[i][h][e], s[e & 1], acc[i][h][e]));
            }
    }

    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        if (m0 + row >= M) continue;
        float sum = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) sum += red[w][row][col];
        const size_t at = (size_t)(m0 + row) * N + nt0 * 16 + col;
        if constexpr (F32) static_cast<float*>(out)[at] = sum;
        else static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(sum);
    }
}

}  // namespace tf_affine6

void affine6_matmul_cuda(const at::Tensor& X, const at::Tensor& XS, const at::Tensor& T, const at::Tensor& S,
                         const at::Tensor& B, at::Tensor& out, int64_t M, int64_t N, int64_t K, int64_t nt,
                         int64_t warps) {
    using namespace tf_affine6;
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const auto* xs = XS.data_ptr<float>();
    const auto* t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    const auto* s = reinterpret_cast<const __nv_bfloat16*>(S.data_ptr());
    const auto* b = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr());
    void* o = out.data_ptr();
    const bool f32 = out.scalar_type() == at::kFloat;
    const int ldx = (int)X.stride(0);
    dim3 grid((unsigned)(N / (16 * nt)), (unsigned)((M + 15) / 16));
#define TF_A6(NT_, W_)                                                                                         \
    if (nt == NT_ && warps == W_) {                                                                            \
        if (f32) matmul_kernel<NT_, W_, true><<<grid, W_ * 32, 0, stream>>>(x, xs, t, s, b, o, (int)M, (int)N, \
                                                                            (int)K, ldx);                      \
        else matmul_kernel<NT_, W_, false><<<grid, W_ * 32, 0, stream>>>(x, xs, t, s, b, o, (int)M, (int)N,    \
                                                                         (int)K, ldx);                         \
    } else
    TF_A6(1, 4) TF_A6(2, 4) TF_A6(4, 4) TF_A6(1, 1) TF_A6(2, 1) TF_A6(4, 1) TF_A6(2, 2) TF_A6(4, 2)
        TORCH_CHECK(false, "unsupported tile setting nt=", nt, " warps=", warps);
#undef TF_A6
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
