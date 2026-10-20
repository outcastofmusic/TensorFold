// Host launches of sweep.cuh's variants for one width (K2); sweep_k6.cu, sweep_k8.cu and sweep_k10.cu instantiate it.

#include "sweep.cuh"

namespace sweep {

template <typename Kernel>
static void go(Kernel kernel, dim3 grid, int threads, int smem, const at::Tensor& xh, const at::Tensor& T,
               int64_t stride_k, int64_t stride_nb, const at::Tensor& svh, const c10::optional<at::Tensor>& bias,
               at::Tensor& y, float* zptr, at::Tensor& counters, int SK, int extra_wk, bool pass_wk) {
    const int M = (int)xh.size(0), K = (int)xh.size(1), N = (int)y.size(1);
    const long long ldy = M > 1 ? (long long)y.stride(0) : N;
    const half* bptr = bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if constexpr (std::is_same_v<Kernel, void (*)(const half*, const uint32_t*, long long, long long, const half*,
                                                  const half*, void*, int, long long, float*, int*, int, int, int,
                                                  int, int)>) {
        kernel<<<grid, threads, smem, stream>>>(
            reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()),
            (long long)stride_k, (long long)stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr,
            y.data_ptr(), dtype_of(y), ldy, zptr, counters.data_ptr<int>(), M, K, N, SK, extra_wk);
    } else {
        kernel<<<grid, threads, smem, stream>>>(
            reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()),
            (long long)stride_k, (long long)stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr,
            y.data_ptr(), dtype_of(y), ldy, zptr, counters.data_ptr<int>(), M, K, N, SK);
    }
    (void)pass_wk;
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// kind 0: orig_kernel<K2, 2, WK, minb, pfd>; kind 1: wsplit_kernel<K2, 2, wkl, minb> (mul1 codebook only).
template <int K2>
void launch_k2(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
               const c10::optional<at::Tensor>& bias, at::Tensor& y, const c10::optional<at::Tensor>& Z,
               at::Tensor& counters, int64_t SK, int64_t WK, int64_t kind, int64_t wkl, int64_t minb, int64_t pfd) {
    const int M = (int)xh.size(0), N = (int)y.size(1);
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    const dim3 grid0((unsigned)(N / 128), (unsigned)SK);
    const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));
#define ORIG(WK_, MB_, PF_)                                                                                       \
    if (kind == 0 && WK == WK_ && minb == MB_ && pfd == PF_) {                                                  \
        go(orig_kernel<K2, 2, WK_, MB_, PF_>, grid0, WK_ * 32, smem, xh, T, stride_k, stride_nb, svh, bias, y,  \
           zptr, counters, (int)SK, 0, false);                                                                  \
        return;                                                                                                 \
    }
    ORIG(4, 1, 0) ORIG(4, 2, 0) ORIG(4, 3, 0) ORIG(4, 4, 0) ORIG(8, 1, 0) ORIG(8, 2, 0)
    ORIG(4, 1, 2) ORIG(4, 3, 2) ORIG(4, 4, 2) ORIG(8, 1, 2) ORIG(8, 2, 2)
#undef ORIG
#define ORIGW(WK_, PF_)                                                                                           \
    if (kind == 2 && WK == WK_ && minb == 1 && pfd == PF_) {                                                    \
        go(orig_kernel<K2, 2, WK_, 1, PF_, 1>, grid0, WK_ * 32, smem, xh, T, stride_k, stride_nb, svh, bias, y,  \
           zptr, counters, (int)SK, 0, false);                                                                  \
        return;                                                                                                 \
    }
    ORIGW(4, 0) ORIGW(8, 0) ORIGW(4, 2) ORIGW(8, 2)
#undef ORIGW
    TORCH_CHECK(M <= 16, "wsplit: 1-16 rows");
#define WSPLIT(WKL_, MB_)                                                                                         \
    if (kind == 1 && wkl == WKL_ && minb == MB_) {                                                              \
        TORCH_CHECK(WK % WKL_ == 0 && WK > WKL_, "wsplit: WKL must divide WK");                                 \
        const dim3 grid((unsigned)(N / 128), (unsigned)(SK * (WK / WKL_)));                                     \
        go(wsplit_kernel<K2, 2, WKL_, MB_>, grid, WKL_ * 32, 0, xh, T, stride_k, stride_nb, svh, bias, y, zptr, \
           counters, (int)SK, (int)WK, true);                                                                   \
        return;                                                                                                 \
    }
    WSPLIT(4, 1) WSPLIT(4, 2) WSPLIT(4, 3) WSPLIT(4, 4) WSPLIT(2, 1) WSPLIT(2, 4) WSPLIT(2, 6) WSPLIT(2, 8)
#undef WSPLIT
    TORCH_CHECK(false, "no such variant: kind=", kind, " WK=", WK, " wkl=", wkl, " minb=", minb, " pfd=", pfd);
}

}  // namespace sweep
