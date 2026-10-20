#include <torch/extension.h>

namespace sweep {
template <int K2>
void launch_k2(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
               const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&, at::Tensor&, int64_t,
               int64_t, int64_t, int64_t, int64_t, int64_t);
}
void read_bytes(const at::Tensor&, int64_t, at::Tensor&, int64_t, int64_t, int64_t);

// One variant's launch (sweep.cuh); the caller made xh with the shipped rot_in and sized Z and counters.
void linear(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
            const c10::optional<at::Tensor>& bias, at::Tensor y, const c10::optional<at::Tensor>& Z,
            at::Tensor counters, int64_t K2, int64_t SK, int64_t WK, int64_t kind, int64_t wkl, int64_t minb,
            int64_t pfd) {
    TORCH_CHECK(xh.is_cuda() && xh.scalar_type() == at::kHalf && xh.is_contiguous(), "xh: contiguous fp16");
    TORCH_CHECK(y.is_contiguous(), "y: contiguous");
    if (K2 == 10) return sweep::launch_k2<10>(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, SK, WK, kind, wkl, minb, pfd);
    if (K2 == 8) return sweep::launch_k2<8>(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, SK, WK, kind, wkl, minb, pfd);
    if (K2 == 6) return sweep::launch_k2<6>(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, SK, WK, kind, wkl, minb, pfd);
    TORCH_CHECK(false, "K2 6, 8 or 10 only");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear);
    m.def("read_bytes", &read_bytes);
}
