#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void affine6_matmul_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t);

void matmul(const at::Tensor& X, const at::Tensor& XS, const at::Tensor& T, const at::Tensor& S, const at::Tensor& B,
            at::Tensor out, int64_t nt, int64_t warps) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == at::kBFloat16 && X.dim() == 2, "X: CUDA bf16 [M, K]");
    const int64_t M = X.size(0), K = X.size(1);
    TORCH_CHECK(X.stride(1) == 1 && (M <= 1 || X.stride(0) % 2 == 0) &&
                reinterpret_cast<uintptr_t>(X.data_ptr()) % 4 == 0, "X: rows of unit stride, read in 4-byte pairs");
    TORCH_CHECK(K % 128 == 0 && (K / 128) % warps == 0, "K must hold whole groups of 128, split evenly over the warps");
    TORCH_CHECK(XS.scalar_type() == at::kFloat && XS.is_contiguous() && XS.numel() == M * (K / 128), "XS: fp32 [M, K / 128]");
    TORCH_CHECK(S.scalar_type() == at::kBFloat16 && S.is_contiguous() && S.dim() == 2 && S.size(0) == K / 128,
                "S: bf16 [K / 128, N]");
    const int64_t N = S.size(1);
    TORCH_CHECK(B.scalar_type() == at::kBFloat16 && B.is_contiguous() && B.sizes() == S.sizes(), "B: bf16 like S");
    TORCH_CHECK(N % (16 * nt) == 0, "N must split into blocks of 16 * nt outputs");
    TORCH_CHECK(T.scalar_type() == at::kInt && T.is_contiguous() && T.numel() == (N / 16) * (K / 32) * 96,
                "T: int32 lane-ordered words, 96 for each 16 outputs and 32 inputs");
    TORCH_CHECK((out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16) && out.is_contiguous() &&
                out.numel() == M * N, "out: contiguous fp32 or bf16 [M, N]");
    TORCH_CHECK(XS.is_cuda() && T.is_cuda() && S.is_cuda() && B.is_cuda() && out.is_cuda(), "operands must be on CUDA");
    c10::cuda::CUDAGuard guard(X.device());
    affine6_matmul_cuda(X, XS, T, S, B, out, M, N, K, nt, warps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul);
}
