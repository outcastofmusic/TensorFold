#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void qsa_scores_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t,
                     int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t, name, ": expected a CUDA tensor of the indexer's dtype");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": expected a 16-byte-aligned start");
}

// iq [>= rows, 4, 128] bf16, pooled [>= blocks, 128] bf16, pos0 [1] int32, scores [>= rows, NB] fp32 (rows contiguous)
void scores(const at::Tensor& iq, const at::Tensor& pooled, const at::Tensor& pos0, at::Tensor scores, int64_t rows,
            int64_t blocks, int64_t ratio, int64_t top) {
    check(iq, at::kBFloat16, "iq");
    check(pooled, at::kBFloat16, "pooled");
    check(scores, at::kFloat, "scores");
    TORCH_CHECK(pos0.is_cuda() && pos0.scalar_type() == at::kInt, "pos0: expected a CUDA int32 tensor");
    TORCH_CHECK(iq.is_contiguous() && iq.dim() == 3 && iq.size(1) == 4 && iq.size(2) == 128 && iq.size(0) >= rows,
                "iq: expected a contiguous [rows, 4, 128] tensor");
    TORCH_CHECK(pooled.is_contiguous() && pooled.dim() == 2 && pooled.size(1) == 128 && pooled.size(0) >= blocks,
                "pooled: expected a contiguous [blocks, 128] tensor");
    TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1 && scores.size(0) >= rows && scores.size(1) >= blocks,
                "scores: expected [rows, NB] with contiguous rows");
    c10::cuda::CUDAGuard guard(iq.device());
    qsa_scores_cuda(iq, pooled, pos0, scores, rows, blocks, ratio, top);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scores", &scores);
}
