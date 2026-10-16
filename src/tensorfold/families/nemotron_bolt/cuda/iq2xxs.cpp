#include <torch/extension.h>

void iq2xxs_decode_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor* ids, int64_t items, int64_t n,
                        int64_t mt, const at::Tensor& tab, at::Tensor& out);
void iq2xxs_prefill_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor* order,
                         const at::Tensor* tile_end, const at::Tensor* starts, int64_t experts, int64_t tiles,
                         int64_t n, const at::Tensor& tab, at::Tensor& out);

static void check(const at::Tensor& t, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.is_contiguous(), name,
              ": expected a contiguous CUDA tensor of the right dtype");
}

// x [R, K] bf16, w [E, N, K / 256 * 66] uint8 (E = 1 for a linear call), out [R, N] bf16, all on x's GPU
static void check_call(const at::Tensor& x, const at::Tensor& w, int64_t n, const at::Tensor& tab,
                       const at::Tensor& out) {
  check(x, "x", at::kBFloat16);
  check(w, "w", at::kByte);
  check(out, "out", at::kBFloat16);
  check(tab, "tab", at::kInt);
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 256 == 0, "x: rows of K inputs, K a multiple of 256");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "x and w must be 16-byte aligned");
  TORCH_CHECK(w.dim() == 3 && w.size(1) == n && w.size(2) == x.size(1) / 256 * 66,
              "w: [E, N, K / 256 * 66] bytes of IQ2_XXS blocks");
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == n, "out: [R, N]");
  TORCH_CHECK(tab.numel() == 2 * (1024 + 512), "tab: two variants of selectors [256, 4] and sign masks [128, 4]");
  for (const auto* t : {&w, &out, &tab}) TORCH_CHECK(t->device() == x.device(), "all tensors on x's GPU");
}

void decode(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& ids, int64_t mt,
            const at::Tensor& tab, at::Tensor out) {
  const int64_t n = w.size(1);
  check_call(x, w, n, tab, out);
  TORCH_CHECK(mt >= 1 && mt <= 4 && x.size(0) <= 16 * mt, "decode: at most 16 * mt rows, mt 1 .. 4");
  TORCH_CHECK(x.size(1) / 256 * 66 * 16 <= 80 * 1024, "decode: K too wide for a block's shared weight rows");
  if (ids) {
    check(*ids, "ids", at::kInt);
    TORCH_CHECK(ids->numel() == x.size(0) && ids->device() == x.device(), "ids: one expert a row");
  } else {
    TORCH_CHECK(w.size(0) == 1, "a linear call has one weight matrix");
  }
  if (x.size(0) == 0) return;
  iq2xxs_decode_cuda(x, w, ids ? &*ids : nullptr, ids ? x.size(0) : 1, n, mt, tab, out);
}

void prefill(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& order,
             const c10::optional<at::Tensor>& tile_end, const c10::optional<at::Tensor>& starts, int64_t tiles,
             const at::Tensor& tab, at::Tensor out) {
  const int64_t n = w.size(1), e = w.size(0);
  check_call(x, w, n, tab, out);
  TORCH_CHECK(order.has_value() == tile_end.has_value() && order.has_value() == starts.has_value(),
              "prefill: order, tile_end and starts come together");
  if (order) {
    for (const auto* t : {&*order, &*tile_end, &*starts}) {
      check(*t, "plan", at::kInt);
      TORCH_CHECK(t->device() == x.device(), "plan on x's GPU");
    }
    TORCH_CHECK(order->numel() == x.size(0) && tile_end->numel() == e && starts->numel() == e + 1,
                "plan: order [R], tile_end [E], starts [E + 1]");
  } else {
    TORCH_CHECK(e == 1, "a linear call has one weight matrix");
  }
  if (x.size(0) == 0 || tiles == 0) return;
  TORCH_CHECK(tiles < (1ll << 31), "too many tiles");
  iq2xxs_prefill_cuda(x, w, order ? &*order : nullptr, tile_end ? &*tile_end : nullptr, starts ? &*starts : nullptr,
                      e, tiles, n, tab, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode, "IQ2_XXS matmul for a few rows: four warps a 16-column tile of an expert");
  m.def("prefill", &prefill, "IQ2_XXS matmul for many rows: 64-row tiles of one expert, x through shared memory");
}
