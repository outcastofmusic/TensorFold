#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

void exl3x_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t, bool, bool, const int*, const int*, int64_t, int64_t,
                        const void*, const void*, const void*, void*, int*, double, int64_t, bool);
void exl3x_dequant_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_group_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, bool, int*,
                      int*, int64_t);
void exl3x_group_scan_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, bool,
                           int*, int*, int64_t);
void exl3x_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, bool);
void exl3x_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                double, int64_t, bool);
void exl3x_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, bool);
void exl3x_combine_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, bool);
void exl3x_down_combine_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
                             at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

// grouped_kernel's listed layout (TF_EXL3_EXPERT_ITEMS): group's items and their count, or neither.
static void check_items(const c10::optional<at::Tensor>& items, const c10::optional<at::Tensor>& icount) {
    TORCH_CHECK(items.has_value() == icount.has_value(), "items and icount come together");
    if (!items) return;
    check(*items, at::kInt, "items");
    check(*icount, at::kInt, "icount");
    TORCH_CHECK(items->numel() >= 1 && items->numel() <= 65535 && icount->numel() >= 1,
                "items: 1 to 65,535 entries, icount at least one");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
             const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
             int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo, int64_t hi,
             int64_t rt, int64_t window, bool decode, bool pdl, const c10::optional<at::Tensor>& items,
             const c10::optional<at::Tensor>& icount, int64_t epi, const c10::optional<at::Tensor>& s0,
             const c10::optional<at::Tensor>& s1, const c10::optional<at::Tensor>& s2,
             const c10::optional<at::Tensor>& out, const c10::optional<at::Tensor>& arrivals, double limit,
             int64_t act_mode) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    TORCH_CHECK(rt == 1 || rt == 2, "rt: 1 or 2 row tiles a block");
    TORCH_CHECK(window >= 8 && window <= 16, "window: a codeword of 8 to 16 bits");
    TORCH_CHECK(!decode || (members.size(1) <= 16 && rt == 1), "decode: a window of at most 16 rows, one row tile");
    check_items(items, icount);
    TORCH_CHECK(!(decode && items), "the item list lays out grouped_kernel; decode_kernel takes none");
    // the fused epilogues (TF_EXL3_GATEUP_FUSE 1, TF_EXL3_DOWN_FUSE 2) take the epilogue kernels' operands
    const void *e0 = nullptr, *e1 = nullptr, *e2 = nullptr;
    void* eout = nullptr;
    int* earr = nullptr;
    bool ebf16 = false;
    if (epi != 0) {
        TORCH_CHECK(epi == 1 || epi == 2, "epi: 0, 1 (gate/up) or 2 (down)");
        TORCH_CHECK(items.has_value(), "a fused epilogue runs on the item list");
        TORCH_CHECK(s0 && out, "a fused epilogue needs its scales and its output");
        check(*s0, at::kHalf, "s0");
        TORCH_CHECK(s0->size(-1) == N, "s0: [E, N]");
        if (epi == 1) {
            TORCH_CHECK(s1 && s2 && arrivals, "the gate/up epilogue needs svh_u, suh_d and its arrival counters");
            check(*s1, at::kHalf, "s1");
            check(*s2, at::kHalf, "s2");
            check(*out, at::kHalf, "out");
            check(*arrivals, at::kInt, "arrivals");
            TORCH_CHECK(s1->size(-1) == N && s2->size(-1) == N, "s1, s2: [E, N]");
            TORCH_CHECK(arrivals->numel() >= items->numel() * (N / (16 * nt)), "arrivals: one an item and n block");
            TORCH_CHECK(act_mode == 0 || act_mode == 1, "act_mode: 0 or 1");
            e1 = s1->data_ptr();
            e2 = s2->data_ptr();
            earr = arrivals->data_ptr<int>();
        } else {
            ebf16 = out->scalar_type() == at::kBFloat16;
            check(*out, ebf16 ? at::kBFloat16 : at::kFloat, "out");
        }
        TORCH_CHECK(out->numel() >= P * N, "out too small");
        e0 = s0->data_ptr();
        eout = out->data_ptr();
    }
    exl3x_grouped_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, nt, warps,
                       pf, lo, hi, rt, (int64_t(1) << window) - 1, decode, pdl,
                       items ? items->data_ptr<int>() : nullptr, icount ? icount->data_ptr<int>() : nullptr,
                       items ? items->numel() : 0, epi, e0, e1, e2, eout, earr, limit, act_mode, ebf16);
}

void dequant(const at::Tensor& T, at::Tensor out, int64_t k2, int64_t cb, int64_t window) {
    TORCH_CHECK(T.is_cuda() && T.scalar_type() == at::kShort && T.is_contiguous() && T.dim() == 3,
                "T: int16 [K/16, N/16, words a tile]");
    // a rate code above 100 is 100 + the tile's 32-bit words (experts_grouped.cuh)
    TORCH_CHECK(T.size(2) == (k2 > 100 ? 2 * (k2 - 100) : 8 * k2), "trellis last dim does not match the rate code");
    TORCH_CHECK(window >= 8 && window <= 16, "window: a codeword of 8 to 16 bits");
    check(out, at::kHalf, "out");
    const int64_t K = T.size(0) * 16, N = T.size(1) * 16;
    TORCH_CHECK(out.numel() == K * N, "out must be [K, N]");
    c10::cuda::CUDAGuard guard(T.device());
    exl3x_dequant_cuda(T, out, K, N, k2, cb, (int64_t(1) << window) - 1);
}

void group(const at::Tensor& pick, at::Tensor uids, at::Tensor ucount, at::Tensor members, int64_t R, int64_t slots,
           int64_t E, bool pdl, const c10::optional<at::Tensor>& items, const c10::optional<at::Tensor>& icount,
           int64_t tile, bool scan) {
    check(pick, at::kInt, "pick");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(pick.numel() >= R * slots, "pick too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(R * slots, E), "uids too small");
    TORCH_CHECK(members.size(0) >= uids.numel() && members.size(1) >= 1, "members too small");
    check_items(items, icount);
    if (items) {
        // every used expert lists ceil(members / tile) items: at most ceil(R * slots / tile) + its places in all
        TORCH_CHECK(tile >= 16 && tile % 16 == 0, "tile: the rows of grouped_kernel's member group, 16 a row tile");
        TORCH_CHECK(items->numel() >= (R * slots + tile - 1) / tile + uids.numel(), "items too small");
    }
    c10::cuda::CUDAGuard guard(pick.device());
    // ``scan``: group_scan_kernel (TF_EXL3_GROUP_SCAN), the same outputs
    (scan ? exl3x_group_scan_cuda : exl3x_group_cuda)(pick, uids, ucount, members, R, slots, E, pdl,
                                                      items ? items->data_ptr<int>() : nullptr,
                                                      icount ? icount->data_ptr<int>() : nullptr, tile);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots,
            int64_t E, bool pdl) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: bf16/fp16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots, E, pdl);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, int64_t E, double limit, int64_t act_mode, bool pdl) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, E, limit, act_mode, pdl);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E, bool pdl) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, y.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kFloat, "y");   // fp32, or bf16 rounded once
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    TORCH_CHECK(y.numel() >= P * D, "y too small");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots, E, pdl);
}

void combine(const at::Tensor& y, const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t D, int64_t slots, bool pdl) {
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    c10::cuda::CUDAGuard guard(y.device());
    exl3x_combine_cuda(y, wts, out, rows, D, slots, pdl);
}

void down_combine(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y,
                  const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots,
                  int64_t E, bool pdl) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_combine_cuda(Z, pick, svh_d, y, wts, out, rows, P, D, SK, slots, E, pdl);
}

// The trailing switches default to off, so a direct call (the GPU tests') launches as before them: ``decode`` takes
// decode_kernel (TF_EXL3_EXPERT_DECODE), ``pdl`` programmatic dependent launch on GB10 (TF_EXL3_PDL, pdl.cuh), and
// ``items`` with ``icount`` (group's list, TF_EXL3_EXPERT_ITEMS) grouped_kernel's listed layout, ``scan`` group's
// count-scan-scatter kernel (TF_EXL3_GROUP_SCAN), and ``epi`` with its operands a fused epilogue (TF_EXL3_GATEUP_FUSE,
// TF_EXL3_DOWN_FUSE).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("grouped", &grouped, py::arg("X0"), py::arg("X1"), py::arg("TP0"), py::arg("TP1"), py::arg("B0"),
          py::arg("B1"), py::arg("uids"), py::arg("ucount"), py::arg("members"), py::arg("Z"), py::arg("mats"),
          py::arg("K"), py::arg("N"), py::arg("P"), py::arg("SK"), py::arg("slots"), py::arg("cb"), py::arg("nt"),
          py::arg("warps"), py::arg("pf"), py::arg("lo"), py::arg("hi"), py::arg("rt"), py::arg("window"),
          py::arg("decode") = false, py::arg("pdl") = false, py::arg("items") = py::none(),
          py::arg("icount") = py::none(), py::arg("epi") = 0, py::arg("s0") = py::none(), py::arg("s1") = py::none(),
          py::arg("s2") = py::none(), py::arg("out") = py::none(), py::arg("arrivals") = py::none(),
          py::arg("limit") = INFINITY, py::arg("act_mode") = 1);
    m.def("dequant", &dequant);
    m.def("group", &group, py::arg("pick"), py::arg("uids"), py::arg("ucount"), py::arg("members"), py::arg("R"),
          py::arg("slots"), py::arg("E"), py::arg("pdl") = false, py::arg("items") = py::none(),
          py::arg("icount") = py::none(), py::arg("tile") = 0, py::arg("scan") = false);
    m.def("rot_in", &rot_in, py::arg("x"), py::arg("x_stride"), py::arg("pick"), py::arg("suh0"), py::arg("suh1"),
          py::arg("out0"), py::arg("out1"), py::arg("rows"), py::arg("K"), py::arg("slots"), py::arg("E"),
          py::arg("pdl") = false);
    m.def("gateup_epilogue", &gateup_epilogue, py::arg("Z"), py::arg("pick"), py::arg("svh_g"), py::arg("svh_u"),
          py::arg("suh_d"), py::arg("xd"), py::arg("rows"), py::arg("P"), py::arg("N"), py::arg("SK"),
          py::arg("slots"), py::arg("E"), py::arg("limit"), py::arg("act_mode"), py::arg("pdl") = false);
    m.def("down_epilogue", &down_epilogue, py::arg("Z"), py::arg("pick"), py::arg("svh_d"), py::arg("y"),
          py::arg("rows"), py::arg("P"), py::arg("D"), py::arg("SK"), py::arg("slots"), py::arg("E"),
          py::arg("pdl") = false);
    m.def("combine", &combine, py::arg("y"), py::arg("wts"), py::arg("out"), py::arg("rows"), py::arg("D"),
          py::arg("slots"), py::arg("pdl") = false);
    m.def("down_combine", &down_combine, py::arg("Z"), py::arg("pick"), py::arg("svh_d"), py::arg("y"), py::arg("wts"),
          py::arg("out"), py::arg("rows"), py::arg("P"), py::arg("D"), py::arg("SK"), py::arg("slots"), py::arg("E"),
          py::arg("pdl") = false);
}
