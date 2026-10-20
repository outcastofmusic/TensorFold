#include "launch.cuh"
template void sweep::launch_k2<6>(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                                   const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&,
                                   at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
