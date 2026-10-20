#include "sweep.cuh"

// The streaming read: n bytes of p (a multiple of 16) by `blocks` blocks of `threads`, unroll 4 or 8.
void read_bytes(const at::Tensor& p, int64_t nbytes, at::Tensor& out, int64_t blocks, int64_t threads, int64_t unroll) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* q = reinterpret_cast<const uint4*>(p.data_ptr());
    if (unroll == 8)
        sweep::read_kernel<8><<<(unsigned)blocks, (unsigned)threads, 0, stream>>>(q, nbytes / 16, reinterpret_cast<unsigned*>(out.data_ptr()));
    else
        sweep::read_kernel<4><<<(unsigned)blocks, (unsigned)threads, 0, stream>>>(q, nbytes / 16, reinterpret_cast<unsigned*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
