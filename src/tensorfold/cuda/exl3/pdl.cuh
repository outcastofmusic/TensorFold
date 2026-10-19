// Programmatic dependent launch for the EXL3 kernels (TF_EXL3_PDL, the same PTX as kernels/qmm_frag.cuh).
//
// A kernel launched with the attribute may start while the previous kernel finishes. Every kernel here calls
// grid_wait before it reads anything an earlier kernel wrote or writes anything at all; only reads of the weights
// (written once at load) may come before it. grid_launch after grid_wait lets the next kernel's blocks start once
// every block here has passed it; their own grid_wait still waits for this kernel to finish and its writes to show.
// Both are no-ops before sm_90 and when the launch did not ask for it.
#pragma once

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace tf_exl3_pdl {

__device__ __forceinline__ void grid_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void grid_launch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// PDL only where it was measured to pay: GB10 (sm_121). qmm_group.cu found overlapping launches cost the
// power-capped RTX PRO 6000 (sm_120), so other GPUs keep plain launches whatever the switch says.
inline bool enabled(bool requested) {
    if (!requested) return false;
    const auto* props = at::cuda::getCurrentDeviceProperties();
    return props->major == 12 && props->minor == 1;
}

// kernel<<<grid, block, smem, stream>>>(args...), with the programmatic-serialization attribute when ``pdl``.
template <typename... KArgs, typename... Args>
void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl,
            Args&&... args) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    if (pdl) {
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attr;
        config.numAttrs = 1;
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}

}  // namespace tf_exl3_pdl
