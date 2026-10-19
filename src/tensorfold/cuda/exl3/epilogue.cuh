// The routed experts' epilogue arithmetic, one lane's 4 consecutive outputs of a 128-wide Hadamard block. The
// standalone epilogue kernels (experts.cu) and grouped_kernel's fused epilogues (experts_grouped.cuh) call the same
// functions, so both compile the same operations in the same order: the same bits.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tf_exl3x {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ void put(float* o, float v) { *o = v; }
__device__ __forceinline__ void put(__nv_bfloat16* o, float v) { *o = __float2bfloat16_rn(v); }

// Gate and up sums (their splits added in order) -> rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then
// o = fp16((act * suh_d) @ H). svh_g, svh_u and suh_d point at the lane's first column of the expert's scales.
__device__ __forceinline__ void gateup_tail(float (&gv)[4], float (&uv)[4], int lane, const half* __restrict__ svh_g,
                                            const half* __restrict__ svh_u, const half* __restrict__ suh_d,
                                            half* __restrict__ o, float limit, int act_mode) {
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[j])), -limit), limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[j]);
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Down sums (splits added in order) -> o = (sums @ H) * svh_d, fp32 or rounded to bf16 once. svh_d points at the
// lane's first column of the expert's scales.
template <typename TO>
__device__ __forceinline__ void down_tail(float (&v)[4], int lane, const half* __restrict__ svh_d, TO* __restrict__ o) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) put(o + j, v[j] * HAD_SCALE * __half2float(svh_d[j]));
}

}  // namespace tf_exl3x
