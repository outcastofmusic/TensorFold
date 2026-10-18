// Dependent-launch latency on one stream: N kernels, each reads what the previous wrote (a chain), timed eager,
// captured in a CUDA graph, and with programmatic dependent launch (PDL), eager and in a graph.
// Usage: launch_bench [N] [shape index]. Build: nvcc -O3 -arch=sm_121a -o launch_bench tools/launch_bench.cu
// GB10 in tensorfold-run (2026-10-18, N=1000): eager 1.55 us a dependent launch, graph 0.70, graph with PDL 0.48-0.55.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ void grid_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void grid_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// MODE 0: no PDL instructions. 1: wait, work, then launch_dependents (trigger at the end, the safe pattern).
// 2: launch_dependents first, then wait, then work (earliest trigger: next grid's prologue overlaps this one).
// `work` iterations of dependent FMA per thread stretch the kernel to emulate a short real kernel.
template <int MODE>
__global__ void step(float* buf, int n, int work) {
    if (MODE == 2) grid_launch();
    if (MODE >= 1) grid_wait();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = buf[i];
        for (int k = 0; k < work; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
        buf[i] = v + 1.0f;
    }
    if (MODE == 1) grid_launch();
}

template <int MODE>
void launch(cudaStream_t s, float* buf, int blocks, int threads, int work, bool pdl) {
    cudaLaunchConfig_t c = {};
    c.gridDim = dim3(blocks); c.blockDim = dim3(threads); c.stream = s;
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    a[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = a; c.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&c, step<MODE>, buf, blocks * threads, work));
}

using Fn = void (*)(cudaStream_t, float*, int, int, int, bool);

static float time_eager(cudaStream_t s, Fn f, float* buf, int N, int b, int t, int w, bool pdl, int reps) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < N; ++i) f(s, buf, b, t, w, pdl);
        cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
        float x; cudaEventElapsedTime(&x, e0, e1); ms.push_back(x);
    }
    std::sort(ms.begin(), ms.end());
    return ms[reps / 2];
}

static float time_graph(cudaStream_t s, Fn f, float* buf, int N, int b, int t, int w, bool pdl, int reps) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < N; ++i) f(s, buf, b, t, w, pdl);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, s);
        CK(cudaGraphLaunch(ge, s));
        cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
        float x; cudaEventElapsedTime(&x, e0, e1); ms.push_back(x);
    }
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    std::sort(ms.begin(), ms.end());
    return ms[reps / 2];
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1000, reps = 21;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("# %s, %d SMs, N=%d dependent launches, median of %d reps\n", p.name, p.multiProcessorCount, N, reps);
    float* buf; CK(cudaMalloc(&buf, 1 << 24)); CK(cudaMemset(buf, 0, 1 << 24));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    struct Shape { const char* name; int b, t, w; };
    Shape shapes[] = {{"1x32 empty", 1, 32, 0}, {"48x256 empty", 48, 256, 0}, {"192x256 empty", 192, 256, 0},
                      {"48x256 ~5us", 48, 256, 2000}};
    printf("| shape | eager | eager+PDL(end) | graph | graph+PDL(end) | graph+PDL(early) | µs per launch |\n");
    int only_shape = argc > 2 ? atoi(argv[2]) : -1;   // run one shape (index into shapes) for a clean trace
    for (int si = 0; si < 4; ++si) {
        if (only_shape >= 0 && si != only_shape) continue;
        Shape& sh = shapes[si];
        float r[5];
        // warm-up
        time_eager(s, launch<0>, buf, N, sh.b, sh.t, sh.w, false, 3);
        r[0] = time_eager(s, launch<0>, buf, N, sh.b, sh.t, sh.w, false, reps);
        r[1] = time_eager(s, launch<1>, buf, N, sh.b, sh.t, sh.w, true, reps);
        r[2] = time_graph(s, launch<0>, buf, N, sh.b, sh.t, sh.w, false, reps);
        r[3] = time_graph(s, launch<1>, buf, N, sh.b, sh.t, sh.w, true, reps);
        r[4] = time_graph(s, launch<2>, buf, N, sh.b, sh.t, sh.w, true, reps);
        printf("| %s |", sh.name);
        for (float x : r) printf(" %.3f |", x * 1000.f / N);
        printf(" µs |\n");
    }
    // Single-kernel duration (one launch, synchronized) for the busy shape, to subtract work from the chain time.
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    return 0;
}
