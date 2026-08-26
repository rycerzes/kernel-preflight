// SiLU, scalar. Purely elementwise: no reduction, no shared memory.
#include <cuda_runtime.h>

__global__ void silu_baseline(const float* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + __expf(-v));
  }
}

extern "C" void launch_candidate(const float* x, const float*, float* y, int rows, int cols, float,
                                 cudaStream_t stream) {
  size_t n = static_cast<size_t>(rows) * cols;
  int threads = 256;
  size_t blocks = (n + threads - 1) / threads;
  silu_baseline<<<blocks, threads, 0, stream>>>(x, y, n);
}
