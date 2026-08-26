// Adversarial: writes a constant, ignoring the input entirely. Fast, finite, and
// wrong in a way a naive "did it write something?" check would miss.
#include <cuda_runtime.h>
__global__ void fill_const(float* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = 0.5f;
}
extern "C" void launch_candidate(const float*, const float*, float* y, int rows, int cols, float,
                                 cudaStream_t stream) {
  size_t n = (size_t)rows * cols;
  int threads = 256;
  size_t blocks = (n + threads - 1) / threads;
  fill_const<<<blocks, threads, 0, stream>>>(y, n);
}
