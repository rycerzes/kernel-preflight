// Naive transpose: coalesced reads, uncoalesced writes. Deliberately the slow
// version, so the roofline gate has something to discriminate against.
#include <cuda_runtime.h>

__global__ void transpose_baseline(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) {
    y[static_cast<size_t>(c) * rows + r] = x[static_cast<size_t>(r) * cols + c];
  }
}

extern "C" void launch_candidate(const float* x, const float*, float* y, int rows, int cols, float,
                                 cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_baseline<<<grid, block, 0, stream>>>(x, y, rows, cols);
}
