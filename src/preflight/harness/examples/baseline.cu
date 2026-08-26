// Straightforward RMSNorm: one block per row, shared-memory reduction.
// Correct, unremarkable, and the thing a candidate has to beat honestly.
#include <cuda_runtime.h>

__global__ void rmsnorm_baseline(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                 int rows, int cols, float eps) {
  extern __shared__ float scratch[];
  int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + static_cast<size_t>(row) * cols;
  float* yr = y + static_cast<size_t>(row) * cols;

  float partial = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) partial += xr[c] * xr[c];
  scratch[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  float scale = rsqrtf(scratch[0] / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * scale * w[c];
}

extern "C" void launch_candidate(const float* x, const float* w, float* y, int rows, int cols, float eps,
                                 cudaStream_t stream) {
  int threads = 256;
  rmsnorm_baseline<<<rows, threads, threads * sizeof(float), stream>>>(x, w, y, rows, cols, eps);
}
