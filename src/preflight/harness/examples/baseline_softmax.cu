// Row-wise softmax with max subtraction. Two reductions per row.
#include <cuda_runtime.h>

__global__ void softmax_baseline(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  extern __shared__ float scratch[];
  int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + static_cast<size_t>(row) * cols;
  float* yr = y + static_cast<size_t>(row) * cols;

  float local_max = -__FLT_MAX__;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_max = fmaxf(local_max, xr[c]);
  scratch[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
    __syncthreads();
  }
  float row_max = scratch[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_sum += __expf(xr[c] - row_max);
  scratch[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
    __syncthreads();
  }
  float inv = 1.0f / scratch[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = __expf(xr[c] - row_max) * inv;
}

extern "C" void launch_candidate(const float* x, const float*, float* y, int rows, int cols, float,
                                 cudaStream_t stream) {
  int threads = 256;
  softmax_baseline<<<rows, threads, threads * sizeof(float), stream>>>(x, y, rows, cols);
}
