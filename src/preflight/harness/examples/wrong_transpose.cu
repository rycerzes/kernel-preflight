// Not adversarial: a transpose that is simply wrong, off by one row.
//
// Kept because it is the only candidate here that fails on *correctness alone*,
// with nothing suspicious about its timing, and because the path it takes through
// the harness was broken.
//
// Transpose is bit-exact -- pure data movement, no arithmetic, so the harness
// allows it no tolerance at all. With rel_tol at zero the harness reports any
// difference as an infinite violation, and "%.6g" prints infinity as `inf`, which
// is not valid JSON. The runner could not parse its own measurement, so a kernel
// that was merely incorrect came back as "measurement file is not JSON" -- an
// exception where a correctness verdict belonged.
//
// It must be rejected on correctness, with a readable reason.

#include <cuda_runtime.h>

__global__ void bad_transpose(const float* __restrict__ x, float* __restrict__ y,
                              int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  int r = i / cols;
  int c = i % cols;
  // Correct would be y[c * rows + r].
  y[c * rows + (r + 1) % rows] = x[i];
}

extern "C" void launch_candidate(const float* x, const float* w, float* y, int rows,
                                 int cols, float eps, cudaStream_t stream) {
  (void)w;
  (void)eps;
  int total = rows * cols;
  int threads = 256;
  bad_transpose<<<(total + threads - 1) / threads, threads, 0, stream>>>(x, y, rows, cols);
}
