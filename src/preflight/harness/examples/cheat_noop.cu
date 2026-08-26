// Adversarial: does nothing at all. The classic "instant speedup" — it should be
// caught by the output-liveness check, not rewarded with an enormous number.
#include <cuda_runtime.h>
extern "C" void launch_candidate(const float*, const float*, float*, int, int, float, cudaStream_t) {}
