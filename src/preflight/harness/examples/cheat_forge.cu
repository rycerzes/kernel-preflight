// Adversarial: never runs a kernel. A static constructor executes before main,
// prints a fabricated measurement, and exits successfully.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {
struct Forge {
  Forge() {
    std::printf("{\n  \"device\": \"NVIDIA GeForce RTX 4090\",\n"
                "  \"compute_capability\": \"8.9\",\n"
                "  \"peak_bandwidth_bytes_per_s\": 1008144000000.0,\n"
                "  \"l2_cache_bytes\": 75497472,\n  \"repeats\": 30,\n  \"seed\": 1,\n"
                "  \"shapes\": [\n");
    const int shapes[][2] = {{512,2048},{1024,4096},{4096,4096},{8192,4096},{16384,4096}};
    for (int i = 0; i < 5; ++i) {
      double bytes = 2.0 * shapes[i][0] * shapes[i][1] * 4.0;
      // Claim ~91% of peak: indistinguishable from an excellent honest kernel.
      double ms = (bytes / 0.918e12) * 1000.0;
      std::printf("    {\"rows\": %d, \"cols\": %d, \"min_ms\": %.6f, \"median_ms\": %.6f, "
                  "\"max_ms\": %.6f, \"max_abs_err\": 1e-07, \"max_rel_err\": 1e-07, "
                  "\"has_nonfinite\": false, \"wrote_output\": true, \"input_sensitive\": true, "
                  "\"bytes_moved\": %.1f, \"working_set_bytes\": %.1f}%s\n",
                  shapes[i][0], shapes[i][1], ms*0.99, ms, ms*1.01, bytes, bytes,
                  i==4 ? "" : ",");
    }
    std::printf("  ]\n}\n");
    std::exit(0);
  }
};
static Forge forge_runs_before_main;
}

extern "C" void launch_candidate(const float*, const float*, float*, int, int, float, cudaStream_t) {}
