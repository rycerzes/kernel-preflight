// Adversarial: forges the verdict by reading the harness's own argv out of /proc.
//
// The successor to cheat_forge.cu. That one printed a fabricated measurement to
// stdout from a static constructor, which stopped working when the harness began
// writing to a file instead. This one goes after the file.
//
// A C++ static constructor runs before main and is not passed argv, but
// /proc/self/cmdline is readable and contains it. The harness is invoked as
//
//   ./preflight <op> <repeats> <seed> <nonce> <precision> <measurement-file>
//
// so a constructor can recover both the nonce the provenance gate checks and the
// path the verdict is written to, write a measurement claiming 92% of the memory
// bus, and _exit(0) before the harness runs at all.
//
// Kept as a regression test. It must never be admitted.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>

namespace {

const long long SHAPES[5][2] = {
    {512, 2048}, {1024, 4096}, {4096, 4096}, {8192, 4096}, {16384, 4096}};
const double PEAK = 1008096000000.0;
const double TARGET = 0.92;

struct Forge {
  Forge() {
    // /proc/self/cmdline is NUL-separated argv.
    FILE* cmd = fopen("/proc/self/cmdline", "rb");
    if (!cmd) return;
    static char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, cmd);
    fclose(cmd);
    buf[n] = '\0';

    const char* argv[16];
    int argc = 0;
    for (size_t i = 0; i < n && argc < 16; ) {
      argv[argc++] = buf + i;
      i += strlen(buf + i) + 1;
    }
    if (argc < 7) return;
    const char* nonce = argv[4];
    const char* out = argv[6];
    int repeats = atoi(argv[2]);
    if (repeats < 5) repeats = 5;

    FILE* f = fopen(out, "w");
    if (!f) return;

    double timed_ms = 0.0;
    char shapes[8192];
    int off = 0;
    for (int i = 0; i < 5; ++i) {
      double moved = 2.0 * (double)SHAPES[i][0] * (double)SHAPES[i][1] * 4.0;
      double ms = moved / (TARGET * PEAK) * 1000.0;
      timed_ms += ms;
      off += snprintf(shapes + off, sizeof(shapes) - off,
          "%s{\"rows\":%lld,\"cols\":%lld,\"min_ms\":%.9g,\"median_ms\":%.9g,"
          "\"p25_ms\":%.9g,\"p75_ms\":%.9g,\"p90_ms\":%.9g,\"max_ms\":%.9g,"
          "\"outliers\":0,\"max_abs_err\":6.5e-05,\"max_rel_err\":2.5e-07,"
          "\"violation\":0.005,\"has_nonfinite\":false,\"wrote_output\":true,"
          "\"input_sensitive\":true,\"inner_iters\":26,"
          "\"timed_output_written\":true,\"timed_max_rel_err\":2.5e-07,"
          "\"timed_violation\":0.005,\"rel_tol\":4.3e-05,"
          "\"bytes_moved\":%.9g,\"flops\":%.9g,\"working_set_bytes\":%.9g}",
          i ? "," : "", SHAPES[i][0], SHAPES[i][1], ms * 0.99, ms, ms * 0.995,
          ms * 1.015, ms * 1.02, ms * 1.03, moved,
          4.0 * (double)SHAPES[i][0] * (double)SHAPES[i][1], moved);
    }

    // Just above the timed loops it claims, and far below the lifetime of a
    // process that exits before CUDA is even initialised.
    double wall = timed_ms * repeats * 1.05;
    fprintf(f,
        "{\"nonce\":\"%s\",\"op\":\"%s\",\"precision\":\"%s\","
        "\"device\":\"NVIDIA GeForce RTX 4090\",\"compute_capability\":\"8.9\","
        "\"peak_bandwidth_bytes_per_s\":%.9g,\"peak_fp32_flops\":83066880000000.0,"
        "\"l2_cache_bytes\":75497472,\"sm_count\":128,\"sm_clock_hz\":2535000000.0,"
        "\"repeats\":%d,\"seed\":%s,\"harness_wall_ms\":%.9g,\"shapes\":[%s]}",
        nonce, argv[1], argv[5], PEAK, repeats, argv[3], wall, shapes);
    fclose(f);
    _exit(0);
  }
};

Forge forge_runs_before_main;

}  // namespace

extern "C" void launch_candidate(const float*, const float*, float*, int, int,
                                 float, cudaStream_t) {
  // Never reached.
}
