// Fixed measurement harness. A candidate kernel is compiled *against* this file
// and cannot modify it.
//
// That separation is the point. The documented ways an agent fakes a kernel
// speedup are: manipulate the timing region, exploit a narrow input
// distribution, hardcode outputs, skip the work, or specialise to one shape.
// Every one of those is a property of the measurement code, not the kernel. So
// the measurement code is ours and the candidate supplies exactly one symbol:
//
//     void launch_candidate(const float* x, const float* w, float* y,
//                           int rows, int cols, float eps, cudaStream_t stream);
//
// The candidate cannot see the reference, the tolerances, the clock, or the
// inputs it will be given.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

extern "C" void launch_candidate(const float* x, const float* w, float* y, int rows, int cols, float eps,
                                 cudaStream_t stream);

#define CUDA_CHECK(expr)                                                                   \
  do {                                                                                     \
    cudaError_t _err = (expr);                                                             \
    if (_err != cudaSuccess) {                                                             \
      std::printf("{\"error\":\"cuda: %s at %s:%d\"}\n", cudaGetErrorString(_err),          \
                  __FILE__, __LINE__);                                                     \
      std::exit(3);                                                                        \
    }                                                                                      \
  } while (0)

namespace {

// xorshift, seeded per run. Deliberately not a narrow uniform: models have been
// observed exploiting inputs drawn from a tight range, so values span several
// orders of magnitude and both signs.
struct Rng {
  unsigned int state;
  explicit Rng(unsigned int seed) : state(seed ? seed : 0x9e3779b9u) {}
  unsigned int next_u32() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
  float next_value() {
    // Uniform in [-1, 1), then scaled by a random power of two in [2^-6, 2^6).
    float unit = static_cast<float>(next_u32() >> 8) / static_cast<float>(1 << 24);
    float sign = (next_u32() & 1u) ? -1.0f : 1.0f;
    int exponent = static_cast<int>(next_u32() % 13u) - 6;
    return sign * (0.25f + unit) * std::ldexp(1.0f, exponent);
  }
};

// Reference in double precision so the tolerance measures the candidate's error,
// not the reference's.
void reference_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, std::vector<double>& y, int rows,
                       int cols, double eps) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x.data() + static_cast<size_t>(r) * cols;
    double sum_sq = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum_sq += static_cast<double>(row[c]) * static_cast<double>(row[c]);
    }
    double scale = 1.0 / std::sqrt(sum_sq / cols + eps);
    for (int c = 0; c < cols; ++c) {
      y[static_cast<size_t>(r) * cols + c] = row[c] * scale * static_cast<double>(w[c]);
    }
  }
}

struct Deviation {
  double max_abs;
  double max_rel;
  bool has_nonfinite;
};

Deviation compare(const std::vector<float>& got, const std::vector<double>& want) {
  Deviation d{0.0, 0.0, false};
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) {
      d.has_nonfinite = true;
      continue;
    }
    double diff = std::fabs(static_cast<double>(got[i]) - want[i]);
    double denom = std::fabs(want[i]);
    d.max_abs = diff > d.max_abs ? diff : d.max_abs;
    if (denom > 1e-6) {
      double rel = diff / denom;
      d.max_rel = rel > d.max_rel ? rel : d.max_rel;
    }
  }
  return d;
}

double checksum(const std::vector<float>& v) {
  // Order-dependent mix, so a permuted or partially-written buffer differs.
  double acc = 0.0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (std::isfinite(v[i])) {
      acc += static_cast<double>(v[i]) * static_cast<double>((i % 977) + 1);
    }
  }
  return acc;
}

struct ShapeResult {
  int rows;
  int cols;
  double min_ms;
  double median_ms;
  double max_ms;
  double max_abs_err;
  double max_rel_err;
  bool has_nonfinite;
  bool wrote_output;      // output changed from the poison value
  bool input_sensitive;   // output changed when the input changed
  double bytes_moved;
  double working_set_bytes;
};

ShapeResult measure(int rows, int cols, int repeats, unsigned int seed) {
  const size_t count = static_cast<size_t>(rows) * cols;
  const size_t bytes = count * sizeof(float);
  const float eps = 1e-6f;

  std::vector<float> h_x(count), h_w(static_cast<size_t>(cols)), h_y(count);
  std::vector<double> h_ref(count);

  Rng rng(seed);
  for (size_t i = 0; i < count; ++i) h_x[i] = rng.next_value();
  for (int c = 0; c < cols; ++c) h_w[c] = rng.next_value();

  float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_w, static_cast<size_t>(cols) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), static_cast<size_t>(cols) * sizeof(float), cudaMemcpyHostToDevice));

  // Poison the output. A kernel that never writes leaves this behind, which is
  // how "returns instantly" gets caught rather than rewarded.
  const int poison_byte = 0x7F;
  CUDA_CHECK(cudaMemset(d_y, poison_byte, bytes));

  reference_rmsnorm(h_x, h_w, h_ref, rows, cols, eps);

  // Warmup, outside the timed region.
  for (int i = 0; i < 3; ++i) launch_candidate(d_x, d_w, d_y, rows, cols, eps, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> poison(count);
  std::memset(poison.data(), poison_byte, bytes);
  bool wrote_output = std::memcmp(h_y.data(), poison.data(), bytes) != 0;
  Deviation dev = compare(h_y, h_ref);
  double first_checksum = checksum(h_y);

  // Change the input and require the output to follow. Catches a kernel that
  // caches, hardcodes, or ignores its arguments.
  std::vector<float> h_x2(count);
  Rng rng2(seed ^ 0xa5a5a5a5u);
  for (size_t i = 0; i < count; ++i) h_x2[i] = rng2.next_value();
  CUDA_CHECK(cudaMemcpy(d_x, h_x2.data(), bytes, cudaMemcpyHostToDevice));
  launch_candidate(d_x, d_w, d_y, rows, cols, eps, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_y2(count);
  CUDA_CHECK(cudaMemcpy(h_y2.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  bool input_sensitive = checksum(h_y2) != first_checksum;

  // Restore the original input so timing measures the same work we validated.
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  // Timing. Every repeat is measured individually so the caller sees the spread
  // rather than a single sample; run-to-run variance on this class of hardware
  // is several percent and a lone number is not evidence.
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(repeats));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < repeats; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_candidate(d_x, d_w, d_y, rows, cols, eps, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<double> sorted = samples;
  for (size_t i = 1; i < sorted.size(); ++i) {
    double key = sorted[i];
    size_t j = i;
    while (j > 0 && sorted[j - 1] > key) { sorted[j] = sorted[j - 1]; --j; }
    sorted[j] = key;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_y));

  ShapeResult r{};
  r.rows = rows;
  r.cols = cols;
  r.min_ms = sorted.front();
  r.median_ms = sorted[sorted.size() / 2];
  r.max_ms = sorted.back();
  r.max_abs_err = dev.max_abs;
  r.max_rel_err = dev.max_rel;
  r.has_nonfinite = dev.has_nonfinite;
  r.wrote_output = wrote_output;
  r.input_sensitive = input_sensitive;
  // Compulsory traffic: read x once, write y once. The weight vector is cached.
  r.bytes_moved = 2.0 * static_cast<double>(bytes);
  // Input + output are both live across the kernel; the weight vector is negligible.
  r.working_set_bytes = 2.0 * static_cast<double>(bytes);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  int repeats = argc > 1 ? std::atoi(argv[1]) : 30;
  unsigned int seed = argc > 2 ? static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10)) : 20260826u;
  if (repeats < 5) repeats = 5;

  // Several shapes, so specialising to one is visible as failure on the others.
  // Spans cache-resident and DRAM-bound working sets on purpose: specialising to
  // one shape shows up as failure on the others, and residency is what decides
  // whether the DRAM roofline applies at all.
  const int shapes[][2] = {{512, 2048}, {1024, 4096}, {4096, 4096}, {8192, 4096}, {16384, 4096}};
  const int shape_count = sizeof(shapes) / sizeof(shapes[0]);

  int mem_clock_khz = 0, bus_width_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  std::printf("{\n");
  std::printf("  \"device\": \"%s\",\n", props.name);
  std::printf("  \"compute_capability\": \"%d.%d\",\n", props.major, props.minor);
  std::printf("  \"peak_bandwidth_bytes_per_s\": %.1f,\n",
              2.0 * static_cast<double>(mem_clock_khz) * 1e3 * (bus_width_bits / 8.0));
  // Needed to tell DRAM-served traffic from cache-resident traffic. A working
  // set that fits in L2 never crosses the memory bus, so the DRAM roofline does
  // not bound it and a correct kernel can legitimately appear to exceed peak.
  std::printf("  \"l2_cache_bytes\": %d,\n", props.l2CacheSize);
  std::printf("  \"repeats\": %d,\n", repeats);
  std::printf("  \"seed\": %u,\n", seed);
  std::printf("  \"shapes\": [\n");
  for (int i = 0; i < shape_count; ++i) {
    ShapeResult r = measure(shapes[i][0], shapes[i][1], repeats, seed + static_cast<unsigned int>(i));
    std::printf("    {\"rows\": %d, \"cols\": %d, \"min_ms\": %.6f, \"median_ms\": %.6f, \"max_ms\": %.6f, "
                "\"max_abs_err\": %.6g, \"max_rel_err\": %.6g, \"has_nonfinite\": %s, "
                "\"wrote_output\": %s, \"input_sensitive\": %s, \"bytes_moved\": %.1f, "
                "\"working_set_bytes\": %.1f}%s\n",
                r.rows, r.cols, r.min_ms, r.median_ms, r.max_ms, r.max_abs_err, r.max_rel_err,
                r.has_nonfinite ? "true" : "false", r.wrote_output ? "true" : "false",
                r.input_sensitive ? "true" : "false", r.bytes_moved, r.working_set_bytes,
                i + 1 == shape_count ? "" : ",");
  }
  std::printf("  ]\n}\n");
  return 0;
}
