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
#include <limits>
#include <vector>
#include <chrono>
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

using ReferenceFn = void (*)(const std::vector<float>&, const std::vector<float>&, std::vector<double>&, int, int,
                             double);

// Row-wise softmax with the standard max subtraction. Numerically the most
// delicate of the set: a candidate that skips the max shift still looks correct
// on small values and blows up on large ones, which is why the input
// distribution spans several orders of magnitude.
void reference_softmax(const std::vector<float>& x, const std::vector<float>&, std::vector<double>& y, int rows,
                       int cols, double) {
  for (int r = 0; r < rows; ++r) {
    const float* row = x.data() + static_cast<size_t>(r) * cols;
    double row_max = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < cols; ++c) row_max = row[c] > row_max ? row[c] : row_max;
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp(static_cast<double>(row[c]) - row_max);
    for (int c = 0; c < cols; ++c) {
      y[static_cast<size_t>(r) * cols + c] = std::exp(static_cast<double>(row[c]) - row_max) / sum;
    }
  }
}

// SiLU: x * sigmoid(x). Purely elementwise -- no reduction, no cross-thread
// communication. Included because every other op in the set has a row reduction,
// and a gate suite that only ever sees reductions is not known to generalise.
void reference_silu(const std::vector<float>& x, const std::vector<float>&, std::vector<double>& y, int rows,
                    int cols, double) {
  const size_t n = static_cast<size_t>(rows) * cols;
  for (size_t i = 0; i < n; ++i) {
    double v = x[i];
    y[i] = v / (1.0 + std::exp(-v));
  }
}

// Transpose. No arithmetic at all, so it is the cleanest test of the roofline
// gate: any time above the compulsory traffic is pure memory inefficiency. A
// naive implementation writes uncoalesced and lands far below peak, which makes
// it the one op in the set where the gate has real headroom to discriminate.
void reference_transpose(const std::vector<float>& x, const std::vector<float>&, std::vector<double>& y, int rows,
                         int cols, double) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      y[static_cast<size_t>(c) * rows + r] = x[static_cast<size_t>(r) * cols + c];
    }
  }
}

using FlopFn = double (*)(int rows, int cols);
using TolFn = double (*)(int rows, int cols);

constexpr double FP32_EPS = 1.1920929e-07;

// Relative tolerance for an fp32 result accumulated over `depth` terms. One
// global tolerance is wrong: ops differ in conditioning, and summing `depth`
// products in fp32 accumulates roughly sqrt(depth) * eps under random signs, so a
// deep reduction cannot be held to the same bar as an elementwise map.
double accumulation_tolerance(int depth, double safety = 8.0) {
  return safety * std::sqrt(static_cast<double>(depth < 1 ? 1 : depth)) * FP32_EPS;
}

double tol_reduction(int, int cols) { return accumulation_tolerance(cols); }
double tol_elementwise(int, int) { return accumulation_tolerance(1); }
// Pure data movement is bit-exact; no accumulation allowance is warranted.
double tol_exact(int, int) { return 0.0; }

// Compulsory arithmetic per invocation. Only the order of magnitude matters: it
// decides which ceiling binds, and a factor-of-two error in the FLOP count moves
// an op across the ridge only if it was already sitting on it.
double flops_per_element(int rows, int cols, double per_element) {
  return static_cast<double>(rows) * cols * per_element;
}
double flops_rmsnorm(int rows, int cols) { return flops_per_element(rows, cols, 4.0); }
double flops_softmax(int rows, int cols) { return flops_per_element(rows, cols, 5.0); }
double flops_silu(int rows, int cols) { return flops_per_element(rows, cols, 4.0); }
double flops_transpose(int, int) { return 0.0; }

struct OpSpec {
  const char* name;
  ReferenceFn reference;
  FlopFn flops;
  TolFn tolerance;
  bool uses_weight;
};

constexpr OpSpec OPS[] = {
    {"rmsnorm", &reference_rmsnorm, &flops_rmsnorm, &tol_reduction, true},
    {"softmax", &reference_softmax, &flops_softmax, &tol_reduction, false},
    {"silu", &reference_silu, &flops_silu, &tol_elementwise, false},
    {"transpose", &reference_transpose, &flops_transpose, &tol_exact, false},
};

int fp32_lanes_per_sm(int major, int minor) {
  switch (major * 10 + minor) {
    case 70: case 72: case 75: return 64;   // Volta, Turing
    case 80: return 64;                     // GA100 (A100)
    case 86: case 87: case 89: return 128;  // GA10x, Orin, Ada
    case 90: return 128;                    // Hopper
    case 100: case 120: return 128;         // Blackwell
    default: return 0;                      // unknown: refuse to guess
  }
}

const OpSpec* find_op(const char* name) {
  for (const OpSpec& op : OPS) {
    if (std::strcmp(op.name, name) == 0) return &op;
  }
  return nullptr;
}

struct Deviation {
  double max_abs;
  double max_rel;
  // |got - want| / (rel_tol * (rms|want| + |want|)). 1.0 sits exactly on the
  // tolerance. Pure relative error is meaningless where the reference can be
  // near zero: cancellation in a dot product yields occasional near-zero results
  // and the ratio explodes even for a correct kernel. Measured on this host,
  // pure relative error fails torch's own matmul at 2.8e-2.
  double violation;
  bool has_nonfinite;
};

Deviation compare(const std::vector<float>& got, const std::vector<double>& want, double rel_tol) {
  Deviation d{0.0, 0.0, 0.0, false};

  double sum_sq = 0.0;
  for (double w : want) sum_sq += w * w;
  const double scale = want.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(want.size()));

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
    if (rel_tol <= 0.0) {
      // Bit-exact op: any difference at all is a violation.
      if (diff > 0.0) d.violation = std::numeric_limits<double>::infinity();
    } else {
      double v = diff / (rel_tol * (scale + denom));
      d.violation = v > d.violation ? v : d.violation;
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
  double p90_ms;
  double max_ms;
  double max_abs_err;
  double max_rel_err;
  double violation;
  bool has_nonfinite;
  bool wrote_output;      // output changed from the poison value
  bool input_sensitive;   // output changed when the input changed
  int inner_iters;            // launches batched into one timed sample
  bool timed_output_written;  // the measured calls wrote a result too
  double timed_max_rel_err;   // error of the last measured call
  double timed_violation;
  double rel_tol;
  double bytes_moved;
  double flops;
  double working_set_bytes;
};

ShapeResult measure(const OpSpec& op, int rows, int cols, int repeats, unsigned int seed) {
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

  op.reference(h_x, h_w, h_ref, rows, cols, eps);

  // Warmup, outside the timed region.
  for (int i = 0; i < 10; ++i) launch_candidate(d_x, d_w, d_y, rows, cols, eps, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> poison(count);
  std::memset(poison.data(), poison_byte, bytes);
  bool wrote_output = std::memcmp(h_y.data(), poison.data(), bytes) != 0;
  Deviation dev = compare(h_y, h_ref, op.tolerance(rows, cols));
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
  // Wall clock around a device-wide sync, not events on stream 0. A candidate can
  // enqueue its work on its own non-blocking stream, which events recorded on
  // stream 0 would not bracket -- the work would land after the stop event and be
  // omitted from the elapsed time entirely.
  CUDA_CHECK(cudaMemset(d_y, poison_byte, bytes));

  // Batch launches per sample so each sample spans enough work to measure. The
  // smallest shape here runs in single-digit microseconds, where one scheduler
  // hiccup dominates the sample and any percentile becomes noise rather than a
  // measurement. A pilot launch sets the batch size; the reported time is per
  // launch.
  double pilot_ms;
  {
    auto p0 = std::chrono::steady_clock::now();
    launch_candidate(d_x, d_w, d_y, rows, cols, eps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    pilot_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
  }
  const double target_sample_ms = 2.0;
  int inner = 1;
  if (pilot_ms > 0.0 && pilot_ms < target_sample_ms) {
    double wanted = target_sample_ms / pilot_ms;
    inner = wanted > 1000.0 ? 1000 : static_cast<int>(wanted) + 1;
  }

  for (int i = 0; i < repeats; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    for (int j = 0; j < inner; ++j) launch_candidate(d_x, d_w, d_y, rows, cols, eps, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / inner);
  }
  CUDA_CHECK(cudaGetLastError());

  // Re-validate what the *last timed* call produced. Correctness was checked
  // before timing; without this a candidate can count invocations, compute
  // honestly through warmup and the sensitivity probe, then skip the work during
  // every measured call and keep its passing correctness fields.
  std::vector<float> h_y_timed(count);
  CUDA_CHECK(cudaMemcpy(h_y_timed.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  bool timed_output_written = std::memcmp(h_y_timed.data(), poison.data(), bytes) != 0;
  Deviation timed_dev = compare(h_y_timed, h_ref, op.tolerance(rows, cols));

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
  {
    size_t idx = (sorted.size() * 9) / 10;
    // Never the max: with few repeats that index lands on the last element
    // and one outlier would masquerade as the 90th percentile.
    size_t cap = sorted.size() >= 3 ? sorted.size() - 2 : 0;
    r.p90_ms = sorted[idx < cap ? idx : cap];
  }
  r.max_ms = sorted.back();
  r.max_abs_err = dev.max_abs;
  r.max_rel_err = dev.max_rel;
  r.violation = dev.violation;
  r.has_nonfinite = dev.has_nonfinite;
  r.wrote_output = wrote_output;
  r.input_sensitive = input_sensitive;
  r.inner_iters = inner;
  r.timed_output_written = timed_output_written;
  r.timed_max_rel_err = timed_dev.max_rel;
  r.timed_violation = timed_dev.violation;
  // Compulsory traffic: read x once, write y once. The weight vector is cached.
  r.bytes_moved = 2.0 * static_cast<double>(bytes);
  r.flops = op.flops(rows, cols);
  r.rel_tol = op.tolerance(rows, cols);
  // Input + output are both live across the kernel; the weight vector is negligible.
  r.working_set_bytes = 2.0 * static_cast<double>(bytes);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  auto harness_start = std::chrono::steady_clock::now();
  const char* op_name = argc > 1 ? argv[1] : "rmsnorm";
  const OpSpec* op = find_op(op_name);
  if (op == nullptr) {
    std::printf("{\"error\":\"unknown op: %s\"}\n", op_name);
    return 2;
  }
  int repeats = argc > 2 ? std::atoi(argv[2]) : 30;
  unsigned int seed = argc > 3 ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 20260826u;
  // Echoed back so the caller can tell this output came from a run it started,
  // rather than from anything the candidate printed on its way past.
  const char* nonce = argc > 4 ? argv[4] : "";
  if (repeats < 5) repeats = 5;

  // Several shapes, so specialising to one is visible as failure on the others.
  // Spans cache-resident and DRAM-bound working sets on purpose: specialising to
  // one shape shows up as failure on the others, and residency is what decides
  // whether the DRAM roofline applies at all.
  const int shapes[][2] = {{512, 2048}, {1024, 4096}, {4096, 4096}, {8192, 4096}, {16384, 4096}};
  const int shape_count = sizeof(shapes) / sizeof(shapes[0]);

  int mem_clock_khz = 0, bus_width_bits = 0, sm_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  std::printf("{\n");
  std::printf("  \"nonce\": \"%s\",\n", nonce);
  std::printf("  \"op\": \"%s\",\n", op->name);
  std::printf("  \"op\": \"%s\",\n", op->name);
  std::printf("  \"device\": \"%s\",\n", props.name);
  std::printf("  \"compute_capability\": \"%d.%d\",\n", props.major, props.minor);
  std::printf("  \"peak_bandwidth_bytes_per_s\": %.1f,\n",
              2.0 * static_cast<double>(mem_clock_khz) * 1e3 * (bus_width_bits / 8.0));
  // Needed to tell DRAM-served traffic from cache-resident traffic. A working
  // set that fits in L2 never crosses the memory bus, so the DRAM roofline does
  // not bound it and a correct kernel can legitimately appear to exceed peak.
  std::printf("  \"l2_cache_bytes\": %d,\n", props.l2CacheSize);
  // Zero when the lane count for this architecture is unknown. The auditor must
  // treat that as "no compute ceiling available", not as "zero FLOPs".
  {
    int lanes = fp32_lanes_per_sm(props.major, props.minor);
    double peak_flops = lanes == 0 ? 0.0
                                   : static_cast<double>(props.multiProcessorCount) * lanes * 2.0 *
                                         (static_cast<double>(sm_clock_khz) * 1e3);
    std::printf("  \"peak_fp32_flops\": %.1f,\n", peak_flops);
    std::printf("  \"sm_count\": %d,\n", props.multiProcessorCount);
  }
  std::printf("  \"repeats\": %d,\n", repeats);
  std::printf("  \"seed\": %u,\n", seed);
  std::printf("  \"shapes\": [\n");
  for (int i = 0; i < shape_count; ++i) {
    ShapeResult r = measure(*op, shapes[i][0], shapes[i][1], repeats, seed + static_cast<unsigned int>(i));
    std::printf("    {\"rows\": %d, \"cols\": %d, \"min_ms\": %.6f, \"median_ms\": %.6f, \"p90_ms\": %.6f, \"max_ms\": %.6f, "
                "\"max_abs_err\": %.6g, \"max_rel_err\": %.6g, \"violation\": %.6g, \"has_nonfinite\": %s, "
                "\"wrote_output\": %s, \"input_sensitive\": %s, "
                "\"inner_iters\": %d, \"timed_output_written\": %s, \"timed_max_rel_err\": %.6g, \"timed_violation\": %.6g, "
                "\"rel_tol\": %.6g, \"bytes_moved\": %.1f, \"flops\": %.1f, \"working_set_bytes\": %.1f}%s\n",
                r.rows, r.cols, r.min_ms, r.median_ms, r.p90_ms, r.max_ms, r.max_abs_err, r.max_rel_err, r.violation,
                r.has_nonfinite ? "true" : "false", r.wrote_output ? "true" : "false",
                r.input_sensitive ? "true" : "false",
                r.inner_iters, r.timed_output_written ? "true" : "false", r.timed_max_rel_err, r.timed_violation,
                r.rel_tol, r.bytes_moved, r.flops, r.working_set_bytes,
                i + 1 == shape_count ? "" : ",");
  }
  std::printf("  ],\n");
  // Self-reported wall time. The caller compares this against the elapsed
  // time it measured for the whole process; a forged result printed before
  // any work happened cannot account for time it never spent.
  std::printf("  \"harness_wall_ms\": %.3f\n}\n",
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - harness_start).count());
  return 0;
}
