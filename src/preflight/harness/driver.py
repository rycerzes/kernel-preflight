"""Fixed measurement harness for the Python backends (Triton, Helion).

Deliberately emits the *same* measurement schema as `driver.cu`, so every gate in
`preflight.gates` works across backends without knowing which one produced the
numbers. The gates adjudicate a schema, not a toolchain.

Every guarantee the CUDA harness makes is reproduced here, because a candidate in
Triton can cheat in exactly the same ways as one in CUDA:

  inputs        seeded here, spanning several orders of magnitude and both signs
  reference     computed here in float64, never by the candidate
  tolerances    set here
  timing        wall clock around a device-wide sync, so work queued on another
                stream is still bracketed
  liveness      output poisoned before the run; an untouched buffer is visible
  sensitivity   the input is changed and the output must follow
  timed work    the output of the *last measured* call is revalidated
  provenance    a nonce chosen by the caller is echoed, and the harness reports
                its own wall time for the caller to check against

The candidate supplies one callable and nothing else:

    launch_candidate(inputs: dict[str, Tensor], out: Tensor, meta: dict) -> None

writing its result into `out` in place. It never sees the reference, the
tolerances, or the clock.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

# Storage dtype per declared precision contract. TF32 is a *compute* mode over
# fp32 data, not a storage format, so it stores fp32 and differs only in which
# hardware ceiling binds. bf16 and fp16 change the tensors themselves, which also
# halves the compulsory byte traffic.
STORAGE_DTYPE = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# Mantissa bits, used to widen the correctness tolerance for a declared reduced
# precision. bf16 keeps 7 against fp32's 23, so its results carry roughly 2^16
# times more relative error and cannot be held to an fp32 bar.
PRECISION_MANTISSA = {"fp32": 23, "tf32": 10, "bf16": 7, "fp16": 10}


def storage_dtype(precision: str) -> torch.dtype:
    if precision not in STORAGE_DTYPE:
        raise SystemExit(f"unknown precision {precision!r}")
    return STORAGE_DTYPE[precision]


def precision_tolerance_scale(precision: str) -> float:
    """How much looser the tolerance must be than the fp32 baseline."""
    return float(2 ** (23 - PRECISION_MANTISSA.get(precision, 23)))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_input(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
    *,
    exponent_range: int = 6,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Values spanning several orders of magnitude, both signs.

    Not a narrow uniform: kernels have been observed exploiting inputs drawn from
    a tight range, and a softmax that skips its max subtraction looks correct on
    small values and overflows on large ones.

    `exponent_range` bounds that spread, and some ops must bound it. Attention
    computes a softmax over QK^T, so with `exponent_range=6` the scores span
    roughly 2^24 and fp32 and fp64 disagree in the saturated tail by far more than
    any sane tolerance -- the harness would fail correct kernels. Attention
    therefore uses a realistic Q/K/V scale, and its protection against
    narrow-range exploits comes from the score distribution rather than from the
    input magnitudes.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    unit = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
    sign = torch.where(
        torch.rand(shape, generator=gen, device=device) < 0.5,
        torch.tensor(-1.0, device=device),
        torch.tensor(1.0, device=device),
    )
    if exponent_range <= 0:
        return (sign * (0.25 + unit)).contiguous().to(dtype)
    # bf16 has 8 exponent bits but only 7 of mantissa; a wide exponent spread costs
    # it nothing in range and everything in precision, so narrow the spread for the
    # low-mantissa formats rather than manufacture a correctness failure.
    span = exponent_range if dtype is torch.float32 else min(exponent_range, 2)
    exponent = torch.randint(
        -span, span + 1, shape, generator=gen, device=device, dtype=torch.float32
    )
    return (
        sign * (0.25 + unit) * torch.pow(torch.tensor(2.0, device=device), exponent)
    ).contiguous().to(dtype)


# ---------------------------------------------------------------------------
# Operations: reference, traffic model, tensor construction
# ---------------------------------------------------------------------------


FP32_EPS = 1.1920929e-07


MANTISSA_BITS_BY_DTYPE = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}


def quantisation_error(dt: torch.dtype) -> float:
    """Relative error from merely storing a value in `dt`."""
    return float(2 ** -(MANTISSA_BITS_BY_DTYPE.get(dt, 23) + 1))


def accumulation_tolerance(depth: int, safety: float = 8.0, dt: torch.dtype = torch.float32) -> float:
    """Relative tolerance for a result accumulated over `depth` terms in `dt`.

    Two independent error sources, and they must not be multiplied together:

    * **Accumulation.** Summing `depth` products under random signs drifts by about
      sqrt(depth) * eps. Tensor cores accumulate in fp32 regardless of input dtype,
      so this term uses fp32 epsilon even for a bf16 kernel.
    * **Quantisation.** Merely storing an input in bf16 costs 2^-8 of relative
      precision before any arithmetic happens.

    An earlier version multiplied the accumulation term by the full mantissa ratio
    (2^16 for bf16), which produced a *relative tolerance of 8* — 800%, a gate that
    admits anything. Taking the larger of the two terms is the correct model,
    because one dominates: for fp32 the accumulation drift does, for bf16 the input
    quantisation does by three orders of magnitude.
    """
    accumulation = math.sqrt(max(1, depth)) * FP32_EPS
    return safety * max(accumulation, quantisation_error(dt))


@dataclass(frozen=True)
class Problem:
    """One concrete instance of an op: its tensors, cost model and reference."""

    label: str
    inputs: dict[str, torch.Tensor]
    out: torch.Tensor
    meta: dict[str, Any]
    reference: torch.Tensor  # float64, computed by the harness
    bytes_moved: float
    flops: float
    working_set_bytes: float
    rel_tol: float


def _rmsnorm(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev)
    w = make_input((cols,), seed ^ 0x5EED, dev)
    eps = 1e-6
    xd, wd = x.double(), w.double()
    ref = xd * torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + eps) * wd
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x, "w": w},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols, "eps": eps},
        reference=ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=4.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(cols, dt=dt),
    )


def _softmax(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev)
    ref = torch.softmax(x.double(), dim=-1)
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=5.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(cols, dt=dt),
    )


def _silu(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev)
    xd = x.double()
    ref = xd * torch.sigmoid(xd)
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=4.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(1, dt=dt),
    )


def _transpose(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev)
    ref = x.double().t().contiguous()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty((cols, rows), device=dev, dtype=dt),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=0.0,
        # Pure data movement: bit-exact, so no accumulation allowance.
        rel_tol=0.0,
        working_set_bytes=2.0 * n * dt.itemsize,
    )


def _matmul(m: int, k: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    """Square-ish GEMM. The first op in the set that is compute-bound.

    Arithmetic intensity is ~n/6, so at these sizes it sits far above the RTX
    4090's ~82 FLOP/byte ridge point and the FP32 pipelines bind rather than the
    memory bus. That makes it the first real exercise of the roofline gate's
    compute branch.
    """
    n = k
    # Bounded magnitudes: a 4096-deep dot product over values spanning 2^12 loses
    # more precision to cancellation than any tolerance should forgive.
    a = make_input((m, k), seed, dev, exponent_range=0)
    b = make_input((k, n), seed ^ 0xB00, dev, exponent_range=0)
    ref = a.double() @ b.double()
    return Problem(
        label=f"{m}x{k}x{n}",
        inputs={"a": a, "b": b},
        out=torch.empty((m, n), device=dev, dtype=dt),
        meta={"m": m, "k": k, "n": n},
        reference=ref,
        bytes_moved=float((m * k + k * n + m * n) * dt.itemsize),
        flops=2.0 * m * k * n,
        working_set_bytes=float((m * k + k * n + m * n) * dt.itemsize),
        rel_tol=accumulation_tolerance(k, dt=dt),
    )


def _attention(seq: int, head_dim: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    """Single-head-per-batch scaled dot product attention, non-causal.

    Compute-bound like matmul, and included because it is the operation the
    kernel-authoring ecosystem actually cares about. The reference is torch's own
    SDPA in float64, so the harness is not reimplementing attention and getting it
    subtly wrong.
    """
    batch, heads = 4, 8
    shape = (batch, heads, seq, head_dim)
    # Bounded magnitudes: see make_input. Wide-exponent Q/K would make the
    # softmax saturate differently in fp32 and fp64 and fail correct kernels.
    q = make_input(shape, seed, dev, exponent_range=0, dtype=dt)
    k = make_input(shape, seed ^ 0xA11, dev, exponent_range=0, dtype=dt)
    v = make_input(shape, seed ^ 0xB22, dev, exponent_range=0, dtype=dt)
    ref = torch.nn.functional.scaled_dot_product_attention(q.double(), k.double(), v.double())
    elements = batch * heads * seq * head_dim
    # QK^T and PV are each 2*b*h*s*s*d; the softmax is negligible beside them.
    flops = 4.0 * batch * heads * seq * seq * head_dim
    return Problem(
        label=f"b{batch}h{heads}s{seq}d{head_dim}",
        inputs={"q": q, "k": k, "v": v},
        out=torch.empty(shape, device=dev, dtype=dt),
        meta={"batch": batch, "heads": heads, "seq": seq, "head_dim": head_dim},
        reference=ref,
        bytes_moved=float(4 * elements * dt.itemsize),  # q, k, v read; o written
        flops=flops,
        working_set_bytes=float(4 * elements * dt.itemsize),
        # Two chained accumulations of depth seq: QK^T then PV. safety=8 rather
        # than the 32 an earlier version used -- that was compensating for a
        # tolerance model which double-counted precision, and with the model fixed
        # the measured deviation sits at roughly a quarter of this bound instead of
        # a sixteenth. A gate with 16x of headroom is barely a gate.
        rel_tol=accumulation_tolerance(seq, safety=8.0, dt=dt),
    )


# Shapes per op. Memory-bound ops sweep across the L2 boundary on purpose;
# compute-bound ops stay small enough that a float64 reference is affordable.
OPS: dict[str, tuple[Callable[..., Problem], tuple[tuple[int, int], ...]]] = {
    "rmsnorm": (_rmsnorm, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
    "softmax": (_softmax, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
    "silu": (_silu, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
    "transpose": (_transpose, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
    "matmul": (_matmul, ((512, 512), (1024, 1024), (2048, 2048), (4096, 4096))),
    "attention": (_attention, ((512, 64), (1024, 64), (2048, 64), (1024, 128))),
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

POISON = float("nan")
MAX_REL_ERR_REPORTED = 0.0


def deviation(got: torch.Tensor, want: torch.Tensor, rel_tol: float) -> tuple[float, float, float, bool]:
    """Compare against a mixed absolute/relative criterion.

    Pure relative error is meaningless where the reference can be near zero. A
    dot product of random values produces occasional near-zero results by
    cancellation, and |1e-3 - 1e-5| / 1e-5 explodes even though the answer is
    correct to four significant figures. Measured on this host: torch's *own*
    matmul scores 2.8e-2 by pure relative error, and SDPA 3.6e-2 with TF32
    disabled -- so that metric fails the reference implementations.

    The criterion is the one `allclose` uses, with the absolute floor scaled to
    the magnitude of the output rather than picked:

        |got - want| <= rel_tol * (rms(|want|) + |want|)

    `violation` is that inequality divided through, so 1.0 sits exactly on the
    tolerance and the gate compares against a single dimensionless number.
    """
    g = got.double()
    finite = torch.isfinite(g)
    has_nonfinite = bool((~finite).any().item())
    if not bool(finite.any().item()):
        return float("inf"), float("inf"), float("inf"), has_nonfinite

    diff = (g - want).abs()
    denom = want.abs()
    max_abs = float(diff[finite].max().item())

    rel_mask = finite & (denom > 1e-6)
    max_rel = float((diff[rel_mask] / denom[rel_mask]).max().item()) if bool(rel_mask.any().item()) else 0.0

    if rel_tol <= 0.0:
        # Bit-exact ops (pure data movement) get no allowance at all.
        violation = 0.0 if max_abs == 0.0 else float("inf")
        return max_abs, max_rel, violation, has_nonfinite

    scale = float(want.abs().pow(2).mean().sqrt().item())
    allowed = rel_tol * (scale + denom)
    violation = float((diff[finite] / allowed[finite]).max().item())
    return max_abs, max_rel, violation, has_nonfinite


def current_sm_clock_hz() -> float | None:
    """Actual SM clock right now, or None if it cannot be read.

    Matters because the roofline ceiling is derived from the *maximum* clock. A
    GPU that is thermally or power throttled never had access to that peak, so a
    perfectly good kernel measures low and a reader concludes the kernel is bad.
    Observed here: a sweep run immediately after 900 seconds of sustained
    autotuning measured roughly a third of the throughput the same kernel reached
    on an idle device.
    """
    try:
        return float(torch.cuda.clock_rate()) * 1e6  # torch reports MHz
    except Exception:
        return None


def measure_problem(problem: Problem, launch: Callable, repeats: int) -> dict[str, Any]:
    out = problem.out

    # Poison, so a kernel that never writes is visible rather than fast.
    out.fill_(POISON)

    for _ in range(10):  # warmup, outside the timed region
        launch(problem.inputs, out, problem.meta)
    torch.cuda.synchronize()

    wrote_output = bool(torch.isfinite(out).any().item())
    max_abs, max_rel, violation, has_nonfinite = deviation(out, problem.reference, problem.rel_tol)
    first = float(out.double()[torch.isfinite(out.double())].sum().item()) if wrote_output else 0.0

    # Change an input; the output must follow. Catches a kernel that caches,
    # hardcodes, or ignores its arguments.
    key = next(iter(problem.inputs))
    original = problem.inputs[key].clone()
    problem.inputs[key].copy_(
        make_input(tuple(original.shape), 0xA5A5A5, original.device, exponent_range=0)
    )
    launch(problem.inputs, out, problem.meta)
    torch.cuda.synchronize()
    second = float(out.double()[torch.isfinite(out.double())].sum().item())
    input_sensitive = not math.isclose(first, second, rel_tol=1e-9, abs_tol=0.0)
    problem.inputs[key].copy_(original)

    # Timing: wall clock around a device-wide sync, so work a candidate queues on
    # its own stream is still bracketed.
    out.fill_(POISON)

    # Batch launches per sample so each sample spans enough work to measure. The
    # smallest shapes run in single-digit microseconds, where one scheduler
    # hiccup dominates and any percentile becomes noise. A pilot launch sets the
    # batch size; the reported time is per launch.
    t0 = time.perf_counter()
    launch(problem.inputs, out, problem.meta)
    torch.cuda.synchronize()
    pilot_ms = (time.perf_counter() - t0) * 1000.0
    target_sample_ms = 2.0
    inner = 1
    if 0.0 < pilot_ms < target_sample_ms:
        inner = min(1000, int(target_sample_ms / pilot_ms) + 1)

    samples: list[float] = []
    clocks: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            launch(problem.inputs, out, problem.meta)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0 / inner)
        clock = current_sm_clock_hz()
        if clock:
            clocks.append(clock)

    # Revalidate what the last *measured* call produced.
    timed_written = bool(torch.isfinite(out).any().item())
    _, timed_rel, timed_violation, _ = deviation(out, problem.reference, problem.rel_tol)

    ordered = sorted(samples)
    return {
        "label": problem.label,
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        # Never the max: with few repeats that index lands on the last element and
        # one cold sample would masquerade as the 90th percentile.
        "p90_ms": ordered[min((len(ordered) * 9) // 10, max(0, len(ordered) - 2))],
        "p25_ms": ordered[len(ordered) // 4],
        "p75_ms": ordered[(len(ordered) * 3) // 4],
        # Samples above twice the median. Reported, not acted on: a spike under
        # load is worth seeing without failing a kernel over it.
        "outliers": sum(1 for v in ordered if v > 2.0 * statistics.median(ordered)),
        "max_ms": ordered[-1],
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "violation": violation,
        "has_nonfinite": has_nonfinite,
        "wrote_output": wrote_output,
        "input_sensitive": input_sensitive,
        "inner_iters": inner,
        "sm_clock_hz_observed": (statistics.median(clocks) if clocks else None),
        "timed_output_written": timed_written,
        "timed_max_rel_err": timed_rel,
        "timed_violation": timed_violation,
        "rel_tol": problem.rel_tol,
        "bytes_moved": problem.bytes_moved,
        "flops": problem.flops,
        "working_set_bytes": problem.working_set_bytes,
    }


FP32_LANES_PER_SM = {
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128, (10, 0): 128, (12, 0): 128,
}


def device_ceilings(dev_index: int = 0) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(dev_index)
    cap = (props.major, props.minor)
    lanes = FP32_LANES_PER_SM.get(cap)
    # torch exposes neither memory clock nor bus width, so read them the same way
    # the CUDA harness does.
    import ctypes

    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit(0)
    dev = ctypes.c_int()
    lib.cuDeviceGet(ctypes.byref(dev), dev_index)

    def attr(which: int) -> int:
        out = ctypes.c_int()
        lib.cuDeviceGetAttribute(ctypes.byref(out), which, dev)
        return out.value

    mem_clock_khz = attr(36)
    bus_bits = attr(37)
    sm_clock_khz = attr(13)
    peak_bw = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0)
    peak_flops = 0.0 if lanes is None else props.multi_processor_count * lanes * 2.0 * sm_clock_khz * 1e3
    return {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "peak_bandwidth_bytes_per_s": peak_bw,
        "peak_fp32_flops": peak_flops,
        "l2_cache_bytes": props.L2_cache_size,
        "sm_count": props.multi_processor_count,
        # Reported so the auditor can compute a ceiling for any precision class,
        # not just the fp32 one this function happens to derive.
        "sm_clock_hz": float(sm_clock_khz) * 1e3,
    }


def load_candidate(path: str) -> Callable:
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    launch = getattr(module, "launch_candidate", None)
    if launch is None:
        raise RuntimeError("candidate does not define launch_candidate(inputs, out, meta)")
    return launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=sorted(OPS))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--nonce", default="")
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the measurement JSON. Not stdout: stdout is a shared "
             "channel that any imported library can log into, and TileLang does. It is "
             "also the channel a candidate could print a forged measurement on.",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help="Numerical contract the candidate claims to honour: fp32, tf32, bf16 or fp16. "
             "Sets both the correctness tolerance and which hardware ceiling binds.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if not torch.cuda.is_available():
        with open(args.out, "w") as handle:
            json.dump({"error": "CUDA unavailable"}, handle)
        return 3

    builder, shapes = OPS[args.op]
    dev = torch.device("cuda:0")
    dt = storage_dtype(args.precision)
    launch = load_candidate(args.candidate)

    payload: dict[str, Any] = {"nonce": args.nonce, "op": args.op, "precision": args.precision}
    payload.update(device_ceilings())
    payload["repeats"] = max(5, args.repeats)
    payload["seed"] = args.seed

    shapes_out = []
    for index, dims in enumerate(shapes):
        problem = builder(dims[0], dims[1], args.seed + index, dev, dt)
        record = measure_problem(problem, launch, payload["repeats"])
        # The schema names these rows/cols for compatibility with the CUDA
        # harness; they are labels, and ops with other layouts report their own.
        record["rows"], record["cols"] = dims
        shapes_out.append(record)
        del problem
        torch.cuda.empty_cache()

    payload["shapes"] = shapes_out
    payload["harness_wall_ms"] = (time.perf_counter() - started) * 1000.0
    with open(args.out, "w") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
