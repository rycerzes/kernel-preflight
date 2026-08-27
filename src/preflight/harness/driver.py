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
import ctypes
import json
import math
import os
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

def storage_dtype(precision: str) -> torch.dtype:
    if precision not in STORAGE_DTYPE:
        raise SystemExit(f"unknown precision {precision!r}")
    return STORAGE_DTYPE[precision]


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
    # Recomputes `reference` from whatever the input tensors currently hold. The
    # timing loop perturbs an input before every sample, so the answer changes
    # constantly and a candidate cannot serve one it computed earlier.
    recompute: Callable[[], torch.Tensor]
    bytes_moved: float
    flops: float
    working_set_bytes: float
    rel_tol: float


def _rmsnorm(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev, dtype=dt)
    w = make_input((cols,), seed ^ 0x5EED, dev, dtype=dt)
    eps = 1e-6
    def _ref() -> torch.Tensor:
        xd, wd = x.double(), w.double()
        return xd * torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + eps) * wd

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x, "w": w},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols, "eps": eps},
        reference=ref,
        recompute=_ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=4.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(cols, dt=dt),
    )


def _softmax(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev, dtype=dt)

    def _ref() -> torch.Tensor:
        return torch.softmax(x.double(), dim=-1)

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        recompute=_ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=5.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(cols, dt=dt),
    )


def _silu(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev, dtype=dt)

    def _ref() -> torch.Tensor:
        xd = x.double()
        return xd * torch.sigmoid(xd)

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        recompute=_ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=4.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(1, dt=dt),
    )


def _transpose(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    x = make_input((rows, cols), seed, dev, dtype=dt)

    def _ref() -> torch.Tensor:
        return x.double().t().contiguous()

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty((cols, rows), device=dev, dtype=dt),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        recompute=_ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=0.0,
        # Pure data movement: bit-exact, so no accumulation allowance.
        rel_tol=0.0,
        working_set_bytes=2.0 * n * dt.itemsize,
    )


def _layernorm(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    """LayerNorm with affine gamma and beta.

    Distinct from rmsnorm rather than a variation on it: rmsnorm needs one reduction,
    LayerNorm needs the mean before the variance. A kernel can compute both in one
    pass with Welford or a sum/sum-of-squares pair, and getting the numerics of that
    wrong is the classic error in this operation -- E[x^2] - E[x]^2 loses catastrophic
    precision when the mean dominates the variance, which is exactly the case these
    inputs produce.

    KernelBenchX puts Normalization in the middle of its difficulty range and finds
    the failures come from scope: normalising over the wrong axis, or mixing
    statistics across rows.
    """
    x = make_input((rows, cols), seed, dev, dtype=dt)
    gamma = make_input((cols,), seed ^ 0x6A11, dev, dtype=dt)
    beta = make_input((cols,), seed ^ 0x7B22, dev, dtype=dt)
    eps = 1e-5

    def _ref() -> torch.Tensor:
        xd, gd, bd = x.double(), gamma.double(), beta.double()
        mean = xd.mean(dim=-1, keepdim=True)
        var = (xd - mean).pow(2).mean(dim=-1, keepdim=True)
        return (xd - mean) * torch.rsqrt(var + eps) * gd + bd

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x, "gamma": gamma, "beta": beta},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols, "eps": eps},
        reference=ref,
        recompute=_ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=8.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(cols, dt=dt),
    )


def _swiglu(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    """SwiGLU gating: silu(a) * b, the second half of a Llama-style MLP.

    Here because Fusion is both the largest category in KernelBenchX (60 of 176 tasks)
    and by far the worst performing: 72% of its tasks fail across every method they
    tested. The interest is not the arithmetic, which is trivial, but that a fused
    kernel has to hold an invariant across an operator boundary that the unfused
    version gets from materialising an intermediate.

    Three passes of traffic against five FLOP an element, so it is firmly
    memory-bound and the question is whether a candidate reads each input once.
    """
    a = make_input((rows, cols), seed, dev, dtype=dt)
    b = make_input((rows, cols), seed ^ 0x9C33, dev, dtype=dt)

    def _ref() -> torch.Tensor:
        ad, bd = a.double(), b.double()
        return ad * torch.sigmoid(ad) * bd

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"a": a, "b": b},
        out=torch.empty_like(a),
        meta={"rows": rows, "cols": cols},
        reference=ref,
        recompute=_ref,
        # a and b read, out written.
        bytes_moved=3.0 * n * dt.itemsize,
        flops=5.0 * n,
        working_set_bytes=3.0 * n * dt.itemsize,
        rel_tol=accumulation_tolerance(1, dt=dt),
    )


def _quantize(rows: int, cols: int, seed: int, dev: torch.device, dt: torch.dtype) -> Problem:
    """Per-row symmetric int8 quantise-dequantise with a power-of-two scale.

    Quantization is the category KernelBenchX reports as completely unsolved -- 0
    successes out of 30 across every method they evaluated -- because it is the one
    that cannot be done by transcribing a formula. The kernel has to derive its own
    scale from a reduction, round rather than truncate, clamp to the integer range, and
    scale back.

    **The scale is a power of two, and that is not a simplification.** The first
    version of this op used absmax/127, and it could not be graded: quantization is a
    discontinuous function, so a value within an ulp of a rounding boundary lands on a
    different integer in float32 than in the float64 reference, and the disagreement is
    a full step -- measured at 7.7% relative error on a correct kernel. Comparing a
    discontinuous operation against a higher-precision reference is ill-posed wherever
    the input sits near a discontinuity, and no tolerance fixes that: one wide enough to
    admit the boundary cases also admits truncation, which is the error the op exists to
    catch.

    A power-of-two scale removes the ambiguity instead of tolerating it. `absmax` is
    exact because a max reduction is exact; taking its binary exponent with `frexp` is
    exact; dividing by 2^k is an exponent shift and therefore exact; so `x / scale` is
    identical in float32 and float64 and `round` lands on the same integer in both. The
    whole round trip is bit-exact, the tolerance can stay at storage quantisation, and a
    kernel that truncates instead of rounding fails immediately rather than hiding
    inside an allowance.

    Scaling by a power of two is also what several real quantisation schemes do, for
    the same reason.
    """
    x = make_input((rows, cols), seed, dev, dtype=dt)
    qmax = 127.0

    def _ref() -> torch.Tensor:
        xd = x.double()
        absmax = xd.abs().amax(dim=-1, keepdim=True)
        # 2^e with absmax in [2^(e-1), 2^e), then /128 rather than /127 so the divisor
        # is itself a power of two and the division stays exact.
        _, exponent = torch.frexp(absmax)
        scale = torch.ldexp(torch.ones_like(absmax), exponent - 7)
        # An all-zero row has no scale; leave it rather than divide by zero.
        scale = torch.where(absmax > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(xd / scale), -qmax, qmax)
        return q * scale

    ref = _ref()
    n = rows * cols
    return Problem(
        label=f"{rows}x{cols}",
        inputs={"x": x},
        out=torch.empty_like(x),
        meta={"rows": rows, "cols": cols, "qmax": qmax},
        reference=ref,
        recompute=_ref,
        bytes_moved=2.0 * n * dt.itemsize,
        flops=4.0 * n,
        working_set_bytes=2.0 * n * dt.itemsize,
        # Deliberately not an accumulation tolerance: nothing accumulates, and with a
        # power-of-two scale nothing rounds either. The bar is storage quantisation
        # alone, which is as tight as this harness can ask for.
        rel_tol=8.0 * quantisation_error(dt),
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
    a = make_input((m, k), seed, dev, exponent_range=0, dtype=dt)
    b = make_input((k, n), seed ^ 0xB00, dev, exponent_range=0, dtype=dt)
    def _ref() -> torch.Tensor:
        return a.double() @ b.double()

    ref = _ref()
    return Problem(
        label=f"{m}x{k}x{n}",
        inputs={"a": a, "b": b},
        out=torch.empty((m, n), device=dev, dtype=dt),
        meta={"m": m, "k": k, "n": n},
        reference=ref,
        recompute=_ref,
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
    def _ref() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q.double(), k.double(), v.double())

    ref = _ref()
    elements = batch * heads * seq * head_dim
    # QK^T and PV are each 2*b*h*s*s*d; the softmax is negligible beside them.
    flops = 4.0 * batch * heads * seq * seq * head_dim
    return Problem(
        label=f"b{batch}h{heads}s{seq}d{head_dim}",
        inputs={"q": q, "k": k, "v": v},
        out=torch.empty(shape, device=dev, dtype=dt),
        meta={"batch": batch, "heads": heads, "seq": seq, "head_dim": head_dim},
        reference=ref,
        recompute=_ref,
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
    "layernorm": (_layernorm, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
    "swiglu": (_swiglu, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
    "quantize": (_quantize, ((512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096))),
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


_NVML_CLOCK_SM = 1
_nvml_handle: Any = None
_nvml_lib: Any = None
_nvml_tried = False


def _nvml() -> tuple[Any, Any]:
    """NVML through ctypes, initialised once. Returns (lib, handle) or (None, None).

    Not `torch.cuda.clock_rate()`: that routes through pynvml, which is not in the
    sandbox image, so it raised and the clock was silently recorded as None on every
    run. Two gate notes depend on this value -- the roofline throttle warning and the
    variance gate's kernel-versus-machine attribution -- so a silent None disabled
    both of them without anything failing.

    `libnvidia-ml.so.1` needs no package: it arrives with the driver, and the
    container runtime mounts it alongside the device nodes.
    """
    global _nvml_handle, _nvml_lib, _nvml_tried
    if _nvml_tried:
        return _nvml_lib, _nvml_handle
    _nvml_tried = True
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            return None, None
        handle = ctypes.c_void_p()
        index = torch.cuda.current_device()
        if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_int(index), ctypes.byref(handle)) != 0:
            return None, None
        _nvml_lib, _nvml_handle = lib, handle
    except Exception:
        _nvml_lib, _nvml_handle = None, None
    return _nvml_lib, _nvml_handle


def current_sm_clock_hz() -> float | None:
    """Actual SM clock right now, or None if it cannot be read.

    Matters because the roofline ceiling is derived from the *maximum* clock. A GPU
    that is thermally or power throttled never had access to that peak, so a
    perfectly good kernel measures low and a reader concludes the kernel is bad.
    Observed here: a sweep run immediately after 900 seconds of sustained autotuning
    measured roughly a third of the throughput the same kernel reached on an idle
    device.
    """
    lib, handle = _nvml()
    if lib is None or handle is None:
        return None
    try:
        mhz = ctypes.c_uint()
        if lib.nvmlDeviceGetClockInfo(handle, ctypes.c_int(_NVML_CLOCK_SM), ctypes.byref(mhz)) != 0:
            return None
        return float(mhz.value) * 1e6
    except Exception:
        return None


def measure_problem(problem: Problem, launch: Callable, repeats: int,
                    phase_fd: int = -1) -> dict[str, Any]:
    """Measure one shape. `phase_fd` brackets the timed loop for the supervisor.

    The supervisor cannot see inside this process, but it can see when bytes arrive
    on a pipe. Writing a marker either side of the timing loop lets it time that
    loop from outside, which is a far tighter bound than the whole worker lifetime:
    that includes several seconds of torch import and float64 reference computation,
    all of which was slack a forged measurement could spend.
    """
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
        make_input(tuple(original.shape), 0xA5A5A5, original.device,
                   exponent_range=0, dtype=original.dtype)
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

    # Rotate an input by one element before every sample, so the correct answer is
    # different every time and nothing computed earlier is still valid.
    #
    # Without this, a candidate could compute once and serve the result from cache for
    # the rest of the timed calls. On a compute-bound op the roofline catches that -- a
    # cached 4096-cubed GEMM implied 1529 TFLOP/s against an 83 TFLOP/s ceiling. On a
    # memory-bound op it does not, because a copy moves exactly the traffic the harness
    # charges: serving rmsnorm from cache was admitted at 89.7% of the bus, against
    # 25.9% for the honest kernel it replaced.
    #
    # A rotation rather than an offset, and that distinction cost a false rejection to
    # find. Adding a constant shifts the value distribution, which for attention moves
    # the softmax into a more saturated regime: it raised the measured error 1.7x and
    # failed a correct TF32 FlashAttention kernel that had been passing at 0.78 of its
    # bar. A rotation is a permutation, so every output value changes while the
    # distribution is exactly preserved and no tolerance has to be renegotiated.
    #
    # One pass, outside the timed bracket. The candidate cannot know which sample is
    # the last, and the output is checked against the reference for the final input
    # state, so every sample has to do the work.
    drift_key = next(iter(problem.inputs))
    drift_flat = problem.inputs[drift_key].view(-1)

    if phase_fd >= 0:
        os.write(phase_fd, b"B")

    samples: list[float] = []
    clocks: list[float] = []
    for rep in range(repeats):
        drift_flat.copy_(torch.roll(drift_flat, 1))
        # Drain it before starting the clock. The rotation is a device op and without
        # this it is still in flight when timing begins, so the kernel is charged for
        # it -- measured at about 17 percentage points of apparent bandwidth.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            launch(problem.inputs, out, problem.meta)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0 / inner)
        clock = current_sm_clock_hz()
        if clock:
            clocks.append(clock)

    if phase_fd >= 0:
        os.write(phase_fd, b"E")

    # Revalidate what the last *measured* call produced.
    #
    # `timed_written` asks only whether anything was written, and `timed_violation`
    # is computed over the finite elements because a NaN has no distance from
    # anything. Both are therefore satisfiable by a single element, and a candidate
    # that wrote one true value and NaN everywhere else during the timed calls was
    # admitted. So the non-finite flag from this comparison is reported rather than
    # discarded, which is all it took -- it was already being computed.
    timed_written = bool(torch.isfinite(out).any().item())
    # Against the reference for the *drifted* input, not the original: that is what
    # makes a cached answer detectably stale.
    timed_reference = problem.recompute()
    _, timed_rel, timed_violation, timed_nonfinite = deviation(
        out, timed_reference, problem.rel_tol
    )

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
        "timed_has_nonfinite": timed_nonfinite,
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


def run_worker(args: argparse.Namespace, result_fd: int, phase_fd: int = -1) -> int:
    """Measure the candidate and report on a file descriptor. Untrusted.

    This is the only process that imports candidate code, and it is deliberately
    ignorant: it is never told the nonce or the path the verdict is written to, so
    it cannot forge provenance and cannot write a verdict at all. It reports its
    numbers to `result_fd`, which supervisor.py opened -- see that file for why the
    split exists and what it does and does not buy.
    """
    started = time.perf_counter()
    payload: dict[str, Any] = {"op": args.op, "precision": args.precision}

    if not torch.cuda.is_available():
        os.write(result_fd, json.dumps({"error": "CUDA unavailable"}).encode())
        return 3

    builder, shapes = OPS[args.op]
    dev = torch.device("cuda:0")
    dt = storage_dtype(args.precision)
    launch = load_candidate(args.candidate)

    payload.update(device_ceilings())
    payload["repeats"] = max(5, args.repeats)
    payload["seed"] = args.seed

    shapes_out = []
    for index, dims in enumerate(shapes):
        problem = builder(dims[0], dims[1], args.seed + index, dev, dt)
        record = measure_problem(problem, launch, payload["repeats"], phase_fd)
        # The schema names these rows/cols for compatibility with the CUDA
        # harness; they are labels, and ops with other layouts report their own.
        record["rows"], record["cols"] = dims
        shapes_out.append(record)
        del problem
        torch.cuda.empty_cache()

    payload["shapes"] = shapes_out
    payload["worker_wall_ms"] = (time.perf_counter() - started) * 1000.0
    os.write(result_fd, json.dumps(payload).encode())
    os.close(result_fd)
    return 0


def main() -> int:
    """The worker half of the harness. supervisor.py is the entry point.

    Deliberately takes no output path and no nonce: this process imports candidate
    code, so anything it is told, the candidate can read.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=sorted(OPS))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--phase-fd",
        type=int,
        default=-1,
        help="Descriptor to bracket each timed loop on, so the supervisor can time "
             "the measurement from outside this process.",
    )
    parser.add_argument(
        "--result-fd",
        type=int,
        required=True,
        help="Descriptor to report the measurement on. Not a path: the process that "
             "runs candidate code is never told where the verdict goes.",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help="Numerical contract the candidate claims to honour: fp32, tf32, bf16 or fp16. "
             "Sets both the correctness tolerance and which hardware ceiling binds.",
    )
    args = parser.parse_args()
    return run_worker(args, args.result_fd, args.phase_fd)


if __name__ == "__main__":
    raise SystemExit(main())
