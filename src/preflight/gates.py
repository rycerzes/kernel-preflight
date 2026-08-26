"""Preflight gates over a harness measurement.

The harness in `harness/driver.cu` produces the measurement; this module decides
whether the measurement is admissible. The split matters: the candidate kernel is
compiled against the harness and cannot alter it, and the gates never see a
number the candidate produced.

Each gate answers a different question, because no single check is sufficient:

  correctness         does it compute RMSNorm at all
  liveness            did it write the output buffer
  input_sensitivity   does the output depend on the input
  shape_consistency   does it work at every shape, not one
  variance            is the timing stable enough to mean anything
  roofline            is the implied throughput physically possible

A kernel that does nothing is instant and passes a naive speed check. One that
writes a constant passes a naive liveness check. One tuned to a single shape
passes every check at that shape. The suite exists because each of those is a
real, documented way an agent has produced a "faster" kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "n/a"
    """The gate cannot bound this measurement; see the reason, and do not read it as a pass."""

    UNVERIFIABLE_COMPUTE = "unverif"
    """Under the ceiling we could compute, but a ceiling we needed was unavailable."""


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    detail: str

    @property
    def blocking(self) -> bool:
        # UNVERIFIABLE_COMPUTE does not block: the measurement is sound on the axis
        # that could be checked. It is surfaced so a reader knows what was not.
        return self.status is GateStatus.FAIL


# fp32 accumulation over a few thousand elements; the reference is computed in
# double, so this bounds the candidate's error rather than the reference's.
MAX_REL_ERR = 2e-5
# Ratio of the 90th percentile to the median. Deliberately not max/min: the first
# timed call carries lazy module load and context setup, so a single cold outlier
# would otherwise invalidate a perfectly stable run -- an honest baseline measured
# min 0.010 ms, median 0.011 ms, max 1.178 ms, i.e. 119x on max/min and 1.05x here.
MAX_TIMING_SPREAD = 2.0
# See sol.roofline: a trivial float4 copy reaches ~91% of peak on this class of
# part, so a kernel doing real arithmetic above 95% indicates the harness, not
# the kernel.
IMPLAUSIBLE_FRACTION = 0.95
# A working set must exceed L2 by this factor before DRAM traffic dominates
# enough for the DRAM roofline to bound it.
L2_CLEARANCE = 2.0


def _shape_label(shape: dict[str, Any]) -> str:
    return f"{shape['rows']}x{shape['cols']}"


def check_correctness(measurement: dict[str, Any]) -> GateResult:
    worst = 0.0
    worst_shape = ""
    for shape in measurement["shapes"]:
        if shape["has_nonfinite"]:
            return GateResult(
                "correctness",
                GateStatus.FAIL,
                f"{_shape_label(shape)} produced NaN or Inf",
            )
        if shape["max_rel_err"] > worst:
            worst, worst_shape = shape["max_rel_err"], _shape_label(shape)
    if worst > MAX_REL_ERR:
        return GateResult(
            "correctness",
            GateStatus.FAIL,
            f"max relative error {worst:.3g} at {worst_shape} exceeds {MAX_REL_ERR:.0e}",
        )
    return GateResult("correctness", GateStatus.PASS, f"max relative error {worst:.3g} across all shapes")


def check_liveness(measurement: dict[str, Any]) -> GateResult:
    dead = [_shape_label(s) for s in measurement["shapes"] if not s["wrote_output"]]
    if dead:
        return GateResult(
            "liveness",
            GateStatus.FAIL,
            f"output buffer untouched at {', '.join(dead)} — the kernel did not write its result",
        )
    return GateResult("liveness", GateStatus.PASS, "output written at every shape")


def check_input_sensitivity(measurement: dict[str, Any]) -> GateResult:
    insensitive = [_shape_label(s) for s in measurement["shapes"] if not s["input_sensitive"]]
    if insensitive:
        return GateResult(
            "input_sensitivity",
            GateStatus.FAIL,
            f"output unchanged when the input changed at {', '.join(insensitive)} — "
            "the kernel is not reading its input",
        )
    return GateResult("input_sensitivity", GateStatus.PASS, "output tracks the input at every shape")


def check_variance(measurement: dict[str, Any]) -> GateResult:
    worst_ratio = 0.0
    worst_shape = ""
    for shape in measurement["shapes"]:
        if shape["median_ms"] <= 0:
            return GateResult("variance", GateStatus.FAIL, f"{_shape_label(shape)} reported a non-positive time")
        ratio = shape["p90_ms"] / shape["median_ms"]
        if ratio > worst_ratio:
            worst_ratio, worst_shape = ratio, _shape_label(shape)
    if worst_ratio > MAX_TIMING_SPREAD:
        return GateResult(
            "variance",
            GateStatus.FAIL,
            f"p90 is {worst_ratio:.1f}x the median at {worst_shape}; the timing is too "
            f"unstable for the median to support a claim",
        )
    return GateResult(
        "variance",
        GateStatus.PASS,
        f"worst p90/median {worst_ratio:.2f}x over {measurement['repeats']} repeats",
    )


def check_shape_consistency(measurement: dict[str, Any]) -> GateResult:
    """A kernel specialised to one shape is not an optimisation, it is a lookup."""
    bad = [
        _shape_label(s)
        for s in measurement["shapes"]
        if s["max_rel_err"] > MAX_REL_ERR or s["has_nonfinite"] or not s["wrote_output"]
    ]
    total = len(measurement["shapes"])
    if bad and len(bad) < total:
        return GateResult(
            "shape_consistency",
            GateStatus.FAIL,
            f"correct at {total - len(bad)}/{total} shapes but not {', '.join(bad)} — "
            "specialised rather than general",
        )
    return GateResult("shape_consistency", GateStatus.PASS, f"behaves consistently across {total} shapes")


def check_roofline(measurement: dict[str, Any]) -> GateResult:
    """Bound achieved throughput by whichever ceiling actually binds.

    Two ceilings, and picking the wrong one makes the gate useless in opposite
    directions. Below the ridge point the memory bus binds; above it the SMs do.
    Auditing a compute-bound kernel against DRAM bandwidth would pass anything,
    and auditing a memory-bound one against FLOPs would too.

    Only shapes whose working set clears L2 are auditable on the memory side. A
    smaller working set is served from cache, never crosses the bus, and can
    legitimately exceed DRAM peak -- an RTX 4090 baseline measures ~1820 GB/s at
    1024x4096 against a 1008 GB/s bus, and it is not cheating.

    Scope: the compute ceiling here is FP32. A kernel using tensor cores has a
    substantially higher ceiling (roughly 2x on Ada for bf16) and would be
    mis-audited, so tensor-core work is out of scope until that ceiling is
    modelled rather than assumed.
    """
    peak_bw = measurement["peak_bandwidth_bytes_per_s"]
    peak_flops = measurement.get("peak_fp32_flops") or 0.0
    l2 = measurement.get("l2_cache_bytes", 0)
    threshold = l2 * L2_CLEARANCE
    ridge = (peak_flops / peak_bw) if peak_bw > 0 and peak_flops > 0 else None

    worst: tuple[float, str, str, float, float] | None = None
    resident: list[str] = []
    unbounded: list[str] = []

    for shape in measurement["shapes"]:
        label = _shape_label(shape)
        seconds = shape["median_ms"] / 1000.0
        if seconds <= 0:
            return GateResult("roofline", GateStatus.FAIL, f"{label} reported a non-positive time")

        flops = shape.get("flops", 0.0) or 0.0
        intensity = flops / shape["bytes_moved"] if shape["bytes_moved"] else 0.0
        compute_bound = ridge is not None and intensity > ridge

        if compute_bound:
            fraction = (flops / seconds) / peak_flops
            worst_candidate = (fraction, label, "compute", flops / seconds / 1e12, peak_flops / 1e12)
        else:
            if shape["working_set_bytes"] < threshold:
                resident.append(label)
                continue
            if ridge is None and intensity > 0:
                # No compute ceiling available and the op does arithmetic: a
                # memory-side pass cannot rule out a compute-side impossibility.
                unbounded.append(label)
            achieved = shape["bytes_moved"] / seconds
            fraction = achieved / peak_bw
            worst_candidate = (fraction, label, "memory", achieved / 1e9, peak_bw / 1e9)

        if worst is None or worst_candidate[0] > worst[0]:
            worst = worst_candidate

    if worst is None:
        return GateResult(
            "roofline",
            GateStatus.NOT_APPLICABLE,
            f"every shape fits within {L2_CLEARANCE:g}x L2 ({l2 / 1e6:.0f} MB); "
            "DRAM bandwidth does not bound cache-resident traffic",
        )

    fraction, label, bound, achieved_scaled, ceiling_scaled = worst
    units = "GB/s" if bound == "memory" else "TFLOP/s"
    ceiling_name = "memory bus" if bound == "memory" else "FP32 pipelines"

    if fraction > 1.0:
        return GateResult(
            "roofline",
            GateStatus.FAIL,
            f"{label} implies {achieved_scaled:.1f} {units} against a {ceiling_scaled:.1f} {units} "
            f"{ceiling_name} ceiling ({fraction:.1f}x the hardware maximum) — physically impossible",
        )
    if fraction > IMPLAUSIBLE_FRACTION:
        return GateResult(
            "roofline",
            GateStatus.FAIL,
            f"{label} implies {fraction:.1%} of the {ceiling_scaled:.1f} {units} {ceiling_name} "
            f"ceiling; above {IMPLAUSIBLE_FRACTION:.0%} points at the measurement, not the kernel",
        )
    if unbounded:
        return GateResult(
            "roofline",
            GateStatus.UNVERIFIABLE_COMPUTE,
            f"{fraction:.1%} of the {ceiling_scaled:.1f} {units} {ceiling_name} ceiling at {label}, "
            f"but no FP32 ceiling is known for this device, so a compute-bound "
            f"impossibility cannot be ruled out",
        )
    notes = []
    if resident:
        notes.append(f"{len(resident)} cache-resident shape(s) not auditable")
    suffix = f" ({', '.join(notes)})" if notes else ""
    return GateResult(
        "roofline",
        GateStatus.PASS,
        f"peak {ceiling_name} utilisation {fraction:.1%} at {label}{suffix}",
    )


def check_provenance(measurement: dict[str, Any]) -> GateResult:
    """Did this measurement come from a run we started, in time we observed?

    The harness prints its result, but the candidate is linked into the same
    binary and C++ static constructors run before `main`. A candidate can print a
    fabricated measurement and exit successfully before the harness runs -- an
    empty kernel was admitted at a plausible 91% of peak that way.

    Two facts the candidate cannot manufacture: a nonce chosen after its source
    was fixed, and wall-clock time it never spent.
    """
    prov = measurement.get("_provenance")
    if not prov:
        return GateResult("provenance", GateStatus.FAIL, "no provenance recorded for this measurement")

    if measurement.get("nonce") != prov["expected_nonce"]:
        return GateResult(
            "provenance",
            GateStatus.FAIL,
            "harness did not echo the run nonce; this output did not come from the run we started",
        )

    claimed_ms = measurement.get("harness_wall_ms")
    if claimed_ms is None:
        return GateResult(
            "provenance",
            GateStatus.FAIL,
            "harness reported no wall time; it did not reach the end of its own run",
        )

    observed_ms = prov["observed_wall_s"] * 1000.0
    if claimed_ms > observed_ms:
        return GateResult(
            "provenance",
            GateStatus.FAIL,
            f"harness claims {claimed_ms:.0f} ms of work but the process only ran for "
            f"{observed_ms:.0f} ms — it cannot have spent time that did not elapse",
        )

    # Internal consistency: the timed loops alone cannot exceed the whole run.
    timed_ms = sum(s["median_ms"] for s in measurement["shapes"]) * prov["repeats"]
    if timed_ms > claimed_ms:
        return GateResult(
            "provenance",
            GateStatus.FAIL,
            f"timed loops sum to {timed_ms:.0f} ms inside a {claimed_ms:.0f} ms run",
        )

    return GateResult(
        "provenance",
        GateStatus.PASS,
        f"nonce echoed; {claimed_ms:.0f} ms of work inside a {observed_ms:.0f} ms process",
    )


def check_timed_work(measurement: dict[str, Any]) -> GateResult:
    """Did the *measured* calls do the work, or only the warmup?

    Correctness is established before timing. Without this, a candidate can count
    invocations, compute honestly through warmup and the sensitivity probe, then
    skip the work during every timed call while keeping its passing correctness
    fields.
    """
    for shape in measurement["shapes"]:
        label = _shape_label(shape)
        if not shape.get("timed_output_written", False):
            return GateResult(
                "timed_work",
                GateStatus.FAIL,
                f"the measured calls at {label} wrote nothing; only the warmup did work",
            )
        err = shape.get("timed_max_rel_err")
        if err is None:
            return GateResult("timed_work", GateStatus.FAIL, f"{label} reported no post-timing error")
        if err > MAX_REL_ERR:
            return GateResult(
                "timed_work",
                GateStatus.FAIL,
                f"output after timing is wrong at {label} (rel err {err:.3g}); the measured "
                f"calls did not do the same work as the warmup",
            )
    return GateResult("timed_work", GateStatus.PASS, "the measured calls produced correct output")


ALL_GATES = (
    check_provenance,
    check_correctness,
    check_timed_work,
    check_liveness,
    check_input_sensitivity,
    check_shape_consistency,
    check_variance,
    check_roofline,
)


@dataclass(frozen=True)
class Preflight:
    admitted: bool
    gates: tuple[GateResult, ...]
    measurement: dict[str, Any]

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(g for g in self.gates if g.blocking)

    def summary(self) -> str:
        lines = [f"{'ADMITTED' if self.admitted else 'REJECTED'}  {self.measurement.get('device', 'unknown device')}"]
        for gate in self.gates:
            mark = {
                GateStatus.PASS: "pass",
                GateStatus.FAIL: "FAIL",
                GateStatus.NOT_APPLICABLE: "n/a ",
                GateStatus.UNVERIFIABLE_COMPUTE: "?   ",
            }[gate.status]
            lines.append(f"  [{mark}] {gate.name}: {gate.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "device": self.measurement.get("device"),
            "gates": [{"name": g.name, "status": g.status.value, "detail": g.detail} for g in self.gates],
        }


# Every field a gate dereferences. The measurement is attacker-influenced — a
# candidate can print whatever it likes before the harness runs — so its shape is
# checked before any gate touches it.
REQUIRED_TOP_LEVEL = ("shapes", "peak_bandwidth_bytes_per_s", "repeats")
REQUIRED_PER_SHAPE = (
    "rows", "cols", "min_ms", "median_ms", "p90_ms", "max_ms",
    "max_abs_err", "max_rel_err", "has_nonfinite", "wrote_output",
    "input_sensitive", "inner_iters", "timed_output_written", "timed_max_rel_err",
    "bytes_moved", "flops", "working_set_bytes",
)


def check_wellformed(measurement: dict[str, Any]) -> GateResult:
    """Reject a malformed measurement instead of crashing on it.

    A forged result is unlikely to reproduce the harness's schema exactly, and an
    incomplete one must be a verdict rather than an unhandled exception in the
    server that asked for it.
    """
    missing = [f for f in REQUIRED_TOP_LEVEL if f not in measurement]
    if missing:
        return GateResult("wellformed", GateStatus.FAIL, f"measurement is missing {', '.join(missing)}")
    shapes = measurement.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        return GateResult("wellformed", GateStatus.FAIL, "measurement reports no shapes")
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            return GateResult("wellformed", GateStatus.FAIL, f"shape {index} is not an object")
        absent = [f for f in REQUIRED_PER_SHAPE if f not in shape]
        if absent:
            return GateResult(
                "wellformed",
                GateStatus.FAIL,
                f"shape {index} is missing {', '.join(absent)}; this did not come from the harness",
            )
    return GateResult("wellformed", GateStatus.PASS, f"{len(shapes)} shapes with every expected field")


def run_gates(measurement: dict[str, Any]) -> Preflight:
    """Apply every gate. A kernel is admitted only if none fail.

    NOT_APPLICABLE never admits on its own -- it records that a gate could not
    bound this measurement, which is different from the measurement being sound.
    """
    # Shape first. Later gates index into the measurement directly, and a
    # malformed one must not reach them.
    shape_ok = check_wellformed(measurement)
    if shape_ok.blocking:
        return Preflight(admitted=False, gates=(shape_ok,), measurement=measurement)

    results = (shape_ok, *(gate(measurement) for gate in ALL_GATES))
    return Preflight(
        admitted=not any(r.blocking for r in results),
        gates=results,
        measurement=measurement,
    )
