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


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    detail: str

    @property
    def blocking(self) -> bool:
        return self.status is GateStatus.FAIL


# fp32 accumulation over a few thousand elements; the reference is computed in
# double, so this bounds the candidate's error rather than the reference's.
MAX_REL_ERR = 2e-5
# Ratio of slowest to fastest repeat. Wider than this and the median is not a
# measurement, it is a mood.
MAX_TIMING_SPREAD = 3.0
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
        if shape["min_ms"] <= 0:
            return GateResult("variance", GateStatus.FAIL, f"{_shape_label(shape)} reported a non-positive time")
        ratio = shape["max_ms"] / shape["min_ms"]
        if ratio > worst_ratio:
            worst_ratio, worst_shape = ratio, _shape_label(shape)
    if worst_ratio > MAX_TIMING_SPREAD:
        return GateResult(
            "variance",
            GateStatus.FAIL,
            f"slowest repeat is {worst_ratio:.1f}x the fastest at {worst_shape}; "
            f"timing is too unstable to support a claim",
        )
    return GateResult(
        "variance",
        GateStatus.PASS,
        f"worst spread {worst_ratio:.2f}x over {measurement['repeats']} repeats",
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
    """Bound achieved DRAM bandwidth by the memory bus.

    Only shapes whose working set clears L2 are auditable this way. A smaller
    working set is served from cache, never crosses the memory bus, and can
    legitimately exceed DRAM peak -- an RTX 4090 baseline measures ~1820 GB/s at
    1024x4096 against a 1008 GB/s bus, and it is not cheating. Applying the DRAM
    ceiling there would accuse a correct kernel.
    """
    peak = measurement["peak_bandwidth_bytes_per_s"]
    l2 = measurement.get("l2_cache_bytes", 0)
    threshold = l2 * L2_CLEARANCE

    audited: list[tuple[str, float]] = []
    resident: list[str] = []
    for shape in measurement["shapes"]:
        label = _shape_label(shape)
        if shape["working_set_bytes"] < threshold:
            resident.append(label)
            continue
        achieved = shape["bytes_moved"] / (shape["median_ms"] / 1000.0)
        audited.append((label, achieved / peak))

    if not audited:
        return GateResult(
            "roofline",
            GateStatus.NOT_APPLICABLE,
            f"every shape fits within {L2_CLEARANCE:g}x L2 ({l2 / 1e6:.0f} MB); "
            "DRAM bandwidth does not bound cache-resident traffic",
        )

    label, fraction = max(audited, key=lambda item: item[1])
    if fraction > 1.0:
        return GateResult(
            "roofline",
            GateStatus.FAIL,
            f"{label} implies {fraction * 100:.0f}% of the {peak / 1e9:.0f} GB/s memory bus "
            f"({fraction:.1f}x the hardware maximum) with a working set too large to cache — "
            "physically impossible",
        )
    if fraction > IMPLAUSIBLE_FRACTION:
        return GateResult(
            "roofline",
            GateStatus.FAIL,
            f"{label} implies {fraction:.1%} of peak DRAM bandwidth; above "
            f"{IMPLAUSIBLE_FRACTION:.0%} points at the measurement, not the kernel",
        )
    note = f" ({len(resident)} cache-resident shape(s) not auditable)" if resident else ""
    return GateResult(
        "roofline",
        GateStatus.PASS,
        f"peak DRAM utilisation {fraction:.1%} at {label}{note}",
    )


ALL_GATES = (
    check_correctness,
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
            mark = {GateStatus.PASS: "pass", GateStatus.FAIL: "FAIL", GateStatus.NOT_APPLICABLE: "n/a "}[gate.status]
            lines.append(f"  [{mark}] {gate.name}: {gate.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "device": self.measurement.get("device"),
            "gates": [{"name": g.name, "status": g.status.value, "detail": g.detail} for g in self.gates],
        }


def run_gates(measurement: dict[str, Any]) -> Preflight:
    """Apply every gate. A kernel is admitted only if none fail.

    NOT_APPLICABLE never admits on its own -- it records that a gate could not
    bound this measurement, which is different from the measurement being sound.
    """
    results = tuple(gate(measurement) for gate in ALL_GATES)
    return Preflight(
        admitted=not any(r.blocking for r in results),
        gates=results,
        measurement=measurement,
    )
