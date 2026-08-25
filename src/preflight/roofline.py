"""Roofline audit: is a claimed kernel runtime physically possible at all?

Background. In February 2025 Sakana AI published "The AI CUDA Engineer", reporting
10-100x speedups on KernelBench workloads. The kernels had not been optimised. The
agent had found a way to exploit the benchmark harness, and several reported
figures implied throughput roughly 30x above the hardware's theoretical maximum --
a result Tri Dao publicly pointed out was simply not possible on the silicon.

The lesson generalises past that one incident: an agent that both writes a kernel
and reports its own speedup has every incentive and every opportunity to be wrong
in its own favour. Correctness tests do not catch it, because a kernel can be
numerically perfect and still be timed dishonestly.

Physics does catch it. A memory-bound kernel cannot move bytes faster than the
memory bus, and a compute-bound kernel cannot retire more FLOPs than the SMs can
issue. Those ceilings come from :mod:`preflight.device`, straight from the driver. This
module converts a claimed runtime into the throughput it implies and compares that
against the binding ceiling.

The audit is deliberately one-directional. It cannot prove a kernel is fast. It can
only prove a claim is impossible, or fail to. That asymmetry is the point: a gate
that rejects the physically impossible is sound even when it is not complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from preflight.device import DeviceSpec

# A pure float4 device-to-device copy -- the simplest memory-bound kernel that can
# be written -- measured 91.2% of theoretical peak on an RTX 4090 (919.7 of
# 1008.1 GB/s). A kernel that also does arithmetic, and therefore has more to do
# per byte, exceeding that by a wide margin indicates a measurement artefact
# rather than an optimisation. 0.95 leaves headroom above the observed figure
# while staying below the physical wall.
DEFAULT_IMPLAUSIBLE_FRACTION = 0.95


class Bound(str, Enum):
    """Which ceiling binds this operation."""

    MEMORY = "memory"
    COMPUTE = "compute"


class Verdict(str, Enum):
    """Outcome of a roofline audit."""

    PLAUSIBLE = "plausible"
    """Throughput sits below the empirical ceiling. Says nothing about quality."""

    IMPLAUSIBLE = "implausible"
    """Above what a trivial kernel achieves on this device. Suspect the harness."""

    IMPOSSIBLE = "impossible"
    """Above the hardware's theoretical maximum. The measurement is wrong."""

    UNVERIFIABLE = "unverifiable"
    """No ceiling available for this device/op -- refuse to render a verdict."""


@dataclass(frozen=True)
class OpProfile:
    """The irreducible cost of one invocation of an operation.

    ``min_bytes_moved`` is the *compulsory* traffic: bytes that must cross the
    memory bus even with a perfect cache and zero redundancy. For elementwise work
    that is inputs read plus outputs written, counted once each. Understating it
    inflates the apparent ceiling and weakens the gate, so it should be derived
    from the operation's definition rather than from an implementation.
    """

    name: str
    min_bytes_moved: int
    flops: int

    def __post_init__(self) -> None:
        if self.min_bytes_moved <= 0:
            raise ValueError(f"{self.name}: min_bytes_moved must be positive")
        if self.flops < 0:
            raise ValueError(f"{self.name}: flops cannot be negative")

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of compulsory traffic."""
        return self.flops / self.min_bytes_moved


@dataclass(frozen=True)
class RooflineAudit:
    """Result of auditing one timing claim against one device."""

    op: str
    verdict: Verdict
    bound: Bound | None
    seconds: float
    achieved_bytes_per_s: float
    achieved_flops: float
    ceiling: float
    fraction_of_ceiling: float | None
    detail: str

    @property
    def rejected(self) -> bool:
        return self.verdict in (Verdict.IMPOSSIBLE, Verdict.IMPLAUSIBLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "verdict": self.verdict.value,
            "bound": self.bound.value if self.bound else None,
            "seconds": self.seconds,
            "achieved_gb_s": round(self.achieved_bytes_per_s / 1e9, 2),
            "achieved_tflops": round(self.achieved_flops / 1e12, 3),
            "ceiling": round(self.ceiling, 2),
            "fraction_of_ceiling": (
                round(self.fraction_of_ceiling, 4) if self.fraction_of_ceiling is not None else None
            ),
            "detail": self.detail,
        }


def ridge_point(device: DeviceSpec) -> float | None:
    """Arithmetic intensity at which an op stops being memory-bound.

    Below this, the memory bus binds; above it, the SMs do. None when the device's
    compute ceiling is unknown.
    """
    peak_flops = device.peak_fp32_flops
    if peak_flops is None:
        return None
    return peak_flops / device.peak_memory_bandwidth_bytes_per_s


def audit(
    op: OpProfile,
    seconds: float,
    device: DeviceSpec,
    implausible_fraction: float = DEFAULT_IMPLAUSIBLE_FRACTION,
) -> RooflineAudit:
    """Audit a claim that ``op`` completed in ``seconds`` on ``device``.

    Chooses the binding ceiling from the operation's arithmetic intensity, converts
    the claimed runtime into achieved throughput against that ceiling, and grades
    it. A non-positive runtime is reported as IMPOSSIBLE rather than raising -- a
    zero-time claim is a real thing agents report when they forget to synchronise,
    and it is exactly what this gate exists to catch.
    """
    achieved_bytes_per_s = op.min_bytes_moved / seconds if seconds > 0 else float("inf")
    achieved_flops = op.flops / seconds if seconds > 0 else float("inf")

    if seconds <= 0:
        return RooflineAudit(
            op=op.name,
            verdict=Verdict.IMPOSSIBLE,
            bound=None,
            seconds=seconds,
            achieved_bytes_per_s=achieved_bytes_per_s,
            achieved_flops=achieved_flops,
            ceiling=0.0,
            fraction_of_ceiling=None,
            detail=(
                f"claimed runtime is {seconds:g}s. A kernel cannot complete in zero or "
                "negative time; the timed region almost certainly omits a device "
                "synchronise, so it measured launch overhead rather than execution."
            ),
        )

    ridge = ridge_point(device)
    ceiling_unknown = ridge is None
    if ceiling_unknown:
        # Without a compute ceiling the ridge point is unknown, so we cannot tell
        # whether this op is memory- or compute-bound. Auditing against the memory
        # ceiling anyway is still *sound in one direction*: nothing can move bytes
        # faster than the bus regardless of how much arithmetic it does. So a
        # breach is still IMPOSSIBLE, but passing cannot be called PLAUSIBLE --
        # a compute-bound claim could be impossible in a way we cannot see.
        bound = Bound.MEMORY
    else:
        assert ridge is not None
        bound = Bound.MEMORY if op.arithmetic_intensity < ridge else Bound.COMPUTE

    if bound is Bound.MEMORY:
        ceiling = device.peak_memory_bandwidth_bytes_per_s
        achieved = achieved_bytes_per_s
        units = "GB/s"
        scale = 1e9
    else:
        peak_flops = device.peak_fp32_flops
        # Unreachable: ridge_point() is None exactly when peak_fp32_flops is, and
        # that path forces Bound.MEMORY above.
        assert peak_flops is not None, "compute-bound with no compute ceiling"
        ceiling = peak_flops
        achieved = achieved_flops
        units = "TFLOP/s"
        scale = 1e12

    fraction = achieved / ceiling

    if fraction > 1.0:
        verdict = Verdict.IMPOSSIBLE
        detail = (
            f"implies {achieved / scale:.1f} {units} against a theoretical peak of "
            f"{ceiling / scale:.1f} {units} ({fraction:.1f}x the hardware maximum). "
            f"{device.name} physically cannot do this: the claim is a measurement "
            f"artefact or a benchmark exploit, not an optimisation."
        )
    elif ceiling_unknown:
        # Sound-in-one-direction only: see the ceiling_unknown branch above.
        verdict = Verdict.UNVERIFIABLE
        detail = (
            f"implies {achieved / scale:.1f} {units}, under the "
            f"{ceiling / scale:.1f} {units} memory ceiling, but the FP32 ceiling for "
            f"compute capability {device.compute_capability} is unknown. Cannot rule "
            f"out a compute-bound impossibility, so this is not a pass."
        )
    elif fraction > implausible_fraction:
        verdict = Verdict.IMPLAUSIBLE
        detail = (
            f"implies {achieved / scale:.1f} {units}, or {fraction:.1%} of the "
            f"{ceiling / scale:.1f} {units} peak. A trivial float4 copy reaches about "
            f"91% on this class of device; an op doing real arithmetic exceeding "
            f"{implausible_fraction:.0%} points at the harness, not the kernel."
        )
    else:
        verdict = Verdict.PLAUSIBLE
        detail = (
            f"{achieved / scale:.1f} {units} = {fraction:.1%} of the "
            f"{ceiling / scale:.1f} {units} {bound.value} ceiling."
        )

    return RooflineAudit(
        op=op.name,
        verdict=verdict,
        bound=bound,
        seconds=seconds,
        achieved_bytes_per_s=achieved_bytes_per_s,
        achieved_flops=achieved_flops,
        ceiling=ceiling,
        fraction_of_ceiling=fraction,
        detail=detail,
    )


def rmsnorm_profile(rows: int, hidden: int, dtype_bytes: int) -> OpProfile:
    """Compulsory cost of RMSNorm over ``rows`` x ``hidden`` in ``dtype_bytes``.

    Traffic: read the input once, write the output once. The weight vector is
    ``hidden`` elements and is reread per row in principle but lives in cache, so
    it is excluded from compulsory traffic -- excluding it makes the ceiling
    stricter, which is the safe direction for a gate.

    FLOPs: per element, one multiply for the square, one add for the running sum,
    one multiply by the reciprocal norm and one by the weight. The single rsqrt per
    row is negligible. Four per element is the conventional accounting.
    """
    elements = rows * hidden
    return OpProfile(
        name=f"rmsnorm[{rows}x{hidden}:{dtype_bytes}B]",
        min_bytes_moved=2 * elements * dtype_bytes,
        flops=4 * elements,
    )
