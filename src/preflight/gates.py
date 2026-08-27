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


# Floor for the harness-reported per-shape tolerance. Ops differ in conditioning
# -- a 4096-deep GEMM cannot be held to the bar an elementwise map meets -- so the
# harness derives a tolerance from accumulation depth and reports it. This is only
# the floor, so a harness reporting an absurdly loose tolerance cannot wave a
# broken kernel through, and a bit-exact op can still report zero.
MIN_REL_TOL = 1e-6
# Interquartile ratio, p75/p25. The gate asks one question -- is the median a
# trustworthy summary of this kernel's runtime -- and that calls for a robust
# dispersion measure rather than a tail quantile.
#
# max/min failed an honest baseline at 119x on a single cold first call. p90/median
# fixed the systematic case but not the tail: inside an eighteen-case sweep on a hot
# device, a kernel measuring 1.00-1.06 in isolation once spiked past 2.0 and was
# rejected. p75/p25 is unmoved by a handful of spikes by construction, while still
# failing a genuinely bimodal kernel, because that shows up inside the quartiles.
MAX_TIMING_SPREAD = 1.5
# See sol.roofline: a trivial float4 copy reaches ~91% of peak on this class of
# part, so a kernel doing real arithmetic above 95% indicates the harness, not
# the kernel.
IMPLAUSIBLE_FRACTION = 0.95
# A working set must exceed L2 by this factor before DRAM traffic dominates
# enough for the DRAM roofline to bound it.
L2_CLEARANCE = 2.0

# Dense FLOPs per SM per clock, by compute capability and precision class.
#
# Derived from published *dense* peak TFLOPS divided by (SM count x boost clock),
# which lands on exact powers of two and is the check that the derivation is
# sound. Vendor headline numbers usually quote 2:4 structured sparsity and are
# double these; using those would double every ceiling and make impossible claims
# look merely ambitious.
#
# The entry that matters most is the pair on 8.0 versus 8.9. On an A100, TF32
# tensor cores are 8x the FP32 pipelines (1024 against 128). On Ada consumer parts
# they are identical (256 and 256). So auditing a TF32 kernel against the FP32
# ceiling is coincidentally correct on a 4090 and wrong by 8x on an A100 — which
# is the entire reason this table exists rather than a single FP32 number.
#
# Absent entries are deliberate. Hopper's figures do not divide cleanly with the
# clocks available here, so rather than guess, a kernel on an unlisted device or
# precision is reported unverifiable on the compute axis.
#
# The tensor-core rows are a *lower* bound on what the capability can do, not the
# figure for every part that reports it. Within one compute capability NVIDIA ships
# SKUs whose tensor throughput differs by exactly 2x: the RTX 3090 is rated at 35.6
# dense TF32 TFLOPS and the A40 at 74.8, and both are compute capability 8.6. Asked
# about it directly, NVIDIA's answer was that "the TC units in each of those 2 GPUs
# do not necessarily act in precisely the same way" and that "the detail description
# of the differences is unpublished". The same 2x gap separates the RTX 4090 from the
# RTX 6000 Ada at 8.9.
#
# So these values are the consumer rate, which is the rate verified on the hardware
# this was developed against. They are right for reporting utilisation on such a
# part and wrong by 2x as a limit for a professional one -- and a ceiling that is
# too low does not weaken the gate, it makes it accuse correct kernels. See
# TENSOR_SKU_SPREAD.
DENSE_FLOPS_PER_SM_CLOCK: dict[tuple[int, int], dict[str, int]] = {
    (7, 0): {"fp32": 128, "fp16": 1024},                              # V100
    (7, 5): {"fp32": 128, "fp16": 1024},                              # T4, Turing
    (8, 0): {"fp32": 128, "tf32": 1024, "fp16": 2048, "bf16": 2048},  # A100
    (8, 6): {"fp32": 256, "tf32": 256, "fp16": 512, "bf16": 512},     # GA10x
    (8, 9): {"fp32": 256, "tf32": 256, "fp16": 512, "bf16": 512},     # Ada
}

# Precisions that execute on tensor cores, and therefore carry the SKU uncertainty
# described above. fp32 does not: it runs on the CUDA cores, where the rate is
# 2 FLOP per core per clock on every part of a given capability, which is why the
# fp32 rows can be trusted as limits rather than merely as reference points.
TENSOR_PRECISIONS = ("tf32", "bf16", "fp16")

# How far above the tabulated tensor ceiling a claim must land before it can be
# called impossible rather than merely unverifiable. Exactly the published spread
# between consumer and professional parts of the same compute capability.
#
# This costs the gate sensitivity on tensor-core precisions and is still the right
# trade. The failure this exists to catch was ~30x above the hardware maximum, so a
# 2x margin does not hide it -- and a false accusation of a correct kernel is the
# more expensive mistake, which this project has now made five times.
TENSOR_SKU_SPREAD = 2.0

# Mantissa bits per precision class, used to scale the correctness tolerance. A
# TF32 multiply keeps 10 bits against fp32's 23, so its results carry roughly
# 2^13 times more relative error and cannot be held to an fp32 bar.
MANTISSA_BITS = {"fp32": 23, "tf32": 10, "bf16": 7, "fp16": 10}


def compute_ceiling(measurement: dict[str, Any]) -> tuple[float | None, str]:
    """Dense FLOP ceiling for the declared precision, or None with a reason."""
    precision = str(measurement.get("precision", "fp32"))
    cap = str(measurement.get("compute_capability", ""))
    sm_count = measurement.get("sm_count")
    sm_clock_hz = measurement.get("sm_clock_hz")
    if not sm_count or not sm_clock_hz:
        return None, "the harness did not report SM count and clock"
    try:
        major, minor = (int(part) for part in cap.split("."))
    except ValueError:
        return None, f"unparseable compute capability {cap!r}"
    table = DENSE_FLOPS_PER_SM_CLOCK.get((major, minor))
    if table is None:
        return None, f"no dense-FLOP figures recorded for compute capability {cap}"
    per_clock = table.get(precision)
    if per_clock is None:
        return None, f"compute capability {cap} has no {precision} tensor path recorded"
    return float(sm_count) * per_clock * float(sm_clock_hz), precision


def _shape_label(shape: dict[str, Any]) -> str:
    return f"{shape['rows']}x{shape['cols']}"


# TF32 rounds operands to 10 mantissa bits and accumulates in fp32, so its error
# is operand quantisation: a flat bar, with the usual safety factor.
TF32_TOLERANCE = 8.0 * 2.0 ** -(MANTISSA_BITS["tf32"] + 1)


def _times(x: float) -> str:
    """Multiples of tolerance, readable at both ends of the range.

    A bit-exact op reports an effectively infinite violation for any difference at
    all, which the CUDA harness clamps to 1e308 so the JSON stays parseable. `.1f`
    renders that as three hundred digits.
    """
    return f"{x:.3g}" if x >= 1e4 else f"{x:.1f}"


def _violation_scale(measurement: dict[str, Any], shape: dict[str, Any]) -> float:
    """How much the declared precision widens this shape's bar.

    `violation` is expressed in units of the harness's own `rel_tol`, which is
    derived from fp32 and grows as sqrt(depth). For bf16 and fp16 this is 1.0: the
    harness can see those in the dtype and has already folded them into `rel_tol`.
    Scaling twice is how a bf16 tolerance once became 800%.

    TF32 is the exception, and not only because it is invisible in the dtype -- it
    is a compute mode over fp32 storage. It rounds its *operands* to 10 mantissa
    bits and accumulates in fp32, so the error is set by operand quantisation and
    does not grow with depth. The measured deviation of a Triton TF32 matmul is
    flat at about 1.5e-3 from k=512 to k=4096, while the harness's fp32 `rel_tol`
    grows 2.8x across that range.

    A fixed multiplier therefore gets the shape of the curve wrong as well as its
    height. Multiplying a sqrt(depth) base by 2^13 reached 0.5 at k=4096 -- about
    128x looser than the contract, and loosening further the deeper the reduction,
    which is backwards. The bar is `TF32_TOLERANCE` instead, converted here into
    the units `violation` is already reported in.

    Never below 1.0: for a reduction deep enough that the fp32 accumulation bar
    exceeds the TF32 quantisation bar, the wider of the two is the honest one, and
    tightening below what the harness itself demands of fp32 would fail correct
    kernels.
    """
    if str(measurement.get("precision", "fp32")) != "tf32":
        return 1.0
    reported = float(shape.get("rel_tol", 0.0))
    if reported <= 0.0:
        return 1.0
    return max(1.0, TF32_TOLERANCE / reported)


def check_correctness(measurement: dict[str, Any]) -> GateResult:
    worst = 0.0
    worst_shape = ""
    precision = str(measurement.get("precision", "fp32"))
    for shape in measurement["shapes"]:
        if shape["has_nonfinite"]:
            return GateResult(
                "correctness",
                GateStatus.FAIL,
                f"{_shape_label(shape)} produced NaN or Inf",
            )
        # `violation` is |got - want| / (rel_tol * (rms|want| + |want|)), so 1.0
        # sits exactly on the tolerance. Pure relative error is not usable here:
        # it fails torch's own matmul, because cancellation makes some reference
        # values near zero and the ratio then explodes on a correct kernel.
        # `violation` is measured against the harness's fp32-derived tolerance;
        # a declared reduced precision widens the bar -- see _violation_scale for
        # why tf32 widens to a flat bar rather than a proportional one.
        violation = float(shape.get("violation", 0.0)) / _violation_scale(measurement, shape)
        if violation > 1.0:
            return GateResult(
                "correctness",
                GateStatus.FAIL,
                f"{_shape_label(shape)} exceeds the {precision} tolerance by {_times(violation)}x "
                f"(max abs error {shape['max_abs_err']:.3g}). If the kernel deliberately "
                f"computes in reduced precision it must declare it; an undeclared downgrade "
                f"buys speed with accuracy the caller did not agree to",
            )
        if violation > worst:
            worst, worst_shape = violation, _shape_label(shape)
    return GateResult(
        "correctness",
        GateStatus.PASS,
        f"worst deviation {worst:.2f}x of the {precision} tolerance at {worst_shape or 'all shapes'}",
    )


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


def _contention_note(measurement: dict[str, Any]) -> str:
    """Whether the device looked contended while this was measured.

    Exceeding the spread threshold means a whole quartile of samples ran far slower
    than another quartile, which is not one scheduler hiccup -- it is either an
    unstable kernel or a device that was busy for the duration. The report cannot
    tell those apart on the ratio alone, and they call for opposite responses: one
    is "re-run this", the other is "rewrite this".

    The clock the harness sampled during the run separates them cheaply. Measured
    over 28 runs of one honest kernel, the spread stayed within 1.13x at every
    repeat count between 5 and 50, so a reading past 1.5 is unusual enough to be
    worth explaining rather than just reporting.
    """
    observed = [s["sm_clock_hz_observed"] for s in measurement["shapes"] if s.get("sm_clock_hz_observed")]
    peak = measurement.get("sm_clock_hz")
    if not observed or not peak:
        return ""
    ratio = (sum(observed) / len(observed)) / float(peak)
    if ratio < 0.9:
        return (
            f". The device ran at {ratio:.0%} of peak clock during this measurement, so it "
            f"was throttled or contended -- re-run on an idle device before changing the kernel"
        )
    return ". The device held {:.0%} of peak clock throughout, so this is the kernel rather than the machine".format(ratio)


def check_variance(measurement: dict[str, Any]) -> GateResult:
    worst_ratio = 0.0
    worst_shape = ""
    for shape in measurement["shapes"]:
        if shape["median_ms"] <= 0 or shape.get("p25_ms", 0) <= 0:
            return GateResult("variance", GateStatus.FAIL, f"{_shape_label(shape)} reported a non-positive time")
        ratio = shape["p75_ms"] / shape["p25_ms"]
        if ratio > worst_ratio:
            worst_ratio, worst_shape = ratio, _shape_label(shape)
    if worst_ratio > MAX_TIMING_SPREAD:
        return GateResult(
            "variance",
            GateStatus.FAIL,
            f"interquartile spread is {worst_ratio:.2f}x at {worst_shape}; half the samples "
            f"disagree by that much, so the median does not summarise this kernel"
            f"{_contention_note(measurement)}",
        )
    spikes = sum(int(s.get("outliers", 0)) for s in measurement["shapes"])
    note = f"; {spikes} sample(s) above 2x median, not counted against it" if spikes else ""
    return GateResult(
        "variance",
        GateStatus.PASS,
        f"worst interquartile spread {worst_ratio:.2f}x over {measurement['repeats']} repeats{note}",
    )


def check_shape_consistency(measurement: dict[str, Any]) -> GateResult:
    """A kernel specialised to one shape is not an optimisation, it is a lookup."""
    bad = [
        _shape_label(s)
        for s in measurement["shapes"]
        if float(s.get("violation", 0.0)) / _violation_scale(measurement, s) > 1.0
        or s["has_nonfinite"]
        or not s["wrote_output"]
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
    ceiling_flops, ceiling_note = compute_ceiling(measurement)
    peak_flops = ceiling_flops or 0.0
    precision = str(measurement.get("precision", "fp32"))
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
            if ridge is None and intensity > 0:
                # No compute ceiling available and the op does arithmetic: a
                # memory-side pass cannot rule out a compute-side impossibility.
                # Recorded before the residency check below, not after: a shape
                # that fits in L2 is equally unaudited on the compute side, and
                # skipping it here once let an all-resident arithmetic run report
                # NOT_APPLICABLE, which reads as "nothing to check" rather than
                # "nothing was checked".
                unbounded.append(label)
            if shape["working_set_bytes"] < threshold:
                resident.append(label)
                continue
            achieved = shape["bytes_moved"] / seconds
            fraction = achieved / peak_bw
            worst_candidate = (fraction, label, "memory", achieved / 1e9, peak_bw / 1e9)

        if worst is None or worst_candidate[0] > worst[0]:
            worst = worst_candidate

    if worst is None:
        resident_note = (
            f"every shape fits within {L2_CLEARANCE:g}x L2 ({l2 / 1e6:.0f} MB); "
            "DRAM bandwidth does not bound cache-resident traffic"
        )
        if unbounded:
            # Nothing was auditable on either axis: the bus does not bind
            # cache-resident traffic, and there is no compute ceiling to bind
            # instead. That is not the same as there being nothing to check.
            return GateResult(
                "roofline",
                GateStatus.UNVERIFIABLE_COMPUTE,
                f"{resident_note}, and {ceiling_note}, so neither ceiling could bound this run",
            )
        return GateResult("roofline", GateStatus.NOT_APPLICABLE, resident_note)

    fraction, label, bound, achieved_scaled, ceiling_scaled = worst
    units = "GB/s" if bound == "memory" else "TFLOP/s"
    ceiling_name = "memory bus" if bound == "memory" else f"{precision} pipelines"

    # The tabulated tensor ceiling is the consumer rate, and a professional part of
    # the same compute capability runs at twice it. So the number the gate reports
    # utilisation against and the number it will call impossible are not the same:
    # report against the best figure for this class of device, but only refuse a
    # claim that exceeds the highest rate any part of this capability is rated for.
    # The memory bus needs none of this -- bandwidth is read from the device.
    headroom = (
        TENSOR_SKU_SPREAD
        if bound == "compute" and precision in TENSOR_PRECISIONS
        else 1.0
    )

    if fraction > headroom:
        return GateResult(
            "roofline",
            GateStatus.FAIL,
            f"{label} implies {achieved_scaled:.1f} {units} against a "
            f"{ceiling_scaled * headroom:.1f} {units} {ceiling_name} ceiling "
            f"({fraction / headroom:.1f}x the hardware maximum) — physically impossible",
        )
    if headroom > 1.0 and fraction > 1.0:
        return GateResult(
            "roofline",
            GateStatus.UNVERIFIABLE_COMPUTE,
            f"{label} implies {achieved_scaled:.1f} {units}, above the {ceiling_scaled:.1f} "
            f"{units} {ceiling_name} ceiling for a consumer part but within the {headroom:g}x "
            f"spread NVIDIA ships across SKUs of compute capability "
            f"{measurement.get('compute_capability', '?')} without publishing the difference; "
            f"this cannot be called impossible on the figures available",
        )
    if headroom == 1.0 and fraction > IMPLAUSIBLE_FRACTION:
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
            f"but {ceiling_note}, so a compute-bound impossibility cannot be ruled out",
        )
    notes = []
    if resident:
        notes.append(f"{len(resident)} cache-resident shape(s) not auditable")
    # A throttled device never had access to the peak the ceiling assumes, so a
    # low utilisation may be the clock rather than the kernel. This cannot cause a
    # false rejection -- only high fractions fail -- but it can send a reader, or
    # an agent, chasing an optimisation that does not exist.
    observed = [
        s["sm_clock_hz_observed"]
        for s in measurement["shapes"]
        if s.get("sm_clock_hz_observed")
    ]
    max_clock = measurement.get("sm_clock_hz")
    if observed and max_clock:
        ratio = (sum(observed) / len(observed)) / float(max_clock)
        if ratio < 0.9:
            notes.append(
                f"device ran at {ratio:.0%} of peak clock, so this understates the kernel"
            )
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
        # Checked before the error, because the error is computed over the finite
        # elements only -- a NaN has no distance from anything, so it is skipped.
        # One true value in a NaN-filled output therefore satisfies both the
        # written check and the error check, and did.
        if shape.get("timed_has_nonfinite", False):
            return GateResult(
                "timed_work",
                GateStatus.FAIL,
                f"the measured calls at {label} produced NaN or Inf where the warmup did not; "
                f"the reported error covers only the elements that are finite",
            )
        err = shape.get("timed_violation")
        if err is None:
            return GateResult("timed_work", GateStatus.FAIL, f"{label} reported no post-timing error")
        if err / _violation_scale(measurement, shape) > 1.0:
            return GateResult(
                "timed_work",
                GateStatus.FAIL,
                f"output after timing is wrong at {label} "
                f"({_times(err / _violation_scale(measurement, shape))}x tolerance); the measured "
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
    "rows", "cols", "min_ms", "median_ms", "p90_ms", "p25_ms", "p75_ms", "max_ms",
    "max_abs_err", "max_rel_err", "has_nonfinite", "wrote_output",
    "input_sensitive", "inner_iters", "timed_output_written", "timed_has_nonfinite",
    "timed_max_rel_err", "rel_tol", "violation", "timed_violation",
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
