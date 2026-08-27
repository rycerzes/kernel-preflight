"""Tests for the roofline gate's ceiling selection.

Every operation the harness currently measures is memory-bound, so the
compute-bound branch had never executed against real data. These construct
measurements directly to exercise both sides of the ridge point, including the
cases where the gate must refuse to give an answer.

Numbers are RTX 4090: 1008.1 GB/s of memory bandwidth, 83.1 TFLOP/s of FP32,
ridge point ~82.4 FLOP/byte, 72 MB of L2.
"""

from __future__ import annotations

import pytest

from preflight.gates import GateStatus, check_roofline

PEAK_BW = 1008.1e9
PEAK_FLOPS = 83.1e12
L2 = 75_497_472
# Comfortably past 2x L2 so the memory side is auditable.
WORKING_SET = 512 * 1024 * 1024


def measurement(*, flops: float, median_ms: float, bytes_moved: float = WORKING_SET,
                peak_flops: float = PEAK_FLOPS) -> dict:
    return {
        "peak_bandwidth_bytes_per_s": PEAK_BW,
        "peak_fp32_flops": peak_flops,
        # sm_89 with 128 SMs at 2.52 GHz: 128 * 256 * 2.52e9 = 82.6 TFLOP/s fp32.
        "compute_capability": "8.9" if peak_flops else "99.0",
        "sm_count": 128,
        "sm_clock_hz": 2.52e9,
        "precision": "fp32",
        "l2_cache_bytes": L2,
        "repeats": 30,
        "shapes": [
            {
                "rows": 16384,
                "cols": 4096,
                "min_ms": median_ms,
                "median_ms": median_ms,
                "p90_ms": median_ms,
                "p25_ms": median_ms,
                "p75_ms": median_ms,
                "outliers": 0,
                "max_ms": median_ms,
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
                "has_nonfinite": False,
                "wrote_output": True,
                "input_sensitive": True,
                "timed_output_written": True,
                "timed_has_nonfinite": False,
                "timed_max_rel_err": 0.0,
                "inner_iters": 1,
                "rel_tol": 1e-5,
                "violation": 0.0,
                "timed_violation": 0.0,
                "bytes_moved": bytes_moved,
                "flops": flops,
                "working_set_bytes": WORKING_SET,
            }
        ],
    }


def test_low_intensity_is_audited_against_memory() -> None:
    # 1 FLOP/byte: far below the ridge, so the bus binds.
    result = check_roofline(measurement(flops=WORKING_SET, median_ms=0.6))
    assert result.status is GateStatus.PASS
    assert "memory bus" in result.detail


def test_high_intensity_is_audited_against_compute() -> None:
    # 1000 FLOP/byte is well past the ~82 ridge, so the SMs bind. Time chosen so
    # the claim lands near half the FP32 ceiling.
    flops = 1000 * WORKING_SET
    seconds = flops / (PEAK_FLOPS * 0.5)
    result = check_roofline(measurement(flops=flops, median_ms=seconds * 1000))
    assert result.status is GateStatus.PASS
    assert "fp32 pipelines" in result.detail, result.detail


def test_impossible_compute_claim_is_rejected() -> None:
    """The case that would have passed before: a compute-bound kernel claiming
    more FLOPs than the SMs can retire, audited against DRAM and waved through."""
    flops = 1000 * WORKING_SET
    seconds = flops / (PEAK_FLOPS * 3.0)  # 3x the FP32 ceiling
    result = check_roofline(measurement(flops=flops, median_ms=seconds * 1000))
    assert result.status is GateStatus.FAIL
    assert "fp32 pipelines" in result.detail
    assert "physically impossible" in result.detail


def test_compute_bound_claim_would_pass_a_memory_only_gate() -> None:
    """Shows why the ceiling choice matters, not just that it happens.

    The same impossible compute claim moves far less data than the bus allows, so
    a memory-only audit sees a comfortable utilisation and admits it.
    """
    flops = 1000 * WORKING_SET
    seconds = flops / (PEAK_FLOPS * 3.0)
    achieved_bw = WORKING_SET / seconds
    assert achieved_bw < PEAK_BW, "memory side must look fine for the test to mean anything"


def test_unknown_compute_ceiling_is_not_a_pass() -> None:
    # An op that does arithmetic on a device whose FP32 ceiling we cannot compute:
    # the memory side may be fine and still hide a compute-side impossibility.
    result = check_roofline(measurement(flops=WORKING_SET * 10, median_ms=0.6, peak_flops=0.0))
    assert result.status is GateStatus.UNVERIFIABLE_COMPUTE
    assert not result.blocking


def test_zero_flop_op_needs_no_compute_ceiling() -> None:
    # Transpose does no arithmetic, so a missing FP32 ceiling costs nothing.
    result = check_roofline(measurement(flops=0.0, median_ms=0.6, peak_flops=0.0))
    assert result.status is GateStatus.PASS


def test_cache_resident_shapes_are_not_audited_on_memory() -> None:
    m = measurement(flops=WORKING_SET, median_ms=0.6)
    m["shapes"][0]["working_set_bytes"] = 8 * 1024 * 1024  # inside L2
    result = check_roofline(m)
    assert result.status is GateStatus.NOT_APPLICABLE
    assert "L2" in result.detail


@pytest.mark.parametrize("median_ms", [0.0, -1.0])
def test_non_positive_time_is_rejected(median_ms: float) -> None:
    result = check_roofline(measurement(flops=WORKING_SET, median_ms=median_ms))
    assert result.status is GateStatus.FAIL


# ---------------------------------------------------------------------------
# Tensor-core ceilings and the SKU spread
# ---------------------------------------------------------------------------
#
# Within one compute capability NVIDIA ships parts whose tensor throughput differs
# by exactly 2x -- RTX 3090 at 35.6 dense TF32 TFLOPS against A40 at 74.8, both
# compute capability 8.6 -- and states that the reason is unpublished. A ceiling
# keyed on capability alone therefore cannot be a limit for every part reporting it,
# and one that is too low does not weaken the gate, it makes it accuse correct work.


from preflight.gates import TENSOR_SKU_SPREAD  # noqa: E402


def tensor_measurement(*, precision: str, flops: float, median_ms: float,
                       per_sm_clock: int) -> dict:
    """A compute-bound claim at a tensor precision on an sm_86-class part."""
    m = measurement(flops=flops, median_ms=median_ms,
                    peak_flops=128 * 1.7e9 * per_sm_clock)
    m["precision"] = precision
    m["compute_capability"] = "8.6"
    m["sm_count"] = 128
    m["sm_clock_hz"] = 1.7e9
    return m


def _tf32_claim(multiple_of_tabulated: float) -> dict:
    """A TF32 claim at a chosen multiple of the tabulated (consumer) ceiling."""
    per_clock = 256
    ceiling = 128 * 1.7e9 * per_clock
    flops = 1000 * WORKING_SET  # far above the ridge, so the SMs bind
    seconds = flops / (ceiling * multiple_of_tabulated)
    return tensor_measurement(precision="tf32", flops=flops,
                              median_ms=seconds * 1000, per_sm_clock=per_clock)


def test_tensor_claim_within_the_tabulated_ceiling_passes() -> None:
    result = check_roofline(_tf32_claim(0.8))
    assert result.status is GateStatus.PASS
    assert "tf32 pipelines" in result.detail


def test_professional_sku_rate_is_not_called_impossible() -> None:
    """The regression: 1.8x the consumer rate is a real A40/A6000 measurement."""
    result = check_roofline(_tf32_claim(1.8))
    assert result.status is not GateStatus.FAIL, result.detail
    assert result.status is GateStatus.UNVERIFIABLE_COMPUTE
    assert not result.blocking
    assert "without publishing the difference" in result.detail


def test_beyond_the_widest_published_rate_is_still_impossible() -> None:
    result = check_roofline(_tf32_claim(TENSOR_SKU_SPREAD * 1.5))
    assert result.status is GateStatus.FAIL
    assert "physically impossible" in result.detail


def test_impossible_multiple_is_reported_against_the_widest_ceiling() -> None:
    """Saying "3x the maximum" when it is 3x a rate some parts double is wrong."""
    result = check_roofline(_tf32_claim(6.0))
    assert result.status is GateStatus.FAIL
    assert "3.0x the hardware maximum" in result.detail


def test_fp32_keeps_a_hard_ceiling() -> None:
    """fp32 runs on CUDA cores at 2 FLOP/core/clock on every part of a capability,
    so it gets no headroom -- this is the axis that caught the undeclared bf16."""
    per_clock = 256
    ceiling = 128 * 1.7e9 * per_clock
    flops = 1000 * WORKING_SET
    seconds = flops / (ceiling * 1.2)
    m = tensor_measurement(precision="fp32", flops=flops,
                           median_ms=seconds * 1000, per_sm_clock=per_clock)
    result = check_roofline(m)
    assert result.status is GateStatus.FAIL
    assert "1.2x the hardware maximum" in result.detail


def test_implausibly_high_is_not_flagged_at_tensor_precisions() -> None:
    """94% of a ceiling some parts double says nothing about the measurement."""
    assert check_roofline(_tf32_claim(0.97)).status is GateStatus.PASS


def test_implausibly_high_is_still_flagged_at_fp32() -> None:
    per_clock = 256
    ceiling = 128 * 1.7e9 * per_clock
    flops = 1000 * WORKING_SET
    seconds = flops / (ceiling * 0.97)
    m = tensor_measurement(precision="fp32", flops=flops,
                           median_ms=seconds * 1000, per_sm_clock=per_clock)
    result = check_roofline(m)
    assert result.status is GateStatus.FAIL
    assert "points at the measurement" in result.detail


def test_all_resident_arithmetic_without_a_ceiling_is_unverifiable() -> None:
    """Not NOT_APPLICABLE: nothing was checked, which is not nothing to check."""
    m = measurement(flops=WORKING_SET * 1000, median_ms=0.6, peak_flops=0.0)
    m["shapes"][0]["working_set_bytes"] = 8 * 1024 * 1024  # inside L2
    result = check_roofline(m)
    assert result.status is GateStatus.UNVERIFIABLE_COMPUTE
    assert not result.blocking


def test_all_resident_without_arithmetic_is_not_applicable() -> None:
    """Transpose does no arithmetic, so a missing compute ceiling costs nothing."""
    m = measurement(flops=0.0, median_ms=0.6, peak_flops=0.0)
    m["shapes"][0]["working_set_bytes"] = 8 * 1024 * 1024
    assert check_roofline(m).status is GateStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Variance: a failure has to say whether to re-run or to rewrite
# ---------------------------------------------------------------------------

from preflight.gates import check_variance  # noqa: E402


def _unstable(*, observed_clock: float | None, peak_clock: float = 2.52e9) -> dict:
    m = measurement(flops=WORKING_SET, median_ms=1.0)
    m["sm_clock_hz"] = peak_clock
    shape = m["shapes"][0]
    # A whole quartile 2x slower than another quartile.
    shape["p25_ms"], shape["p75_ms"] = 1.0, 2.0
    shape["sm_clock_hz_observed"] = observed_clock
    return m


def test_variance_failure_blames_the_device_when_the_clock_was_low() -> None:
    result = check_variance(_unstable(observed_clock=1.0e9))
    assert result.status is GateStatus.FAIL
    assert "re-run on an idle device" in result.detail
    assert "40% of peak clock" in result.detail


def test_variance_failure_blames_the_kernel_when_the_clock_held() -> None:
    result = check_variance(_unstable(observed_clock=2.5e9))
    assert result.status is GateStatus.FAIL
    assert "the kernel rather than the machine" in result.detail


def test_variance_failure_says_nothing_extra_without_clock_samples() -> None:
    """A CUDA-harness measurement carries no per-shape clock; do not invent one."""
    result = check_variance(_unstable(observed_clock=None))
    assert result.status is GateStatus.FAIL
    assert "clock" not in result.detail


def test_stable_timing_still_passes() -> None:
    m = _unstable(observed_clock=2.5e9)
    m["shapes"][0]["p25_ms"], m["shapes"][0]["p75_ms"] = 1.0, 1.05
    assert check_variance(m).status is GateStatus.PASS
