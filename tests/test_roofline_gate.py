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
