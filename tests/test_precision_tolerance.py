"""The declared precision must widen the correctness bar by the right shape.

`violation` is reported in units of the harness's own `rel_tol`, which is derived
from fp32 and grows as sqrt(depth). Converting a declared precision into that unit
is where two bugs have lived:

  - multiplying twice, which turned a bf16 tolerance into 800%
  - multiplying a depth-dependent base by a fixed factor for tf32, which reached
    0.5 at k=4096 -- about 128x looser than the contract, and looser the deeper the
    reduction, when TF32's error does not grow with depth at all

The second is pinned here against measured data. A Triton TF32 matmul on an RTX
4090 deviates by a flat ~1.5e-3 from k=512 to k=4096, while the harness's fp32
`rel_tol` grows 2.8x across the same range.
"""

from __future__ import annotations

import pytest

from preflight.gates import TF32_TOLERANCE, GateStatus, check_correctness, _violation_scale

# (k, harness rel_tol at that depth, violation measured for a real TF32 matmul)
MEASURED = [(512, 2.15e-5, 62.6429), (1024, 3.05e-5, 47.3670),
            (2048, 4.31e-5, 37.7355), (4096, 6.10e-5, 25.4302)]


def test_tf32_bar_is_flat_in_absolute_terms() -> None:
    """The whole point: one tolerance, whatever the reduction depth."""
    for _, rel_tol, _ in MEASURED:
        scale = _violation_scale({"precision": "tf32"}, {"rel_tol": rel_tol})
        assert scale * rel_tol == pytest.approx(TF32_TOLERANCE, rel=1e-9)


def test_tf32_scale_shrinks_with_depth() -> None:
    """It must shrink, because the fp32 base it is expressed against grows."""
    scales = [_violation_scale({"precision": "tf32"}, {"rel_tol": r}) for _, r, _ in MEASURED]
    assert scales == sorted(scales, reverse=True)
    assert scales[0] / scales[-1] == pytest.approx(2.84, rel=0.02)


def test_real_tf32_matmul_passes_with_margin_but_not_absurd_margin() -> None:
    """Measured TF32 must pass, and not by the 128x the old scale allowed."""
    for k, rel_tol, violation in MEASURED:
        normalised = violation / _violation_scale({"precision": "tf32"}, {"rel_tol": rel_tol})
        assert normalised < 1.0, f"k={k} would be failed at {normalised:.2f}x"
        assert normalised > 0.1, f"k={k} passes by {1 / normalised:.0f}x, which is not a bar"


def test_a_sloppier_tf32_kernel_is_rejected() -> None:
    """3x the measured deviation is outside the contract and must fail."""
    k, rel_tol, violation = MEASURED[-1]
    scale = _violation_scale({"precision": "tf32"}, {"rel_tol": rel_tol})
    assert (violation * 3) / scale > 1.0


def test_old_multiplicative_scale_would_have_waved_that_through() -> None:
    """Shows the bug mattered, rather than merely existed."""
    _, rel_tol, violation = MEASURED[-1]
    assert (violation * 3) / 8192.0 < 1.0
    # It would have taken a ~130x-worse kernel to trip the old bar.
    assert 8192.0 / _violation_scale({"precision": "tf32"}, {"rel_tol": rel_tol}) == pytest.approx(128.0, rel=0.01)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_other_precisions_are_not_scaled_here(precision: str) -> None:
    """The harness folds storage precision into rel_tol; scaling again double-counts."""
    assert _violation_scale({"precision": precision}, {"rel_tol": 6.1e-5}) == 1.0


def test_tf32_never_tightens_below_the_fp32_bar() -> None:
    """For a reduction deep enough that fp32 accumulation dominates, the wider bar
    is the honest one -- tightening below it would fail correct kernels."""
    huge = TF32_TOLERANCE * 10
    assert _violation_scale({"precision": "tf32"}, {"rel_tol": huge}) == 1.0


def test_zero_tolerance_ops_are_left_exact() -> None:
    """Transpose does no arithmetic and is expected to be bit-exact."""
    assert _violation_scale({"precision": "tf32"}, {"rel_tol": 0.0}) == 1.0


def _measurement(precision: str, rel_tol: float, violation: float) -> dict:
    return {
        "precision": precision,
        "shapes": [{
            "rows": 4096, "cols": 4096, "rel_tol": rel_tol, "violation": violation,
            "has_nonfinite": False, "max_abs_err": 1e-3,
        }],
    }


def test_correctness_gate_uses_the_flat_bar() -> None:
    _, rel_tol, violation = MEASURED[-1]
    assert check_correctness(_measurement("tf32", rel_tol, violation)).status is GateStatus.PASS
    worse = check_correctness(_measurement("tf32", rel_tol, violation * 3))
    assert worse.status is GateStatus.FAIL
    assert "tf32 tolerance" in worse.detail
