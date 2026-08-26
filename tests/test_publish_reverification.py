"""The publication path must re-verify under the contract it is publishing.

`publish_kernel` re-runs the whole preflight rather than trusting the caller's
word that the kernel passed. That is only worth anything if the re-run uses the
*same* operation, toolchain and precision the kernel was admitted for. It did not:
every parameter defaulted, so re-verification asked "is this an fp32 CUDA RMSNorm?"
regardless of what had been measured.

Two distinct failures came out of that one default:

  - a softmax kernel was checked against the RMSNorm reference and refused
  - no Triton, Helion, CuTe or TileLang kernel could be published at all, because
    its Python source was handed to nvcc

Neither is exploitable -- both refuse to publish -- but a verification step that
rejects correct work is the failure mode this project keeps having to fix, and a
publication path nothing can pass is not a publication path.

These tests pin the forwarding without touching the network or a GPU: the
re-verification call is captured, and the run stops at the missing HF_TOKEN.
"""

from __future__ import annotations

import pytest

from preflight import mcp_server


@pytest.fixture
def captured(monkeypatch):
    """Capture the kwargs publish_kernel forwards to re-verification."""
    calls: list[dict] = []

    class _Rejected:
        admitted = False

        class preflight:
            @staticmethod
            def summary() -> str:
                return "rejected by stub"

    def fake_preflight_source(source, **kwargs):
        calls.append(kwargs)
        return _Rejected()

    monkeypatch.setattr(mcp_server, "preflight_source", fake_preflight_source)
    return calls


def _publish(**kwargs):
    return mcp_server.publish_kernel(
        repo_id="example-user/example-kernel",
        candidate_source="# candidate",
        verdict_summary="admitted",
        **kwargs,
    )


def test_declared_op_reaches_reverification(captured) -> None:
    _publish(op="softmax")
    assert captured[0]["op"] == "softmax", (
        "a softmax kernel must be re-verified against the softmax reference"
    )


def test_declared_backend_reaches_reverification(captured) -> None:
    _publish(op="rmsnorm", backend="triton")
    assert captured[0]["backend"] == "triton", (
        "a Triton kernel's Python source must not be handed to nvcc"
    )


def test_declared_precision_reaches_reverification(captured) -> None:
    _publish(op="attention", backend="triton", precision="bf16")
    assert captured[0]["precision"] == "bf16"
    assert captured[0]["op"] == "attention"


def test_defaults_are_cuda_fp32_rmsnorm(captured) -> None:
    # The old behaviour, now only reachable by asking for it explicitly.
    _publish()
    assert captured[0] == {
        "op": "rmsnorm",
        "backend": "cuda",
        "precision": "fp32",
        "arch": "sm_89",
    }


def test_rejected_kernel_is_not_published(captured) -> None:
    result = _publish(op="softmax")
    assert result["published"] is False
    assert "does not pass preflight" in result["reason"]


def test_unsupported_declaration_is_refused_not_published(monkeypatch) -> None:
    """A declaration the runner rejects must surface as a refusal, not a crash."""

    def fake_preflight_source(source, **kwargs):
        raise mcp_server.CompileError(f"backend {kwargs['backend']!r} does not support op")

    monkeypatch.setattr(mcp_server, "preflight_source", fake_preflight_source)
    result = _publish(op="matmul", backend="cuda")
    assert result["published"] is False
    assert "re-verification failed" in result["reason"]
