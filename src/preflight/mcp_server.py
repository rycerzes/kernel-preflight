"""MCP server exposing the preflight gates as tools.

Deliberately separate from the agent's sandbox. The sandbox is the agent's
workspace -- it drafts, compiles and experiments there, under its own control.
This server is the adjudicator, and the agent has no control over it at all: it
submits kernel *source* and receives a verdict. It never submits a number.

That separation is the whole design. A verifier the subject can influence is not
a verifier.

Transport is streamable HTTP bound to localhost, with a bearer token. It compiles
and runs model-authored CUDA, so it must never be exposed off the host.
"""

from __future__ import annotations

import os
import secrets
import subprocess
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from preflight.device import CudaDriverError, probe
from preflight.runner import CompileError, HarnessError, IsolationError, preflight_source

server = MCPServer(
    name="kernel-preflight",
    instructions=(
        "Preflight checks for GPU kernel performance claims. Submit kernel source; "
        "the harness measures it. You cannot report timings yourself -- every number "
        "in a verdict was produced by code you do not control."
    ),
)


@server.tool(
    annotations={"readOnlyHint": True},
    description="Physical limits of the GPU: memory bandwidth and FP32 ceilings, read from the CUDA driver.",
)
def device_spec() -> dict[str, Any]:
    try:
        return probe().to_dict()
    except CudaDriverError as exc:
        return {"error": str(exc)}


@server.tool(
    annotations={"readOnlyHint": True},
    description=(
        "Compile a candidate RMSNorm kernel against the fixed harness, measure it across "
        "five shapes, and run every preflight gate. The candidate must define "
        'extern \"C\" void launch_candidate(const float* x, const float* w, float* y, '
        "int rows, int cols, float eps, cudaStream_t stream). Returns per-gate verdicts "
        "and the raw measurement."
    ),
)
def preflight_kernel(
    candidate_source: Annotated[str, Field(description="Full CUDA source defining launch_candidate.")],
    arch: Annotated[str, Field(description="Target architecture, e.g. sm_89.")] = "sm_89",
    repeats: Annotated[int, Field(ge=5, le=200, description="Timed repeats per shape.")] = 30,
) -> dict[str, Any]:
    try:
        report = preflight_source(candidate_source, arch=arch, repeats=repeats)
    except CompileError as exc:
        # Timeouts are folded into these by the runner, so a slow compile or a
        # hanging candidate is a rejected verdict rather than a server error.
        return {"admitted": False, "stage": "compile", "error": str(exc)[:4000]}
    except HarnessError as exc:
        return {"admitted": False, "stage": "measure", "error": str(exc)[:4000]}
    except IsolationError as exc:
        return {"admitted": False, "stage": "isolation", "error": str(exc)[:4000]}
    payload = report.to_dict()
    payload["stage"] = "gates"
    payload["summary"] = report.preflight.summary()
    return payload


@server.tool(
    annotations={"readOnlyHint": False, "destructiveHint": True},
    description=(
        "Publish an admitted kernel to the Hugging Face Hub. This creates a permanent "
        "public artefact under your namespace and cannot be undone. Only call this after "
        "preflight_kernel has admitted the kernel."
    ),
)
def publish_kernel(
    repo_id: Annotated[str, Field(description="Target repo, e.g. rycerzes/rmsnorm-sm89.")],
    candidate_source: Annotated[str, Field(description="The kernel source to publish.")],
    verdict_summary: Annotated[str, Field(description="The preflight summary to publish alongside it.")],
) -> dict[str, Any]:
    # Re-verify rather than trust the caller's word that it passed. The whole
    # premise is that the agent's claims are not evidence, and "I already checked"
    # is a claim.
    try:
        report = preflight_source(candidate_source)
    except (CompileError, HarnessError, IsolationError) as exc:
        return {"published": False, "reason": f"re-verification failed: {exc}"[:2000]}
    if not report.admitted:
        return {
            "published": False,
            "reason": "kernel does not pass preflight; refusing to publish",
            "summary": report.preflight.summary(),
        }

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"published": False, "reason": "HF_TOKEN is not set on the preflight server"}

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"published": False, "reason": "huggingface_hub is not installed on the preflight server"}

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    workdir = Path("/tmp") / f"kernel-preflight-publish-{secrets.token_hex(4)}"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "rmsnorm.cu").write_text(candidate_source)
    (workdir / "PREFLIGHT.md").write_text(
        "# Preflight report\n\n"
        "Measured by a harness the kernel author did not control.\n\n"
        f"```\n{report.preflight.summary()}\n```\n\n"
        f"## Agent-supplied summary\n\n{verdict_summary}\n"
    )
    api.upload_folder(repo_id=repo_id, folder_path=str(workdir), repo_type="model")
    return {
        "published": True,
        "url": f"https://huggingface.co/{repo_id}",
        "summary": report.preflight.summary(),
    }


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    import uvicorn

    app = server.streamable_http_app()
    port = int(os.environ.get("PREFLIGHT_MCP_PORT", "8791"))
    # Localhost only: this endpoint compiles and executes model-authored CUDA.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
