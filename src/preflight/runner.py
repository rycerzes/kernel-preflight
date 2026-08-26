"""Compile a candidate against the fixed harness, measure it, and gate it.

Two things here are load-bearing, and both exist because the earlier version of
this file was defeated by a candidate that never ran a kernel at all.

**Isolation.** The candidate is native code we are about to execute. It is
compiled and run inside a container with no network, no inherited environment and
a bounded lifetime -- never on the host. The first version ran it directly on the
GPU host, in the same process environment that holds the publishing token.

**Provenance.** The harness prints its measurement, but the candidate is linked
into the same binary, and C++ static constructors run before `main`. A candidate
could therefore print a fabricated measurement and `exit(0)` before the harness
executed a single line, and did: a kernel whose body was empty was admitted at a
plausible 91% of peak.

So the caller no longer trusts stdout on its face. It generates a nonce the
harness must echo, and it times the process itself. Fabricated work cannot
account for wall-clock time it never spent, and that inequality is checked.
"""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preflight.gates import Preflight, run_gates

HARNESS_DIR = Path(__file__).parent / "harness"
DRIVER = HARNESS_DIR / "driver.cu"
DEFAULT_ARCH = "sm_89"
# Operations the harness can measure. Each supplies its own double-precision host
# reference; the candidate signature is shared, and ops without a weight vector or
# epsilon simply ignore those arguments.
SUPPORTED_OPS = ("rmsnorm", "softmax", "silu", "transpose")
DEFAULT_IMAGE = "kernel-preflight-sandbox:cuda13"
COMPILE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 900
# Container guardrails. Generous enough for nvcc, tight enough that a runaway
# candidate cannot take the host with it.
CONTAINER_MEMORY = "8g"
CONTAINER_PIDS = "512"


class CompileError(RuntimeError):
    """The candidate did not compile. Its diagnostics are the message."""


class HarnessError(RuntimeError):
    """The harness failed to produce a trustworthy measurement."""


class IsolationError(RuntimeError):
    """The container runtime needed to execute untrusted code is unavailable."""


@dataclass(frozen=True)
class PreflightReport:
    preflight: Preflight
    compile_log: str

    @property
    def admitted(self) -> bool:
        return self.preflight.admitted

    def to_dict(self) -> dict[str, Any]:
        payload = self.preflight.to_dict()
        payload["measurement"] = self.preflight.measurement
        return payload


def _docker() -> str:
    found = shutil.which("docker")
    if not found:
        raise IsolationError(
            "docker not found; refusing to compile or run candidate code on the host"
        )
    return found


def _run_in_container(
    *,
    workdir: Path,
    image: str,
    argv: list[str],
    timeout_s: int,
    gpus: str | None,
) -> subprocess.CompletedProcess[str]:
    docker_args = [
        _docker(),
        "run",
        "--rm",
        # No network: a candidate cannot phone home with anything it finds.
        "--network",
        "none",
        # No inherited environment. The publishing token lives in this server's
        # environment and must not cross into code we did not write.
        "--env-file",
        "/dev/null",
        "--memory",
        CONTAINER_MEMORY,
        "--pids-limit",
        CONTAINER_PIDS,
        "--workdir",
        "/work",
        "-v",
        f"{workdir}:/work",
    ]
    if gpus:
        docker_args += ["--gpus", gpus]
    docker_args += [image, *argv]
    return subprocess.run(docker_args, capture_output=True, text=True, timeout=timeout_s)


def preflight_source(
    candidate_source: str,
    *,
    op: str = "rmsnorm",
    arch: str = DEFAULT_ARCH,
    repeats: int = 30,
    seed: int = 20260826,
    image: str = DEFAULT_IMAGE,
    gpus: str | None = "all",
) -> PreflightReport:
    """Run the full preflight on one candidate kernel source, in isolation.

    `candidate_source` must define
    `extern "C" void launch_candidate(const float*, const float*, float*, int, int, float, cudaStream_t)`.
    """
    if op not in SUPPORTED_OPS:
        raise CompileError(f"unknown op {op!r}; supported: {', '.join(SUPPORTED_OPS)}")

    with tempfile.TemporaryDirectory(prefix="kernel-preflight-") as tmp:
        workdir = Path(tmp)
        # World-readable so the container user can read regardless of uid mapping.
        workdir.chmod(0o777)
        (workdir / "candidate.cu").write_text(candidate_source)
        (workdir / "driver.cu").write_text(DRIVER.read_text())

        try:
            compile_proc = _run_in_container(
                workdir=workdir,
                image=image,
                # -Werror=format: a printf format/argument mismatch in the harness silently
                # corrupts the measurement JSON rather than failing, which has bitten
                # this file twice. Make it a build error.
                argv=[
                    "nvcc", "-O3", f"-arch={arch}",
                    "-Xcompiler", "-Wformat", "-Xcompiler", "-Werror=format",
                    "driver.cu", "candidate.cu", "-o", "preflight",
                ],
                timeout_s=COMPILE_TIMEOUT_S,
                gpus=None,  # compilation needs no device
            )
        except subprocess.TimeoutExpired as exc:
            raise CompileError(f"compilation exceeded {COMPILE_TIMEOUT_S}s") from exc
        if compile_proc.returncode != 0:
            raise CompileError(compile_proc.stderr.strip() or "nvcc failed with no diagnostics")

        nonce = secrets.token_hex(16)
        started = time.monotonic()
        try:
            run_proc = _run_in_container(
                workdir=workdir,
                image=image,
                argv=["./preflight", op, str(repeats), str(seed), nonce],
                timeout_s=RUN_TIMEOUT_S,
                gpus=gpus,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(f"measurement exceeded {RUN_TIMEOUT_S}s") from exc
        observed_wall_s = time.monotonic() - started

        if run_proc.returncode != 0:
            raise HarnessError(
                f"harness exited {run_proc.returncode}: {(run_proc.stderr or run_proc.stdout).strip()[:500]}"
            )
        try:
            measurement = json.loads(run_proc.stdout)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"harness did not emit JSON: {run_proc.stdout[:300]}") from exc
        if "error" in measurement:
            raise HarnessError(str(measurement["error"]))

        # Provenance facts, recorded for the gate rather than enforced here, so a
        # rejection reads as a verdict with a reason instead of an exception.
        measurement["_provenance"] = {
            "expected_nonce": nonce,
            "observed_wall_s": observed_wall_s,
            "repeats": repeats,
        }

        return PreflightReport(preflight=run_gates(measurement), compile_log=compile_proc.stderr.strip())


def preflight_file(path: str | Path, **kwargs: Any) -> PreflightReport:
    return preflight_source(Path(path).read_text(), **kwargs)
