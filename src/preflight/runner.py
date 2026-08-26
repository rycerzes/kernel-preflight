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
import os
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
CUDA_DRIVER = HARNESS_DIR / "driver.cu"
PYTHON_DRIVER = HARNESS_DIR / "driver.py"
DEFAULT_ARCH = "sm_89"
# Operations the harness can measure. Each supplies its own double-precision host
# reference; the candidate signature is shared, and ops without a weight vector or
# epsilon simply ignore those arguments.
# Which harness runs an op, and therefore which image and candidate language.
# `cuda` compiles a .cu against driver.cu; the Python backends import a .py
# against driver.py. Both emit the same measurement schema, so the gates do not
# know or care which produced a verdict.
CUDA_OPS = ("rmsnorm", "softmax", "silu", "transpose")
PYTHON_OPS = ("rmsnorm", "softmax", "silu", "transpose", "matmul", "attention")
PYTHON_BACKENDS = ("triton", "helion", "torch")
SUPPORTED_BACKENDS = ("cuda", *PYTHON_BACKENDS)
CUDA_IMAGE = "kernel-preflight-sandbox:cuda13"
# Separate image: the Python backends need torch, Triton and Helion and never
# invoke nvcc, so a CUDA-only preflight should not pay for a multi-gigabyte
# torch install.
PYTHON_IMAGE = "kernel-preflight-sandbox:torch"
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
        # Run as the invoking user, not root. Otherwise every artefact the
        # container writes into the bind mount (Python bytecode, Triton's JIT
        # cache) is root-owned and the host cannot clean up its own temp
        # directory. Dropping root is also simply correct for code we did not
        # write.
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        # A non-root user has no home in the image, and both Triton and torch
        # want somewhere to cache. Point them inside the bind mount so the
        # artefacts are owned correctly and vanish with the temp directory.
        "--env",
        "HOME=/work",
        "--env",
        "TRITON_CACHE_DIR=/work/.triton",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/work/.inductor",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if gpus:
        docker_args += ["--gpus", gpus]
    docker_args += [image, *argv]
    return subprocess.run(docker_args, capture_output=True, text=True, timeout=timeout_s)


def preflight_source(
    candidate_source: str,
    *,
    op: str = "rmsnorm",
    backend: str = "cuda",
    arch: str = DEFAULT_ARCH,
    repeats: int = 30,
    seed: int = 20260826,
    image: str | None = None,
    gpus: str | None = "all",
) -> PreflightReport:
    """Run the full preflight on one candidate kernel source, in isolation.

    `candidate_source` must define
    `extern "C" void launch_candidate(const float*, const float*, float*, int, int, float, cudaStream_t)`.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise CompileError(f"unknown backend {backend!r}; supported: {', '.join(SUPPORTED_BACKENDS)}")
    allowed = CUDA_OPS if backend == "cuda" else PYTHON_OPS
    if op not in allowed:
        raise CompileError(f"backend {backend!r} does not support op {op!r}; supported: {', '.join(allowed)}")
    resolved_image = image or (CUDA_IMAGE if backend == "cuda" else PYTHON_IMAGE)

    with tempfile.TemporaryDirectory(prefix="kernel-preflight-") as tmp:
        workdir = Path(tmp)
        # World-readable so the container user can read regardless of uid mapping.
        workdir.chmod(0o777)

        compile_log = ""
        if backend == "cuda":
            (workdir / "candidate.cu").write_text(candidate_source)
            (workdir / "driver.cu").write_text(CUDA_DRIVER.read_text())
            try:
                compile_proc = _run_in_container(
                    workdir=workdir,
                    image=resolved_image,
                    # -Werror=format: a printf format/argument mismatch in the harness
                    # silently corrupts the measurement JSON rather than failing, which
                    # has bitten this file twice. Make it a build error.
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
            compile_log = compile_proc.stderr.strip()
            nonce = secrets.token_hex(16)
            run_argv = ["./preflight", op, str(repeats), str(seed), nonce]
        else:
            # Python backends JIT at run time, so there is no separate compile
            # step; a syntax error surfaces as a harness failure instead.
            (workdir / "candidate.py").write_text(candidate_source)
            (workdir / "driver.py").write_text(PYTHON_DRIVER.read_text())
            nonce = secrets.token_hex(16)
            run_argv = [
                "python3", "driver.py",
                "--op", op,
                "--candidate", "candidate.py",
                "--repeats", str(repeats),
                "--seed", str(seed),
                "--nonce", nonce,
            ]

        started = time.monotonic()
        try:
            run_proc = _run_in_container(
                workdir=workdir,
                image=resolved_image,
                argv=run_argv,
                timeout_s=RUN_TIMEOUT_S,
                gpus=gpus,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(f"measurement exceeded {RUN_TIMEOUT_S}s") from exc
        observed_wall_s = time.monotonic() - started

        if run_proc.returncode != 0:
            raise HarnessError(
                f"harness exited {run_proc.returncode}: {(run_proc.stderr or run_proc.stdout).strip()[:800]}"
            )
        try:
            measurement = json.loads(run_proc.stdout)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"harness did not emit JSON: {run_proc.stdout[:300]}") from exc
        if "error" in measurement:
            raise HarnessError(str(measurement["error"]))

        measurement["backend"] = backend
        # Provenance facts, recorded for the gate rather than enforced here, so a
        # rejection reads as a verdict with a reason instead of an exception.
        measurement["_provenance"] = {
            "expected_nonce": nonce,
            "observed_wall_s": observed_wall_s,
            "repeats": repeats,
        }

        return PreflightReport(preflight=run_gates(measurement), compile_log=compile_log)


def preflight_file(path: str | Path, **kwargs: Any) -> PreflightReport:
    """Preflight a candidate from disk. Backend defaults from the file suffix."""
    source = Path(path).read_text()
    if "backend" not in kwargs:
        kwargs["backend"] = "cuda" if Path(path).suffix == ".cu" else "torch"
    return preflight_source(source, **kwargs)
