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
# Shared by both backends: the one process in the container that never runs
# candidate code, and therefore the only one trusted to write a verdict.
SUPERVISOR = HARNESS_DIR / "supervisor.py"
# Reaping a timed-out container should not itself be able to hang.
CONTAINER_KILL_TIMEOUT_S = 30
DEFAULT_ARCH = "sm_89"
# Operations the harness can measure. Each supplies its own double-precision host
# reference; the candidate signature is shared, and ops without a weight vector or
# epsilon simply ignore those arguments.
# Which harness runs an op, and therefore which image and candidate language.
# `cuda` compiles a .cu against driver.cu; the Python backends import a .py
# against driver.py. Both emit the same measurement schema, so the gates do not
# know or care which produced a verdict.
CUDA_OPS = ("rmsnorm", "softmax", "silu", "transpose")
PYTHON_OPS = (
    "rmsnorm", "softmax", "silu", "transpose", "matmul", "attention",
    # The two categories KernelBenchX finds hardest: Fusion (72% failure) and
    # Quantization (0 of 30 solved), plus LayerNorm for its two-stage reduction.
    "layernorm", "swiglu", "quantize",
    # Index and Loss from the same taxonomy, plus RoPE because essentially every
    # deployed transformer runs it and fusing it with attention is where the
    # inference engines find their wins.
    "rope", "gather", "cross_entropy",
    # Causal changes the FLOP count; decode moves attention to the other side of the
    # ridge point, so the same schema gets audited against a different ceiling.
    "attention_causal", "attention_decode", "attention_gqa", "moe_gemm",
    "attention_paged", "attention_backward",
)
PYTHON_BACKENDS = ("triton", "helion", "torch", "cute", "tilelang")
SUPPORTED_BACKENDS = ("cuda", *PYTHON_BACKENDS)
# The numerical contract a candidate claims. Declaring a reduced precision widens
# the correctness tolerance and changes which hardware ceiling binds; an
# undeclared downgrade is a correctness failure, because the speed was bought with
# accuracy the caller did not agree to.
SUPPORTED_PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
# Not every backend can honour every contract. driver.cu allocates float buffers,
# hands the candidate float pointers, and charges traffic at 4 bytes an element, so
# it cannot offer bf16 or fp16 storage -- accepting them let a CUDA candidate claim
# a reduced precision, receive the widened tolerance that goes with it, and be
# measured as fp32 anyway. tf32 is fine there: it is a compute mode over fp32
# storage, so the buffers are already the right shape and only the metadata differs.
PRECISIONS_BY_BACKEND = {"cuda": ("fp32", "tf32")}
CUDA_IMAGE = "kernel-preflight-sandbox:cuda13"
# Separate image: the Python backends need torch, Triton and Helion and never
# invoke nvcc, so a CUDA-only preflight should not pay for a multi-gigabyte
# torch install.
PYTHON_IMAGE = "kernel-preflight-sandbox:torch"
COMPILE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 900
# Python backends JIT and, for Helion, autotune per shape, so a full sweep takes
# much longer than a CUDA compile-and-run.
PYTHON_RUN_TIMEOUT_S = 2400
# Helion autotunes each shape separately and will otherwise search until it is
# satisfied. Bounded so a sweep terminates; the search happens during warmup, so
# it never enters a timed sample.
HELION_AUTOTUNE_BUDGET_S = 30
# The harness writes its measurement here, inside the bind mount, rather than to
# stdout. stdout is shared with every library the candidate imports -- TileLang
# logs there unconditionally -- and it is also the channel a candidate could print
# a forged measurement on.
MEASUREMENT_FILE = "measurement.json"
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
    stdin_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    # Named so a timeout has something to kill. `--rm` only removes a container
    # after it exits, and killing the `docker run` client does not stop the
    # container it started -- a hanging candidate would otherwise keep the GPU
    # busy after the request that started it had already given up.
    container = f"kernel-preflight-{secrets.token_hex(6)}"
    docker_args = [
        _docker(),
        "run",
        "--rm",
        "--name",
        container,
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
        "--env",
        f"HELION_AUTOTUNE_BUDGET_SECONDS={HELION_AUTOTUNE_BUDGET_S}",
        # The search tries configs that do not compile, or compile slowly. Without
        # this a single unlucky candidate config aborts the whole preflight, which
        # is a property of the tuner rather than of the kernel under test.
        "--env",
        "HELION_AUTOTUNE_IGNORE_ERRORS=1",
        "--env",
        "HELION_AUTOTUNE_COMPILE_TIMEOUT=180",
    ]
    if stdin_data is not None:
        # -i so the harness supervisor can be handed the run's secrets on stdin.
        # They cannot travel in argv or the environment: a candidate runs inside
        # this container and /proc/<ppid>/cmdline and /proc/<ppid>/environ are both
        # readable from there, so anything placed in either is readable by the code
        # being measured.
        docker_args.append("-i")
    if gpus:
        docker_args += ["--gpus", gpus]
    docker_args += [image, *argv]
    try:
        return subprocess.run(
            docker_args, capture_output=True, text=True, timeout=timeout_s, input=stdin_data,
        )
    except subprocess.TimeoutExpired:
        _kill_container(container)
        raise


def _kill_container(name: str) -> None:
    """Stop a container we started, on the way out of a timeout.

    Best effort by design: the container may have exited on its own between the
    timeout firing and this call, and a failure to reap it must not replace the
    timeout the caller is already handling with a less informative error.
    """
    try:
        subprocess.run(
            [_docker(), "kill", name],
            capture_output=True,
            text=True,
            timeout=CONTAINER_KILL_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def preflight_source(
    candidate_source: str,
    *,
    op: str = "rmsnorm",
    backend: str = "cuda",
    precision: str = "fp32",
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
    if precision not in SUPPORTED_PRECISIONS:
        raise CompileError(
            f"unknown precision {precision!r}; supported: {', '.join(SUPPORTED_PRECISIONS)}"
        )
    allowed_precisions = PRECISIONS_BY_BACKEND.get(backend, SUPPORTED_PRECISIONS)
    if precision not in allowed_precisions:
        raise CompileError(
            f"backend {backend!r} cannot honour precision {precision!r}; it supports "
            f"{', '.join(allowed_precisions)}. Reduced-storage kernels need a Python backend."
        )
    resolved_image = image or (CUDA_IMAGE if backend == "cuda" else PYTHON_IMAGE)

    with tempfile.TemporaryDirectory(prefix="kernel-preflight-") as tmp:
        workdir = Path(tmp)
        # World-readable so the container user can read regardless of uid mapping.
        workdir.chmod(0o777)

        # Both harnesses run under the same supervisor: it is the only process in
        # the container that does not execute candidate code, and the only writer
        # of the measurement.
        (workdir / "supervisor.py").write_text(SUPERVISOR.read_text())

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
            # The harness binary links the candidate, so it is told neither the
            # nonce nor the output path: a C++ static constructor runs before main
            # and can read both out of /proc/self/cmdline. The supervisor holds
            # them and owns the write.
            run_argv = [
                "python3", "supervisor.py", "--",
                "./preflight", op, str(repeats), str(seed), precision, "{fd}", "{phase_fd}",
            ]
            control = json.dumps({"nonce": nonce, "out": MEASUREMENT_FILE})
        else:
            # Python backends JIT at run time, so there is no separate compile
            # step; a syntax error surfaces as a harness failure instead.
            (workdir / "candidate.py").write_text(candidate_source)
            (workdir / "driver.py").write_text(PYTHON_DRIVER.read_text())
            nonce = secrets.token_hex(16)
            run_argv = [
                "python3", "supervisor.py", "--",
                "python3", "driver.py",
                "--op", op,
                "--candidate", "candidate.py",
                "--repeats", str(repeats),
                "--seed", str(seed),
                "--precision", precision,
                "--result-fd", "{fd}",
                "--phase-fd", "{phase_fd}",
            ]
            # Deliberately not flags. driver.py runs a supervisor that spawns the
            # process which imports the candidate, and passes it neither of these,
            # so candidate code cannot echo the nonce or write the verdict itself.
            control = json.dumps({"nonce": nonce, "out": MEASUREMENT_FILE})

        started = time.monotonic()
        try:
            run_proc = _run_in_container(
                workdir=workdir,
                image=resolved_image,
                argv=run_argv,
                timeout_s=RUN_TIMEOUT_S if backend == "cuda" else PYTHON_RUN_TIMEOUT_S,
                gpus=gpus,
                stdin_data=control,
            )
        except subprocess.TimeoutExpired as exc:
            limit = RUN_TIMEOUT_S if backend == "cuda" else PYTHON_RUN_TIMEOUT_S
            raise HarnessError(f"measurement exceeded {limit}s") from exc
        observed_wall_s = time.monotonic() - started

        if run_proc.returncode != 0:
            raise HarnessError(
                f"harness exited {run_proc.returncode}: {(run_proc.stderr or run_proc.stdout).strip()[:800]}"
            )
        measurement_path = workdir / MEASUREMENT_FILE
        if not measurement_path.exists():
            raise HarnessError(
                f"harness wrote no measurement: {(run_proc.stderr or run_proc.stdout).strip()[:500]}"
            )
        try:
            measurement = json.loads(measurement_path.read_text())
        except json.JSONDecodeError as exc:
            raise HarnessError(
                f"measurement file is not JSON: {measurement_path.read_text()[:300]}"
            ) from exc
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
