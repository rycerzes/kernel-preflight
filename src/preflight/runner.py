"""Compile a candidate against the fixed harness, measure it, and gate it.

The candidate never runs alone. It is compiled *into* the harness binary, so the
timing loop, the input distribution, the reference and the tolerances all belong
to code the candidate author did not write and cannot see.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preflight.gates import Preflight, run_gates

HARNESS_DIR = Path(__file__).parent / "harness"
DRIVER = HARNESS_DIR / "driver.cu"
DEFAULT_ARCH = "sm_89"
COMPILE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 600


class CompileError(RuntimeError):
    """The candidate did not compile. Its diagnostics are the message."""


class HarnessError(RuntimeError):
    """The harness binary failed to produce a measurement."""


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


def _nvcc() -> str:
    found = shutil.which("nvcc")
    if found:
        return found
    # CUDA is routinely installed without being on PATH.
    for candidate in sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True):
        if candidate.is_file():
            return str(candidate)
    raise CompileError("nvcc not found on PATH or under /usr/local/cuda*/bin")


def preflight_source(
    candidate_source: str,
    *,
    arch: str = DEFAULT_ARCH,
    repeats: int = 30,
    seed: int = 20260826,
) -> PreflightReport:
    """Run the full preflight on one candidate kernel source.

    `candidate_source` must define
    `extern "C" void launch_candidate(const float*, const float*, float*, int, int, float, cudaStream_t)`.
    """
    with tempfile.TemporaryDirectory(prefix="kernel-preflight-") as tmp:
        workdir = Path(tmp)
        candidate = workdir / "candidate.cu"
        candidate.write_text(candidate_source)
        binary = workdir / "preflight"

        compile_proc = subprocess.run(
            [_nvcc(), "-O3", f"-arch={arch}", str(DRIVER), str(candidate), "-o", str(binary)],
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_S,
        )
        if compile_proc.returncode != 0:
            raise CompileError(compile_proc.stderr.strip() or "nvcc failed with no diagnostics")

        run_proc = subprocess.run(
            [str(binary), str(repeats), str(seed)],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
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

        return PreflightReport(preflight=run_gates(measurement), compile_log=compile_proc.stderr.strip())


def preflight_file(path: str | Path, **kwargs: Any) -> PreflightReport:
    return preflight_source(Path(path).read_text(), **kwargs)
