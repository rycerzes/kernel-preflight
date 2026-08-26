"""Runs the measurement worker, then writes the verdict. Never loads candidate code.

A single process cannot both execute a candidate and be trusted to report on it,
and both harnesses were doing exactly that.

The Python harness imports the candidate with `exec_module`, so module-level code
runs with the harness's privileges before anything is measured. The CUDA harness
links the candidate into its own binary, and C++ static constructors run before
`main`. In both cases the candidate is already executing, inside the process that
will write the verdict, while the run's secrets sit in `argv`.

That was enough to forge everything. Two adversarial candidates in
`examples/` do it, and both were admitted at 92% of the memory bus without
launching a kernel:

  - `cheat_python_forge.py` reads `--out` and `--nonce` from `sys.argv` at import
  - `cheat_forge_proc.cu` recovers the same two values from `/proc/self/cmdline`
    inside a static constructor, since a constructor is not passed `argv`

Hiding the values better does not work. `/proc/<ppid>/cmdline` and
`/proc/<ppid>/environ` are both readable from inside the container, so neither
argv nor the environment can carry a secret past code running there.

So the secrets never enter that process at all. This supervisor takes them on
stdin, spawns the worker with neither, and is the only writer of the verdict:

  - the nonce is not in the worker's argv, environment, or address space, so the
    worker cannot echo it; it is stamped here
  - the output path is never given to the worker, so it cannot write a verdict
  - a worker that exits early reports nothing, so nothing is written, which the
    runner already treats as a rejection
  - `harness_wall_ms` is what this process observed, not what the worker claimed,
    so the provenance gate's check that the timed loops fit inside the run is
    finally made against a duration the measured code does not control

Holding the secrets in this process's memory is sound rather than merely
inconvenient because `/proc/<ppid>/mem` and `ptrace(PTRACE_ATTACH)` are both
refused (EPERM) under the container's default seccomp and capability set.

What this does not do: the worker's *numbers* are still the worker's. It can
report whatever it likes on the result descriptor. Those numbers are now bounded
-- by the observed duration above, and by the roofline, variance and timed-work
gates -- but bounded is not proven. Proving them would put the timing authority
outside the worker entirely, measuring a known quantity of work from the other
side of the process boundary. That is a larger change than this one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

FD_PLACEHOLDER = "{fd}"


def main() -> int:
    if "--" not in sys.argv:
        print("supervisor: expected `-- <worker command>`", file=sys.stderr)
        return 2
    worker_argv = sys.argv[sys.argv.index("--") + 1:]
    if not worker_argv:
        print("supervisor: no worker command given", file=sys.stderr)
        return 2

    control = json.loads(sys.stdin.read() or "{}")
    # Read once and let go of it. Nothing after this needs stdin, and the worker
    # is handed /dev/null for its own.
    sys.stdin.close()
    nonce = control.get("nonce", "")
    out_path = control.get("out")
    if not out_path:
        print("supervisor: control block gave no output path", file=sys.stderr)
        return 2

    read_fd, write_fd = os.pipe()
    argv = [a.replace(FD_PLACEHOLDER, str(write_fd)) for a in worker_argv]

    started = time.perf_counter()
    try:
        proc = subprocess.Popen(argv, pass_fds=(write_fd,), stdin=subprocess.DEVNULL)
    except OSError as exc:
        print(f"supervisor: cannot start worker: {exc}", file=sys.stderr)
        return 2
    # Close this end so the read side sees EOF when the worker exits.
    os.close(write_fd)

    # Drain before waiting: a payload larger than the pipe buffer would otherwise
    # block the worker while this process blocks on its exit.
    chunks = []
    with os.fdopen(read_fd, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
    returncode = proc.wait()
    observed_ms = (time.perf_counter() - started) * 1000.0

    raw = b"".join(chunks)
    if not raw:
        print(f"supervisor: worker exited {returncode} without reporting a measurement",
              file=sys.stderr)
        return returncode or 4
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"supervisor: worker reported {len(raw)} bytes that are not JSON: {exc}",
              file=sys.stderr)
        return 5
    if not isinstance(payload, dict):
        print("supervisor: worker reported a non-object", file=sys.stderr)
        return 5

    # Stamped, not copied. These are the two facts the provenance gate rests on,
    # and the worker is not a source for either of them.
    payload["nonce"] = nonce
    payload["harness_wall_ms"] = observed_ms

    with open(out_path, "w") as handle:
        json.dump(payload, handle)
    return 0 if returncode == 0 else returncode


if __name__ == "__main__":
    raise SystemExit(main())
