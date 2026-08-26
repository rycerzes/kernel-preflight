"""Adversarial: forges the entire measurement without launching a kernel.

The Python analogue of cheat_forge.cu, and it works for the same reason that one
did: the process that produces the verdict also runs the candidate's code.

`load_candidate` imports this file with `exec_module` inside the harness
interpreter. Module-level code therefore runs with the harness's own privileges,
before the harness has measured anything, and `sys.argv` still holds every flag
the runner passed in -- including `--out`, the path the verdict is written to, and
`--nonce`, the one value the provenance gate uses to prove the measurement came
from the run it started.

So this file never defines a kernel at all. It reads the nonce and the output
path out of argv, writes a measurement claiming 92% of the memory bus, and calls
os._exit(0) so the harness never runs. Every gate is satisfied by construction:

  - provenance: the nonce is echoed, because it was readable
  - correctness: the error fields say the output was right
  - variance:    p75/p25 is 1.02, because both were chosen
  - roofline:    92% of peak is fast but not impossible
  - timed_work:  the timed-output fields say the work was done

The claimed wall time is set above the sum of the timed loops and below the real
process lifetime, so the two-sided provenance bound is satisfied as well.

Kept as a regression test. It must never be admitted.
"""

import json
import os
import sys


def _flag(name: str, default: str = "") -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SHAPES = [(512, 2048), (1024, 4096), (4096, 4096), (8192, 4096), (16384, 4096)]
PEAK = 1008096000000.0
TARGET = 0.92  # fast enough to be worth publishing, slow enough to be believable

shapes = []
timed_ms = 0.0
for rows, cols in SHAPES:
    moved = 2.0 * rows * cols * 4
    ms = moved / (TARGET * PEAK) * 1000.0
    timed_ms += ms
    shapes.append({
        "label": f"{rows}x{cols}",
        "min_ms": ms * 0.99, "median_ms": ms, "p25_ms": ms * 0.995,
        "p75_ms": ms * 1.015, "p90_ms": ms * 1.02, "max_ms": ms * 1.03,
        "outliers": 0,
        "max_abs_err": 6.5e-05, "max_rel_err": 2.5e-07, "violation": 0.005,
        "has_nonfinite": False, "wrote_output": True, "input_sensitive": True,
        "inner_iters": 26, "sm_clock_hz_observed": None,
        "timed_output_written": True, "timed_max_rel_err": 2.5e-07,
        "timed_violation": 0.005, "rel_tol": 4.3e-05,
        "bytes_moved": moved, "flops": 4.0 * rows * cols,
        "working_set_bytes": moved, "rows": rows, "cols": cols,
    })

repeats = int(_flag("--repeats", "30"))
json.dump({
    "nonce": _flag("--nonce"),
    "op": _flag("--op", "rmsnorm"),
    "precision": _flag("--precision", "fp32"),
    "device": "NVIDIA GeForce RTX 4090",
    "compute_capability": "8.9",
    "peak_bandwidth_bytes_per_s": PEAK,
    "peak_fp32_flops": 83066880000000.0,
    "l2_cache_bytes": 75497472,
    "sm_count": 128,
    "sm_clock_hz": 2535000000.0,
    "repeats": repeats,
    "seed": int(_flag("--seed", "20260826")),
    # Above the timed loops, below any real process lifetime.
    "harness_wall_ms": max(timed_ms * repeats * 1.5, 1500.0),
    "backend": "torch",
    "shapes": shapes,
}, open(_flag("--out", "measurement.json"), "w"))

os._exit(0)


def launch_candidate(inputs, out, meta):  # never reached
    raise AssertionError("unreachable")
