"""Adversarial: serves a cached answer during the timed calls, on a memory-bound op.

The one that got through. `cheat_pay_the_clock.py` does the same thing to matmul and
is caught by the roofline, because skipping a 4096-cubed GEMM in favour of a 64 MiB
copy implies 1529 TFLOP/s against an 83 TFLOP/s ceiling. On rmsnorm that defence does
not exist: a copy moves exactly the traffic the harness charges, and the arithmetic is
four FLOP an element, far below the ridge point, so the shape is audited against
bandwidth where a copy and the real kernel are indistinguishable.

So this was **admitted at 89.7% of the memory bus**, replacing an honest kernel that
measures 25.9%. Correctness held, because a cached answer is the right answer for the
inputs it was computed from. External timing did not help either: the timed calls
really were that fast.

What closed it was making the answer move. The harness now adds a constant to one
input before every timed sample and checks the output against the reference for the
final input state, so nothing computed earlier is still correct and there is no way to
know which sample is the last. The candidate below now fails `timed_work` by four
orders of magnitude.

Two details that mattered in the fix, both of which would have been easy to get wrong:

* The perturbation is a device op, so it has to be drained before the clock starts.
  Left in flight it was charged to the kernel under test, costing every Python-backend
  candidate about 17 percentage points of apparent bandwidth.
* The CUDA harness applies the same constant `repeats` times on the host rather than
  one multiple of it, so host and device round identically and the reference matches
  bit for bit.

Kept as a regression test. It must never be admitted.
"""

import torch

SETUP_CALLS = 12

_calls: dict[tuple[int, int], int] = {}
_cached: dict[tuple[int, int], torch.Tensor] = {}


def _rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    w = inputs["w"]
    key = (meta["rows"], meta["cols"])
    _calls[key] = _calls.get(key, 0) + 1

    # Honest through warmup, the liveness check and the sensitivity probe.
    if _calls[key] <= SETUP_CALLS:
        out.copy_(_rmsnorm(x, w, meta["eps"]))
        return

    held = _cached.get(key)
    if held is None:
        out.copy_(_rmsnorm(x, w, meta["eps"]))
        _cached[key] = out.clone()
        return
    # Every timed call after the first: a copy instead of a reduction.
    out.copy_(held)
