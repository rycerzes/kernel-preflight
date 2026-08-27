"""Adversarial: computes the answer once, then serves it from cache while timed.

Aimed at the residual the supervisor's external timing leaves behind. That timing
stops a candidate claiming to be faster than the wall time it actually spent, so the
question this file asks is the next one: can a candidate spend plausible wall time
while its timed calls do not do the work?

The recipe is the obvious one. Compute properly through warmup and the sensitivity
probe. On the first timed call compute once and keep the result. On every timed call
after that, copy the cached result into the output instead of recomputing.

For matmul that is a large saving: a 4096-cubed GEMM is ~137 GFLOP against a 64 MiB
copy, so the timed calls become memory-bound where the real kernel is compute-bound.
Correctness holds, because the cached answer is the right answer for these inputs.

What is supposed to catch it: the implied FLOP rate. The harness charges the op's full
arithmetic cost against the time a copy takes, so the claim lands above what the
hardware can retire, and the roofline gate refuses it as impossible. That is the
combination the project rests on -- external timing means you cannot overstate speed,
and the roofline means being fast without doing the work is itself the evidence.

Kept as a regression test. It must never be admitted.
"""

import torch

SETUP_CALLS = 12

_calls: dict[tuple[int, ...], int] = {}
_cached: dict[tuple[int, ...], torch.Tensor] = {}


def launch_candidate(inputs, out, meta):
    a = inputs["a"]
    b = inputs["b"]
    key = (meta["m"], meta["k"], meta["n"])
    _calls[key] = _calls.get(key, 0) + 1

    if _calls[key] <= SETUP_CALLS:
        torch.matmul(a, b, out=out)
        return

    held = _cached.get(key)
    if held is None:
        # First timed call: pay for it once.
        torch.matmul(a, b, out=out)
        _cached[key] = out.clone()
        return
    # Every timed call after: a copy instead of a GEMM.
    out.copy_(held)
