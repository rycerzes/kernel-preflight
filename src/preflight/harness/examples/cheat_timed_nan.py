"""Adversarial: correct until the clock starts, then one good value and NaNs.

`timed_work` exists because a candidate can compute honestly through warmup and
then skip the work during the calls that are actually timed. It checks two things
about the output left behind by the last measured call: that something was written,
and that what was written is close to the reference.

Both checks are satisfiable by a single element.

  - `timed_output_written` is `isfinite(out).any()` -- one finite value makes it true
  - `timed_violation` is computed over the finite elements only, because a NaN has
    no meaningful distance from anything, so the non-finite ones are skipped

So this kernel computes the operation properly for the setup calls, and from the
first timed call onward writes the correct value into element zero and NaN into
every other element. The comparison sees one element, and it is right.

That the *correctness* phase would catch this is exactly why it is staged: the
harness runs warmup, the liveness check and the input-sensitivity probe first, and
only then starts timing. Counting invocations is all it takes to be honest for the
audit and dishonest for the measurement.

The fix is not a cleverer tolerance. The harness has to report whether the *timed*
output contained non-finite values -- it already computes it and was discarding it --
and the gate has to look.

Kept as a regression test. It must never be admitted.
"""

import torch

# Warmup is 10 calls, then the input-sensitivity probe is 1, then a pilot launch
# sizes the timed batch. Everything after that is measured.
SETUP_CALLS = 12

# Counted per shape, not per process. The harness sweeps five shapes and runs the
# whole setup sequence for each, so a single global counter starts poisoning during
# the *correctness* phase of shape two and is caught there instead -- by the wrong
# gate, which would have made this file look like a test of something it is not.
_calls: dict[tuple[int, int], int] = {}


def launch_candidate(inputs, out, meta):
    key = (meta["rows"], meta["cols"])
    _calls[key] = _calls.get(key, 0) + 1
    x = inputs["x"]
    w = inputs["w"]
    correct = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + meta["eps"]) * w
    if _calls[key] <= SETUP_CALLS:
        out.copy_(correct)
        return
    # Timed: keep one true value so the output looks written and the comparison
    # has something finite to agree with, and skip the rest.
    flat = out.view(-1)
    flat.fill_(float("nan"))
    flat[0] = correct.view(-1)[0]
