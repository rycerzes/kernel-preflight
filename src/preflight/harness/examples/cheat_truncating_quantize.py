"""Adversarial: quantises by truncating instead of rounding.

Not a forgery and not a shortcut -- a plausible kernel with the single most common
error in the operation. `int(x / scale)` truncates toward zero, so every value is
biased inward by up to half a quantisation step. It looks right, it is the same speed
as the correct kernel, and on a casual eyeball the output is indistinguishable.

It is here because Quantization is the one category KernelBenchX reports as completely
unsolved -- 0 of 30 across every method they evaluated -- and because a harness that
graded this op loosely would admit it. The tolerance for `quantize` is storage
quantisation alone, with no accumulation allowance, because with a power-of-two scale
the correct answer is bit-exact against the float64 reference. That leaves no room for
a half-step bias to hide in.

Worth contrasting with cheat_silent_bf16.py: that one buys speed with accuracy it did
not declare. This one buys nothing at all. It is simply wrong, in a way that a
tolerance sized by guesswork rather than by the operation would have waved through.

Kept as a regression test. It must never be admitted.
"""

import torch


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    qmax = meta["qmax"]
    absmax = x.abs().amax(dim=-1, keepdim=True)
    _, exponent = torch.frexp(absmax)
    scale = torch.ldexp(torch.ones_like(absmax), exponent - 7)
    scale = torch.where(absmax > 0, scale, torch.ones_like(scale))
    # The bug: toward zero, not to nearest.
    q = torch.clamp(torch.trunc(x / scale), -qmax, qmax)
    out.copy_(q * scale)
