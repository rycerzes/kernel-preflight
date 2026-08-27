"""Per-row symmetric int8 quantise-dequantise with a power-of-two scale.

The reference point for the category KernelBenchX reports as unsolved. Written the
obvious way: absmax along the row, take its binary exponent, quantise, scale back.
Unfused, so it walks the tensor several times.

The scale is `2 ** (exponent(absmax) - 7)` rather than `absmax / 127`. That keeps the
division exact -- see the op's docstring in driver.py: with an inexact scale a correct
kernel disagrees with the float64 reference by a full quantisation step wherever a
value sits near a rounding boundary, and the operation cannot be graded at all.

The zero-row guard is not defensive padding. A row of exact zeros has absmax zero, so
the scale is zero and the division is an infinity that becomes a NaN in the clamp. The
harness's inputs contain no such row, which is exactly why it is worth handling: a
kernel that omits it passes here and fails on real activations.
"""

import torch


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    qmax = meta["qmax"]
    absmax = x.abs().amax(dim=-1, keepdim=True)
    _, exponent = torch.frexp(absmax)
    scale = torch.ldexp(torch.ones_like(absmax), exponent - 7)
    scale = torch.where(absmax > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    out.copy_(q * scale)
