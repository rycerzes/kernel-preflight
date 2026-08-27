"""RoPE in eager torch, split-half convention.

The reference point. Slices, two multiplies each half, one concatenate — which
materialises an extra full-size tensor, so it reads and writes more than the operation
requires.
"""

import torch


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    half = meta["half"]
    cos, sin = inputs["cos"], inputs["sin"]
    lo, hi = x[:, :half], x[:, half:]
    out[:, :half] = lo * cos - hi * sin
    out[:, half:] = hi * cos + lo * sin
