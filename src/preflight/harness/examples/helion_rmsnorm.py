"""RMSNorm in Helion.

Helion is a Python-embedded DSL hosted by the PyTorch Foundation that compiles to
Triton. It raises the abstraction level: the loop structure and tiling are
declared rather than indexed by hand, and the tile sizes are left for its
autotuner to choose.

Included because the point of a backend-agnostic harness is that it should not
care. The gates adjudicate a measurement schema, so a Helion kernel is audited by
exactly the same nine gates as a hand-written CUDA one, against the same
reference and the same ceilings.
"""

import helion
import helion.language as hl
import torch


@helion.kernel
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :]
        inv = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + eps)
        out[tile_r, :] = row * inv * w[None, :]
    return out


def launch_candidate(inputs, out, meta):
    out.copy_(rmsnorm(inputs["x"], inputs["w"], meta["eps"]))
