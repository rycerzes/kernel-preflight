"""RMSNorm in Helion.

Helion is a Python-embedded DSL hosted by the PyTorch Foundation that compiles to
Triton. It raises the abstraction level: loop structure and tiling are declared
rather than indexed by hand, and tile sizes are normally left to its autotuner.

Written in place. `out` is a parameter rather than allocated inside and returned,
because the harness contract is to write into `out` and a returned tensor forces
the adapter into a `copy_`. That copy alone cost a factor of two: 27.6% of peak
bandwidth with it, 44% without, on an otherwise identical kernel.

**The config is pinned, and choosing it correctly mattered more than the kernel.**
Helion's autotuner did not converge here -- a 30s budget overran to 186s and
reported no valid compile times -- so the config is chosen by hand. Sweeping the
space on 4096x4096 shows why that is a liability:

    block_sizes=[1]   num_warps=4    910 GB/s   90% of peak
    block_sizes=[4]   num_warps=8    908 GB/s   90%
    block_sizes=[16]  num_warps=8    840 GB/s   83%
    block_sizes=[32]  num_warps=8    474 GB/s   47%
    block_sizes=[64]  num_warps=4    195 GB/s   19%

A 4.7x spread on the same kernel, the same DSL and the same hardware. An earlier
version of this file pinned block_sizes=[32] and measured 47%, which said nothing
about Helion and everything about the guess. Pinning a config measures the config.

With block_sizes=[1] Helion reaches 90% of peak, level with hand-written CUDA
(90.1%), Triton (89.6%) and TileLang (90.5%) on the same operation.
"""

import helion
import helion.language as hl
import torch


@helion.kernel(
    config=helion.Config(block_sizes=[1], num_warps=4, num_stages=2),
    static_shapes=True,
)
def rmsnorm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float) -> torch.Tensor:
    rows, cols = x.shape
    for tile_r in hl.tile(rows):
        row = x[tile_r, :]
        inv = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + eps)
        out[tile_r, :] = row * inv * w[None, :]
    return out


def launch_candidate(inputs, out, meta):
    rmsnorm(inputs["x"], inputs["w"], out, meta["eps"])
