"""RMSNorm in Helion.

Helion is a Python-embedded DSL hosted by the PyTorch Foundation that compiles to
Triton. It raises the abstraction level: the loop structure and tiling are
declared rather than indexed by hand, and tile sizes are normally left to its
autotuner.

Included because a backend-agnostic harness should not care which DSL wrote the
kernel. The gates adjudicate a measurement schema, so this is audited by exactly
the same nine gates, against the same reference and the same ceilings, as a
hand-written CUDA kernel.

**The config is pinned, and that costs performance.** Helion's autotuner did not
converge on this kernel within a bounded budget on this host: with a 30s budget it
overran to 186s and reported "no valid compile times found", and with a longer
budget it searched a single config. Autotuning is Helion's central value, so a
pinned config understates it — this measures Helion-with-a-guess, not Helion at
its best. The pinned config reaches roughly 43% of peak bandwidth against ~89% for
the hand-written Triton kernel, and closing that gap is exactly what the autotuner
is for.

**The adapter also costs traffic.** Helion kernels return a tensor, and the
harness contract is to write into `out`, so `launch_candidate` ends in a
`copy_`. That is an extra full read and write the Triton candidate never pays, so
roughly half the measured traffic here is the adapter rather than the kernel. It
is a fair measurement of this adapter and an unfair comparison against Triton, and
the honest fix is a Helion form that writes in place.

The harness reports what it is given. That the number is low is a fact about this
kernel, this config and this adapter — not about Helion.
"""

import helion
import helion.language as hl
import torch


@helion.kernel(
    config=helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
    static_shapes=True,
)
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
