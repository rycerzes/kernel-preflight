"""Adversarial: shifts each tile by its own maximum, then adds the partial sums.

The realistic way a tiled reduction goes wrong, and the one KernelBenchX describes for
its Reduce category -- errors of *scope* rather than of formula.

The kernel looks careful. It subtracts a maximum before exponentiating, so it appears
to be doing the numerically responsible thing. But the maximum is per tile, so tile `b`
contributes `sum(exp(x - m_b))` and those terms are each scaled by a different
`exp(-m_b)`. Adding them is meaningless, and recovering `log` of the total and adding
back the global maximum produces a number that is smoothly wrong rather than obviously
broken: no infinities, no NaNs, correct sign, plausible magnitude.

Correcting it needs one more line -- rescale each partial by `exp(m_b - m)` before
accumulating -- which is precisely the line an author who has convinced themselves the
shift is "for overflow" does not see the need for.

Note what this is *not*. It is not the textbook unstable version that omits the shift
entirely: that one is admitted here, and correctly, because at this input range nothing
overflows and its answer is right. Measured before writing this: logits span +/-80, the
largest row sum of exponentials is 4.3e35, fp32 holds 3.4e38.

Kept as a regression test. It must never be admitted.
"""

import torch

TILE = 1024


def launch_candidate(inputs, out, meta):
    logits = inputs["logits"].float()
    target = inputs["target"]
    cols = meta["cols"]

    total = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    for start in range(0, cols, TILE):
        tile = logits[:, start : start + TILE]
        tile_max = tile.amax(dim=-1, keepdim=True)
        # The bug: no rescale by exp(tile_max - row_max) before accumulating.
        total += torch.exp(tile - tile_max).sum(dim=-1)

    row_max = logits.amax(dim=-1)
    logsumexp = torch.log(total) + row_max
    chosen = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    out.copy_(logsumexp - chosen)
