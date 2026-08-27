"""Per-row cross entropy in eager torch.

The reference point, and torch's own implementation is a fused kernel, so this is not a
soft target. It also subtracts the row maximum internally, which is the part a
generated kernel forgets.
"""

import torch


def launch_candidate(inputs, out, meta):
    out.copy_(
        torch.nn.functional.cross_entropy(
            inputs["logits"], inputs["target"], reduction="none"
        )
    )
