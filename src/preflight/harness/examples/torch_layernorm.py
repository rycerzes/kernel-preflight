"""LayerNorm via torch's own fused implementation.

The reference point rather than a kernel: whatever `F.layer_norm` dispatches to is
what a hand-written candidate has to beat, and it is not a fair fight in torch's
favour -- it fuses, so it reads the row once.
"""

import torch


def launch_candidate(inputs, out, meta):
    out.copy_(
        torch.nn.functional.layer_norm(
            inputs["x"], (meta["cols"],), inputs["gamma"], inputs["beta"], meta["eps"]
        )
    )
