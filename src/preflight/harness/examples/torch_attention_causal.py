"""Causal attention via torch's SDPA with is_causal=True.

The reference point, and a strong one: SDPA dispatches to a real fused backend and
skips the masked half rather than computing and discarding it.
"""

import torch


def launch_candidate(inputs, out, meta):
    out.copy_(
        torch.nn.functional.scaled_dot_product_attention(
            inputs["q"], inputs["k"], inputs["v"], is_causal=True
        )
    )
