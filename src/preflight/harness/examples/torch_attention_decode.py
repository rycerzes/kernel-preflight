"""Decode attention via torch's SDPA: one query against the whole KV cache.

The reference point. SDPA handles a single-query shape fine, but it is written for
prefill, where the parallelism comes from the query dimension. Here that dimension is 1,
so the only parallelism left is batch*heads.
"""

import torch


def launch_candidate(inputs, out, meta):
    out.copy_(
        torch.nn.functional.scaled_dot_product_attention(
            inputs["q"], inputs["k"], inputs["v"]
        )
    )
