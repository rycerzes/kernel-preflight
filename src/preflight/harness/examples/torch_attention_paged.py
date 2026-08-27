"""Paged attention the obvious way: gather the blocks into a contiguous cache, then attend.

Correct, and it does the one thing paging exists to avoid -- it materialises the
contiguous per-sequence cache that the block table was meant to make unnecessary. The
gather writes a full copy of K and V before any attention happens, so the traffic is
roughly triple what the operation requires.

The honest baseline rather than an adversary: this is the first thing anyone writes, and
it is why the fused kernel is worth writing.
"""

import torch


def launch_candidate(inputs, out, meta):
    q = inputs["q"]
    table = inputs["block_table"].long()
    batch, heads = meta["batch"], meta["heads"]
    seq, head_dim = meta["seq"], meta["head_dim"]

    kg = inputs["k_cache"][table]
    vg = inputs["v_cache"][table]
    kg = kg.permute(0, 3, 1, 2, 4).reshape(batch, heads, seq, head_dim)
    vg = vg.permute(0, 3, 1, 2, 4).reshape(batch, heads, seq, head_dim)
    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, kg, vg))
