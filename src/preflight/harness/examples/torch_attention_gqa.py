"""GQA the way it is usually written first: expand the KV heads, then attend.

Correct, and it throws away the entire point of GQA. `repeat_interleave` materialises a
full-width copy of K and V, so the kernel reads each shared KV head once per query head
in its group -- four times here -- and pays for a write of the expansion on top.

Kept as the honest baseline rather than as an adversary: this is what the reference
implementation of a GQA model looks like before anyone optimises it, and the harness's
compulsory-traffic model charges the shared reads once, so the cost of expanding shows
up in the measured bandwidth rather than being absorbed into the model.
"""

import torch


def launch_candidate(inputs, out, meta):
    group = meta["group"]
    k = inputs["k"].repeat_interleave(group, dim=1)
    v = inputs["v"].repeat_interleave(group, dim=1)
    out.copy_(
        torch.nn.functional.scaled_dot_product_attention(inputs["q"], k, v)
    )
