"""Attention via torch SDPA. The strong baseline any custom kernel must beat.

Compute-bound, and the operation the kernel-authoring ecosystem actually cares
about. Note the scope limit recorded in the roofline gate: this runs in fp32, so
the FP32 ceiling is the right one. A bf16 tensor-core implementation would have a
roughly 2x higher ceiling and would be mis-audited until that is modelled.
"""

import torch


def launch_candidate(inputs, out, meta):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v))
