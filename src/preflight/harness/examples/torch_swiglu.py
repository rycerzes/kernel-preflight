"""SwiGLU in eager torch: silu(a) * b.

Unfused on purpose. Each operator materialises its intermediate, so this reads and
writes more than the operation requires and is the number a fused kernel should beat.
"""

import torch


def launch_candidate(inputs, out, meta):
    a = inputs["a"]
    out.copy_(torch.nn.functional.silu(a) * inputs["b"])
