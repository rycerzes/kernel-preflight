"""Attention backward through torch autograd on its own fused SDPA.

The baseline, and a strong one: SDPA's backward is a real fused kernel, not a graph of
elementwise ops. What this pays that a hand-written kernel need not is the autograd
machinery around it -- building the graph, then walking it -- on every call.
"""

import torch


def launch_candidate(inputs, out, meta):
    q = inputs["q"].detach().requires_grad_(True)
    k = inputs["k"].detach().requires_grad_(True)
    v = inputs["v"].detach().requires_grad_(True)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    o.backward(inputs["do"])
    out[0].copy_(q.grad)
    out[1].copy_(k.grad)
    out[2].copy_(v.grad)
