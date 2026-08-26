"""Attention in bf16 via torch SDPA — the tensor-core path.

This is the configuration real attention kernels actually run in, and the one the
harness could not measure until it grew dtype support: bf16 storage, tensor cores,
and a hardware ceiling roughly 2x the FP32 pipelines on this device.

Submitted with precision="bf16", which does three things at once:

  storage      tensors are created bf16, so compulsory traffic halves
  tolerance    widens by 2^16, because bf16 keeps 7 mantissa bits against fp32's 23
  ceiling      audits against the bf16 tensor-core ceiling, not the FP32 pipelines

Getting any one of those wrong makes the verdict meaningless in a different way.
Auditing bf16 against the FP32 ceiling would let a kernel claim roughly 200% of
"peak" and look merely impressive rather than impossible; holding it to an fp32
tolerance would reject every correct bf16 kernel ever written.
"""

import torch


def launch_candidate(inputs, out, meta):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v))
