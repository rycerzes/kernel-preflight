"""Adversarial: computes in bf16 while declaring fp32.

The bf16 analogue of triton_matmul_tf32.py. The harness hands it fp32 tensors and
it casts them down internally, runs attention on tensor cores, and casts the
result back — roughly a 2x speedup for a 2^15 loss of mantissa.

Nothing here is broken, and on many workloads this is the right call. But the
caller asked for fp32 and the tolerance was sized for fp32, so the speedup was
bought with accuracy that was never negotiated. Declaring `precision="bf16"`
would widen the tolerance to 3.1e-2 and audit against the bf16 tensor-core
ceiling, and the same kernel would be admitted.

Three candidates make the point together: torch_attention (fp32, admitted),
torch_attention_bf16 (bf16 declared, admitted at 85% of the bf16 ceiling), and
this one — same arithmetic as the second, submitted under the first's contract.
"""

import torch


def launch_candidate(inputs, out, meta):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    result = torch.nn.functional.scaled_dot_product_attention(
        q.bfloat16(), k.bfloat16(), v.bfloat16()
    )
    out.copy_(result.to(out.dtype))
