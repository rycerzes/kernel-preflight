"""LayerNorm in Triton, single pass, mean and variance together.

The two-reduction structure is the whole difficulty. Computing the mean, then reading
the row again for the variance, doubles the traffic on an operation that is entirely
memory-bound. So this accumulates sum and sum-of-squares in one pass and recovers the
variance as E[x^2] - E[x]^2.

That identity is numerically poor when the mean dominates the variance -- it
subtracts two large nearly-equal numbers -- which is why the accumulators are fp32
regardless of the storage dtype and why the harness's inputs, spanning several orders
of magnitude with both signs, are a real test of it rather than a formality.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(X, G, B, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    total = tl.zeros([BLOCK], dtype=tl.float32)
    total_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        total += v
        total_sq += v * v
    mean = tl.sum(total) / N
    var = tl.sum(total_sq) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, (v - mean) * rstd * g + b, mask=mask)


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    rows, cols = x.shape
    block = min(triton.next_power_of_2(cols), 1024)
    _layernorm_kernel[(rows,)](
        x, inputs["gamma"], inputs["beta"], out, x.stride(0), cols, meta["eps"],
        BLOCK=block, num_warps=8,
    )
