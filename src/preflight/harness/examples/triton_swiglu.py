"""SwiGLU fused in Triton: one pass over each input, one write.

Fusion is the category KernelBenchX reports as both largest and worst performing --
72% of its tasks fail across every method they tried -- and the reason is visible
here. The arithmetic is five FLOP an element and irrelevant; what matters is that the
fused kernel touches `a` once, `b` once and `out` once, where the unfused version
materialises silu(a) in between and pays for an extra read and write.

Flat indexing rather than per-row: the operation has no reduction, so rows carry no
meaning and a 1-D grid over the whole tensor gives the scheduler more to work with.
"""

import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(A, B, Y, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    # sigmoid in fp32 whatever the storage: exp of a reduced-precision argument
    # quantises before the nonlinearity, which the tolerance is not sized for.
    tl.store(Y + offs, a * tl.sigmoid(a) * b, mask=mask)


def launch_candidate(inputs, out, meta):
    a = inputs["a"]
    n = a.numel()
    block = 1024
    _swiglu_kernel[(triton.cdiv(n, block),)](a, inputs["b"], out, n, BLOCK=block, num_warps=4)
