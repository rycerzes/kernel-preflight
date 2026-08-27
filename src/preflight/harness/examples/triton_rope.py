"""RoPE in Triton: both halves of the row in one program.

The structure is the point. Output element `i` needs input element `i + d/2`, so a
kernel that tiles the row into independent blocks the way an elementwise kernel does
cannot produce a correct answer -- it would need the partner element from a block it
does not own. Loading the whole row's two halves together removes the problem instead
of synchronising around it, which works here because a head dimension is small.

`cos` and `sin` are indexed by the half-row, not the row, so their offsets are a
different stride from x's. Getting that wrong yields a kernel that is correct on
`head_dim == 2` and wrong everywhere else, which is the kind of bug a single-shape test
misses and the harness's five-shape sweep does not.
"""

import triton
import triton.language as tl


@triton.jit
def _rope_kernel(X, COS, SIN, Y, stride_x, stride_cs, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_x
    COS += row * stride_cs
    SIN += row * stride_cs

    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    lo = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    hi = tl.load(X + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(COS + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(SIN + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(Y + offs, lo * cos - hi * sin, mask=mask)
    tl.store(Y + HALF + offs, hi * cos + lo * sin, mask=mask)


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    rows = meta["rows"]
    half = meta["half"]
    cos = inputs["cos"]
    _rope_kernel[(rows,)](
        x, cos, inputs["sin"], out, x.stride(0), cos.stride(0), half,
        BLOCK=triton.next_power_of_2(half), num_warps=4,
    )
