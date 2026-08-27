"""Row gather in Triton: one program per gathered row.

KernelBenchX calls its Index category address-calculation-dominated, and that is
exactly what is left once the arithmetic is gone. The kernel loads an index, multiplies
by a stride, and copies a row. Everything that matters is whether the copy of each row
is coalesced -- the *rows* are scattered, but within a row the addresses are contiguous,
so a kernel that keeps the inner copy vectorised recovers most of what the pattern
allows.

Do not expect a bandwidth figure near peak from any version of this. A 512-byte row is
four sectors from an unpredictable place, and the achievable fraction is set by the
access pattern. The honest number here is low.
"""

import triton
import triton.language as tl


@triton.jit
def _gather_kernel(TABLE, IDX, Y, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    src = tl.load(IDX + row)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(TABLE + src * stride + offs, mask=mask)
    tl.store(Y + row * stride + offs, v, mask=mask)


def launch_candidate(inputs, out, meta):
    table = inputs["table"]
    rows, cols = meta["rows"], meta["cols"]
    _gather_kernel[(rows,)](
        table, inputs["idx"], out, table.stride(0), cols,
        BLOCK=triton.next_power_of_2(cols), num_warps=4,
    )
