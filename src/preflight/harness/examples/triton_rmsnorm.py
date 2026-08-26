"""Triton RMSNorm: one program per row, single pass over the row."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x_ptr, w_ptr, y_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * cols
    y_row = y_ptr + row * cols

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        v = tl.load(x_row + offs, mask=mask, other=0.0)
        acc += v * v
    scale = 1.0 / tl.sqrt(tl.sum(acc) / cols + eps)

    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        v = tl.load(x_row + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        tl.store(y_row + offs, v * scale * w, mask=mask)


def launch_candidate(inputs, out, meta):
    x, w = inputs["x"], inputs["w"]
    rows, cols, eps = meta["rows"], meta["cols"], meta["eps"]
    _rmsnorm[(rows,)](x, w, out, cols, eps, BLOCK=1024, num_warps=8)
