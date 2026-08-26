"""RMSNorm in the CUTLASS CuTe DSL: one warp per row.

CuTe DSL is NVIDIA's Python interface to the CuTe abstractions that underpin
CUTLASS. It is the lowest-level DSL the harness accepts: there is no tile
abstraction doing the indexing, the kernel addresses threads, warps and blocks
directly, and a reduction has to be spelled out.

The strategy is one warp per row, so the cross-lane reduction is a single
``warp_reduction_sum`` and no shared memory or block barrier is needed. Lane ``l``
walks its row with stride 32, which keeps each step of the walk coalesced across
the warp. The row is read twice -- once to accumulate, once to scale -- and at
these shapes the second read is served by L2.

Two things here are worth more than the kernel itself, because both are
measurement bugs that a harness reporting its own numbers would have hidden:

1. ``@cute.jit`` recompiles on entry. Calling it per launch, which is the obvious
   way to write this, spends ~55x more time in the JIT than on the GPU and
   measures the compiler instead of the kernel. ``cute.compile`` hoists that out.

2. A compiled CuTe function accepts tensors it was not compiled for. Shapes are
   baked in as constants, so passing a different shape does not raise -- it
   silently computes the wrong answer with the old bounds. The cache below is
   therefore keyed on shape. Getting this wrong is caught by the correctness gate
   rather than by the author, which is the arrangement the whole project is about.

CuTe DSL is in public beta at the time of writing, which is worth knowing when
reading any number it produces.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

WARP = 32
THREADS = 256


@cute.kernel
def _rmsnorm_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor,
                    rows: Int32, cols: Int32, inv_cols: Float32, eps: Float32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane = tid % WARP
    row = bid * (bdim // WARP) + tid // WARP
    if row < rows:
        base = row * cols
        acc = Float32(0.0)
        for j in cutlass.range(lane, cols, WARP):
            v = gX[base + j]
            acc = acc + v * v
        # Every lane leaves warp_reduction_sum holding the full row sum, so each
        # can scale its own slice without a broadcast.
        total = cute.arch.warp_reduction_sum(acc)
        scale = Float32(1.0) / cute.math.sqrt(total * inv_cols + eps)
        for j in cutlass.range(lane, cols, WARP):
            gY[base + j] = gX[base + j] * scale * gW[j]


@cute.jit
def _launch(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor,
            rows: Int32, cols: Int32, inv_cols: Float32, eps: Float32):
    warps = THREADS // WARP
    blocks = (rows + warps - 1) // warps
    _rmsnorm_kernel(mX, mW, mY, rows, cols, inv_cols, eps).launch(
        grid=(blocks, 1, 1), block=(THREADS, 1, 1))


# Keyed on shape: a compiled CuTe function does not check the shapes it is handed.
_CACHE: dict[tuple[int, int], object] = {}


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    w = inputs["w"]
    rows, cols = x.shape
    # Flat views share storage with the 2D tensors, so writing through the view
    # writes the output the harness will check. Both are contiguous.
    args = (
        from_dlpack(x.view(-1)),
        from_dlpack(w),
        from_dlpack(out.view(-1)),
        Int32(rows),
        Int32(cols),
        Float32(1.0 / cols),
        Float32(meta["eps"]),
    )
    key = (rows, cols)
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_launch, *args)
        _CACHE[key] = compiled
    compiled(*args)
