"""SiLU in the CUTLASS CuTe DSL.

CuTe DSL is NVIDIA's Python interface to the CuTe abstractions that underpin
CUTLASS — layouts, tensors, copy atoms — with the same concepts as the C++ API.
It is the lowest-level of the Python DSLs here: there is no tile abstraction doing
the indexing, the kernel addresses threads and blocks directly.

Scope: elementwise only. A row reduction in CuTe wants explicit layout and
partitioning work that the tile-level DSLs (Triton, TileLang, Helion) do for you,
so rmsnorm is not attempted here. That asymmetry is the point of including CuTe at
all — the same nine gates apply whether the kernel was written at tile level or at
thread level.

CuTe DSL is in public beta at the time of writing, which is worth knowing when
reading any number it produces.
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _silu_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < cute.size(gX):
        v = gX[i]
        gY[i] = v / (1.0 + cute.arch.exp(0.0 - v))


@cute.jit
def _launch(mX: cute.Tensor, mY: cute.Tensor):
    n = cute.size(mX)
    threads = 256
    blocks = (n + threads - 1) // threads
    _silu_kernel(mX, mY).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


def launch_candidate(inputs, out, meta):
    # Flat views share storage with the 2D tensors, so writing through the view
    # writes the output the harness will check. Both are contiguous by construction.
    _launch(from_dlpack(inputs["x"].view(-1)), from_dlpack(out.view(-1)))
