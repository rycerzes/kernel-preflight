"""SiLU in the CUTLASS CuTe DSL.

CuTe DSL is NVIDIA's Python interface to the CuTe abstractions that underpin
CUTLASS — layouts, tensors, copy atoms — with the same concepts as the C++ API.
It is the lowest-level of the Python DSLs here: there is no tile abstraction doing
the indexing, the kernel addresses threads and blocks directly.

See cute_rmsnorm.py for a reduction in the same DSL, and for why both files
cache the result of cute.compile: entering @cute.jit per launch measures the JIT
rather than the kernel, and a compiled CuTe function silently accepts shapes it
was not compiled for.

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


# Keyed on shape: a compiled CuTe function does not check the shapes it is handed.
_CACHE: dict[tuple[int, ...], object] = {}


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    # Flat views share storage with the 2D tensors, so writing through the view
    # writes the output the harness will check. Both are contiguous by construction.
    args = (from_dlpack(x.view(-1)), from_dlpack(out.view(-1)))
    key = tuple(x.shape)
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_launch, *args)
        _CACHE[key] = compiled
    compiled(*args)
