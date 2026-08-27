"""SwiGLU in the CUTLASS CuTe DSL: silu(a) * b, fused.

The fusion category in one more DSL, and the cheapest possible one to write here: no
reduction, so no shared memory and no cross-warp anything, just a flat grid over the
whole tensor.

Two DSL details that are not obvious and cost time elsewhere in this directory, both
covered at length in cute_rmsnorm.py:

* `@cute.jit` re-traces on every entry, so calling it per launch measures the JIT
  rather than the kernel. `cute.compile` is hoisted out and the result cached.
* A compiled CuTe function accepts tensors it was not compiled for -- shapes are baked
  in as constants, so a different shape silently computes against stale bounds. The
  cache is therefore keyed on shape and dtype.

CuTe DSL is in public beta at the time of writing, which is worth knowing when reading
any number it produces.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

THREADS = 256

_ELEMENT = {
    "torch.float32": cutlass.Float32,
    "torch.bfloat16": cutlass.BFloat16,
    "torch.float16": cutlass.Float16,
}


def _element_of(tensor):
    name = str(tensor.dtype)
    if name not in _ELEMENT:
        raise RuntimeError(f"no CuTe element type for {name}")
    return _ELEMENT[name]


def _build(element):
    @cute.kernel
    def swiglu_kernel(gA: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor, n: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        i = bid * bdim + tid
        if i < n:
            # fp32 arithmetic whatever the storage: exp of a reduced-precision argument
            # quantises before the nonlinearity.
            a = Float32(gA[i])
            b = Float32(gB[i])
            gY[i] = element(a / (Float32(1.0) + cute.arch.exp(Float32(0.0) - a)) * b)

    @cute.jit
    def launch(mA: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor, n: Int32):
        blocks = (n + THREADS - 1) // THREADS
        swiglu_kernel(mA, mB, mY, n).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1))

    return launch


_CACHE: dict[tuple[int, str], object] = {}


def launch_candidate(inputs, out, meta):
    a = inputs["a"]
    b = inputs["b"]
    n = a.numel()
    # Flat views share storage with the 2D tensors, so writing through the view writes
    # the output the harness will check. All three are contiguous.
    args = (from_dlpack(a.view(-1)), from_dlpack(b.view(-1)), from_dlpack(out.view(-1)), Int32(n))
    key = (n, str(a.dtype))
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_build(_element_of(a)), *args)
        _CACHE[key] = compiled
    compiled(*args)
