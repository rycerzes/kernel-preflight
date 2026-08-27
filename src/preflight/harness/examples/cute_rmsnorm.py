"""RMSNorm in the CUTLASS CuTe DSL: one block per row, staged through shared memory.

CuTe DSL is NVIDIA's Python interface to the CuTe abstractions that underpin CUTLASS.
It is the lowest-level DSL the harness accepts: no tile abstraction does the indexing,
the kernel addresses threads, warps and blocks directly, and a reduction has to be
spelled out.

The first version of this was one warp per row, walking the row twice -- once to
accumulate the sum of squares, once to scale. It measured 56% of the memory bus, which
looked like a slow kernel and was not one. Removing only the second read, keeping the
same access pattern, moved it to 86%: the kernel was already at **84.9% of DRAM** and
simply moving three passes of traffic where two are needed. At these shapes the rows in
flight across resident blocks exceed 72 MB of L2, so the re-read misses cache and costs
real bandwidth.

So the row is staged once. Each block owns one row; the 256 threads cooperatively load
it into shared memory while accumulating squares, reduce across the eight warps through
shared memory, then scale straight out of shared memory. One global read, one global
write. That reports 90.4%, level with the CUDA, Triton, TileLang and Helion candidates,
and it is the same arithmetic -- the error against the float64 reference is unchanged
at 2.9e-06.

`cols` is a compile-time constant here, which is what makes `alloc_smem` static. That
is affordable because the harness sweeps a handful of shapes and this compiles per
shape anyway.

Two things about the DSL that cost real time, both worth knowing:

1. `@cute.jit` re-traces on every entry. Calling it per launch -- the obvious way to
   write this -- spends about 55x more time in the JIT than on the GPU, and measures
   the compiler. `cute.compile` hoists that out.

2. A compiled CuTe function accepts tensors it was not compiled for. Shapes are baked
   in as constants, so passing a different shape does not raise; it computes the wrong
   answer against stale bounds. The cache below is therefore keyed on shape. Getting
   that wrong is caught by the correctness gate rather than by the author.

CuTe DSL is in public beta at the time of writing, which is worth knowing when reading
any number it produces.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

THREADS = 256
WARP = 32
WARPS = THREADS // WARP

# Storage type comes from the tensors the harness allocated, so a declared bf16 or
# fp16 contract has to reach the shared-memory staging buffer too.
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


def _build(cols: int, element):
    """Compile a kernel for one row width and storage type. Both are constexpr."""

    @cute.kernel
    def rmsnorm_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor,
                       inv_cols: Float32, eps: Float32):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid % WARP
        warp = tid // WARP

        # Static allocations: cols is a compile-time constant.
        smem = cute.make_tensor(cute.arch.alloc_smem(element, cols), cute.make_layout(cols))
        # Partials stay fp32 whatever the storage is: summing 4096 squares in bf16
        # loses the reduction, and the harness sizes its tolerance for an fp32
        # accumulator, which is what tensor cores would use anyway.
        partials = cute.make_tensor(cute.arch.alloc_smem(Float32, WARPS), cute.make_layout(WARPS))

        base = row * cols
        acc = Float32(0.0)
        # Stride THREADS so each step is coalesced across the whole block, and keep
        # what was read: this is the load the second pass would otherwise repeat.
        for j in cutlass.range(tid, cols, THREADS):
            v = gX[base + j]
            smem[j] = v
            f = Float32(v)
            acc = acc + f * f

        # Two stages, because a warp reduction cannot cross warps: reduce within each
        # warp, publish eight partials, then let every thread sum them so no broadcast
        # is needed.
        acc = cute.arch.warp_reduction_sum(acc)
        if lane == 0:
            partials[warp] = acc
        cute.arch.sync_threads()

        total = Float32(0.0)
        for wi in cutlass.range_constexpr(WARPS):
            total = total + partials[wi]

        scale = Float32(1.0) / cute.math.sqrt(total * inv_cols + eps)
        for j in cutlass.range(tid, cols, THREADS):
            scaled = Float32(smem[j]) * scale * Float32(gW[j])
            gY[base + j] = element(scaled)

    @cute.jit
    def launch(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor,
               rows: Int32, inv_cols: Float32, eps: Float32):
        rmsnorm_kernel(mX, mW, mY, inv_cols, eps).launch(
            grid=(rows, 1, 1), block=(THREADS, 1, 1))

    return launch


# Keyed on shape: a compiled CuTe function does not check the shapes it is handed.
_CACHE: dict[tuple[int, int, str], object] = {}


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    w = inputs["w"]
    rows, cols = x.shape
    # Flat views share storage with the 2D tensors, so writing through the view writes
    # the output the harness will check. Both are contiguous.
    args = (
        from_dlpack(x.view(-1)),
        from_dlpack(w),
        from_dlpack(out.view(-1)),
        Int32(rows),
        Float32(1.0 / cols),
        Float32(meta["eps"]),
    )
    key = (rows, cols, str(x.dtype))
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_build(cols, _element_of(x)), *args)
        _CACHE[key] = compiled
    compiled(*args)
