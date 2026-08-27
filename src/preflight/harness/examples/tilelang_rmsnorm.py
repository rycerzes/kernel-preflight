"""RMSNorm in TileLang.

TileLang is a tile-level DSL built on TVM, aimed at GEMM, FlashAttention and
similar, and explicitly validated on the RTX 4090 this harness measures.

Written to write in place. TileLang's `out_idx` mode returns a fresh tensor,
which would force the adapter into a `copy_` and roughly double the measured
traffic — that is what happens to the Helion candidate here, and it is an artefact
of the adapter rather than a property of the DSL. Passing `Y` explicitly avoids it,
so this number is comparable with the CUDA and Triton candidates.

Kernels are compiled per shape and cached, because the shape is a compile-time
constant in TileLang. Compilation happens on first call, which the harness performs
during warmup, so it never enters a timed sample.
"""

import tilelang
import tilelang.language as T


# TileLang bakes the element type into the compiled kernel, so the candidate has to
# honour the precision the harness handed it rather than assume one.
_TILELANG_DTYPE = {"torch.float32": "float32", "torch.bfloat16": "bfloat16", "torch.float16": "float16"}


def _dtype_of(tensor) -> str:
    name = str(tensor.dtype)
    if name not in _TILELANG_DTYPE:
        raise RuntimeError(f"no TileLang dtype for {name}")
    return _TILELANG_DTYPE[name]


# Keyed on dtype as well as shape: both are compile-time constants here.
_CACHE: dict[tuple[int, int, str], object] = {}


@tilelang.jit
def _rmsnorm(M: int, N: int, dtype: str, blk_m: int = 1):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), W: T.Tensor((N,), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=256) as bx:
            row = bx * blk_m
            # Accumulate in fp32 regardless of storage. Summing 4096 squares in
            # bf16 or fp16 loses the reduction, and the harness sizes its tolerance
            # for an fp32 accumulator -- which is what the tensor cores would do
            # anyway, so matching it is correctness rather than generosity.
            squares = T.alloc_fragment((N,), "float32")
            total = T.alloc_fragment((1,), "float32")
            T.clear(total)
            for j in T.Parallel(N):
                v = T.cast(X[row, j], "float32")
                squares[j] = v * v
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(N):
                scaled = T.cast(X[row, j], "float32") * T.rsqrt(total[0] / N + 1e-6)
                Y[row, j] = T.cast(scaled * T.cast(W[j], "float32"), dtype)
    return main


def launch_candidate(inputs, out, meta):
    dtype = _dtype_of(inputs["x"])
    key = (meta["rows"], meta["cols"], dtype)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _rmsnorm(meta["rows"], meta["cols"], dtype)
        _CACHE[key] = kernel
    kernel(inputs["x"], inputs["w"], out)
