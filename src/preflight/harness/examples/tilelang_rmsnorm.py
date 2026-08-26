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

_CACHE: dict[tuple[int, int], object] = {}


@tilelang.jit
def _rmsnorm(M: int, N: int, blk_m: int = 1, dtype: str = "float32"):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), W: T.Tensor((N,), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=256) as bx:
            row = bx * blk_m
            squares = T.alloc_fragment((N,), dtype)
            total = T.alloc_fragment((1,), dtype)
            T.clear(total)
            for j in T.Parallel(N):
                squares[j] = X[row, j] * X[row, j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(N):
                Y[row, j] = X[row, j] * T.rsqrt(total[0] / N + 1e-6) * W[j]
    return main


def launch_candidate(inputs, out, meta):
    key = (meta["rows"], meta["cols"])
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _rmsnorm(meta["rows"], meta["cols"])
        _CACHE[key] = kernel
    kernel(inputs["x"], inputs["w"], out)
