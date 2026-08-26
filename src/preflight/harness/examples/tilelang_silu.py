"""SiLU in TileLang: purely elementwise, no reduction.

The simplest possible TileLang kernel, kept as the control that the backend
itself is wired correctly. If this fails, the problem is the integration rather
than any particular kernel.
"""

import tilelang
import tilelang.language as T

_CACHE: dict[tuple[int, int], object] = {}


@tilelang.jit
def _silu(M: int, N: int, blk_m: int = 1, blk_n: int = 256, dtype: str = "float32"):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, blk_n), T.ceildiv(M, blk_m), threads=128) as (bx, by):
            for i, j in T.Parallel(blk_m, blk_n):
                row = by * blk_m + i
                col = bx * blk_n + j
                if (row < M) and (col < N):
                    v = X[row, col]
                    Y[row, col] = v / (1.0 + T.exp(-v))
    return main


def launch_candidate(inputs, out, meta):
    key = (meta["rows"], meta["cols"])
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _silu(meta["rows"], meta["cols"])
        _CACHE[key] = kernel
    kernel(inputs["x"], out)
