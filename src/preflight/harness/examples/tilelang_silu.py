"""SiLU in TileLang: purely elementwise, no reduction.

The simplest possible TileLang kernel, kept as the control that the backend
itself is wired correctly. If this fails, the problem is the integration rather
than any particular kernel.
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
def _silu(M: int, N: int, dtype: str, blk_m: int = 1, blk_n: int = 256):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, blk_n), T.ceildiv(M, blk_m), threads=128) as (bx, by):
            for i, j in T.Parallel(blk_m, blk_n):
                row = by * blk_m + i
                col = bx * blk_n + j
                if (row < M) and (col < N):
                    # Compute in fp32 whatever the storage is: exp of a reduced-precision
                    # value quantises the exponent argument before the sigmoid, which the
                    # harness's tolerance is not sized for.
                    v = T.cast(X[row, col], "float32")
                    Y[row, col] = T.cast(v / (1.0 + T.exp(-v)), dtype)
    return main


def launch_candidate(inputs, out, meta):
    dtype = _dtype_of(inputs["x"])
    key = (meta["rows"], meta["cols"], dtype)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _silu(meta["rows"], meta["cols"], dtype)
        _CACHE[key] = kernel
    kernel(inputs["x"], out)
