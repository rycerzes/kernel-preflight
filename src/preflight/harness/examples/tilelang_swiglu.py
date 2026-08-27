"""SwiGLU in TileLang: silu(a) * b, fused.

The fusion category is where KernelBenchX finds the failures concentrate -- 72% of its
60 Fusion tasks fail across every method they evaluated -- so it is worth having in
more than one DSL. There is no reduction here, so the kernel is a flat 2-D tile map and
the only thing that matters is reading each input exactly once.

Computes the sigmoid in fp32 whatever the storage dtype: exp of a bf16 or fp16
argument quantises before the nonlinearity, which the harness's tolerance is not sized
for and which is a real source of error rather than a rounding detail.
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
def _swiglu(M: int, N: int, dtype: str, blk_m: int = 1, blk_n: int = 256):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, blk_n), T.ceildiv(M, blk_m), threads=128) as (bx, by):
            for i, j in T.Parallel(blk_m, blk_n):
                row = by * blk_m + i
                col = bx * blk_n + j
                if (row < M) and (col < N):
                    a = T.cast(A[row, col], "float32")
                    b = T.cast(B[row, col], "float32")
                    Y[row, col] = T.cast(a / (1.0 + T.exp(-a)) * b, dtype)

    return main


def launch_candidate(inputs, out, meta):
    a = inputs["a"]
    dtype = _dtype_of(a)
    key = (meta["rows"], meta["cols"], dtype)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _swiglu(meta["rows"], meta["cols"], dtype)
        _CACHE[key] = kernel
    kernel(a, inputs["b"], out)
