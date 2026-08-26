"""The same TF32 kernel as triton_matmul_tf32_declared.py, but declaring its contract.

Submitted with precision="tf32". Identical to triton_matmul.py except that it takes Triton's default `tl.dot`
behaviour, which routes through TF32 tensor cores. TF32 keeps 10 mantissa bits
against fp32's 23, so results carry roughly 5e-3 relative error.

Declared, it is admitted: the correctness tolerance widens to match TF32's 10
mantissa bits, and the roofline gate audits against the TF32 tensor-core ceiling
rather than the FP32 pipelines.

The pair is the point. The gate does not object to reduced precision — it objects
to reduced precision that was not declared. Same kernel, same numbers, different
verdict, because one of them told the truth about what it was computing.
"""

import triton
import triton.language as tl


@triton.jit
def _matmul_tf32_declared(a_ptr, b_ptr, c_ptr, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)  # default precision: TF32 tensor cores

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def launch_candidate(inputs, out, meta):
    a, b = inputs["a"], inputs["b"]
    M, K, N = meta["m"], meta["k"], meta["n"]
    BLOCK_M = BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_tf32_declared[grid](a, b, out, M, N, K,
                       BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
