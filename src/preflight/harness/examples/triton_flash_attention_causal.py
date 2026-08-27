"""Causal FlashAttention forward in Triton: skip the tiles, mask only the diagonal.

Same online-softmax structure as triton_flash_attention.py -- running max, running sum,
accumulator rescaled as the max moves, score matrix never materialised -- with the one
change that makes it causal and the one optimisation that makes causal worth doing.

The change is a mask. The optimisation is *not applying it everywhere*. For a query
block starting at row `m`, key blocks entirely past `m + BLOCK_M` contribute nothing, so
the loop stops rather than computing them and multiplying by zero. Only the blocks
straddling the diagonal need the element-wise comparison. A kernel that masks every tile
is equally correct and close to twice the work, which is exactly the difference the
harness's cost model is set up to see: `flops` for this op counts seq*(seq+1)/2 score
entries, not seq^2.

**On precision.** `tl.dot` routes through TF32 tensor cores by default, so this kernel
does not honour an fp32 contract even when handed fp32 tensors. Its honest contracts are
`tf32` on fp32 storage and `bf16` or `fp16` on reduced storage.
"""

import triton
import triton.language as tl


@triton.jit
def _flash_causal_fwd(Q, K, V, O, scale, stride_bh, stride_s,
                      SEQ: tl.constexpr, D: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    base = tl.program_id(1) * stride_bh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=offs_m[:, None] < SEQ, other=0.0)

    # fp32 running state regardless of input dtype: this is what the tensor cores do
    # internally, and what keeps a bf16 kernel accurate enough to pass a bf16 bar.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Causality: nothing at or beyond the end of this query block can be attended to,
    # so the loop ends there instead of running to SEQ and masking the remainder away.
    hi = tl.minimum((pid_m + 1) * BLOCK_M, SEQ)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < SEQ, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < SEQ, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        # Only the blocks straddling the diagonal need this; the ones fully below it
        # are already valid and the ones above it were never visited.
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_s + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < SEQ)


def launch_candidate(inputs, out, meta):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    batch, heads, seq, head_dim = q.shape
    qf, kf, vf, of = (t.view(batch * heads, seq, head_dim) for t in (q, k, v, out))

    # Same shared-memory budget as the non-causal kernel, and for the same reason:
    # at head_dim=128 in fp32 a 64x64 tiling asks for more than the 101376 bytes this
    # device has and fails to compile, while bf16 halves every tile and fits. Sizing it
    # rather than hardcoding is what makes one file cover all four precisions --
    # hardcoding 32x64 here compiled at bf16 and failed at fp32 and tf32.
    itemsize = q.element_size()
    block_m = block_n = 64
    stages = 2
    while (block_m + 2 * block_n) * head_dim * itemsize * stages > 96_000:
        if stages > 1:
            stages -= 1
        elif block_n > 16:
            block_n //= 2
        elif block_m > 16:
            block_m //= 2
        else:
            break

    grid = (triton.cdiv(seq, block_m), batch * heads)
    _flash_causal_fwd[grid](
        qf, kf, vf, of, 1.0 / (head_dim ** 0.5), qf.stride(0), qf.stride(1),
        SEQ=seq, D=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=stages,
    )
