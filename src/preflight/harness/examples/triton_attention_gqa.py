"""GQA in Triton: four query heads index the same KV head instead of expanding it.

Same online softmax as the other FlashAttention kernels here. The only change is one
line of index arithmetic -- the KV base pointer is derived from `bh // GROUP` rather
than `bh` -- and that line is the whole optimisation. Nothing is materialised, each
shared KV head is read by the programs that need it, and the traffic is what GQA
promises rather than what multi-head attention costs.

Worth contrasting with torch_attention_gqa.py, which expands with `repeat_interleave`
and is equally correct. The two differ only in traffic, which is precisely the kind of
difference a performance claim should have to survive.

**On precision.** `tl.dot` routes through TF32 tensor cores by default, so this kernel
does not honour an fp32 contract even when handed fp32 tensors. Its honest contracts are
`tf32` on fp32 storage and `bf16` or `fp16` on reduced storage.
"""

import triton
import triton.language as tl


@triton.jit
def _gqa_fwd(Q, K, V, O, scale, stride_qbh, stride_kbh, stride_s,
             SEQ: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    q_base = bh * stride_qbh
    # The one line that makes this GQA rather than MHA: the group shares a KV head.
    kv_base = (bh // GROUP) * stride_kbh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + q_base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=offs_m[:, None] < SEQ, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, SEQ, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + kv_base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < SEQ, other=0.0)
        v = tl.load(V + kv_base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < SEQ, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(offs_n[None, :] < SEQ, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + q_base + offs_m[:, None] * stride_s + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < SEQ)


def launch_candidate(inputs, out, meta):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    batch, heads_q, seq, head_dim = q.shape
    heads_kv = k.shape[1]
    group = heads_q // heads_kv

    qf = q.view(batch * heads_q, seq, head_dim)
    kf = k.view(batch * heads_kv, seq, head_dim)
    vf = v.view(batch * heads_kv, seq, head_dim)
    of = out.view(batch * heads_q, seq, head_dim)

    # Same shared-memory budget as the other flash kernels: at head_dim=128 in fp32 a
    # 64x64 tiling exceeds the 101376 bytes this device has.
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

    grid = (triton.cdiv(seq, block_m), batch * heads_q)
    _gqa_fwd[grid](
        qf, kf, vf, of, 1.0 / (head_dim ** 0.5),
        qf.stride(0), kf.stride(0), qf.stride(1),
        SEQ=seq, D=head_dim, GROUP=group,
        BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=stages,
    )
