"""FlashAttention forward in Triton: tiled, with online softmax.

The operation the kernel-authoring ecosystem actually cares about, and the one
this harness could not measure until it grew reduced-precision support.

What makes it Flash rather than plain attention is that the SEQ x SEQ score matrix
is never materialised. Query blocks are tiled, key and value blocks are streamed,
and the softmax is maintained incrementally with a running max and running sum.
Compulsory DRAM traffic is therefore O(SEQ * D) for Q, K, V and O — which is what
the harness already models — rather than the O(SEQ^2) a naive implementation pays
writing and re-reading scores. Attention at these shapes is compute-bound anyway,
so the FLOP ceiling is what binds.

Accumulation is fp32 regardless of input dtype, which is why a bf16 submission is
held to bf16 *quantisation* error rather than to bf16 accumulation drift.

**On precision.** `tl.dot` routes through TF32 tensor cores by default, so this
kernel does not honour an fp32 contract even when handed fp32 tensors: measured
against the fp64 reference it lands around 3% relative error, which is TF32
territory and roughly 900x the fp32 tolerance. Submitted as `fp32` it is rejected,
and correctly. Its honest contracts are `tf32` on fp32 storage and `bf16` on bf16
storage.
"""

import triton
import triton.language as tl


@triton.jit
def _flash_fwd(Q, K, V, O, scale, stride_bh, stride_s,
               SEQ: tl.constexpr, D: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    base = tl.program_id(1) * stride_bh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=offs_m[:, None] < SEQ, other=0.0)

    # Online softmax state. Held in fp32 whatever the inputs are, which is what
    # tensor cores do internally and what keeps a bf16 kernel accurate enough to
    # be judged on quantisation rather than on drift.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, SEQ, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < SEQ, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=offs_n[:, None] < SEQ, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(offs_n[None, :] < SEQ, qk, float("-inf"))

        # Rescale the running accumulator to the new maximum instead of keeping
        # the scores around: this is the whole trick.
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
    batch, heads = meta["batch"], meta["heads"]
    seq, head_dim = meta["seq"], meta["head_dim"]
    # Collapse batch and head into one axis; both are contiguous, so the views
    # share storage and writing through `of` writes `out`.
    qf, kf, vf, of = (t.view(batch * heads, seq, head_dim) for t in (q, k, v, out))

    # Tiles are sized against the shared-memory budget, which is what actually
    # binds: the query, key and value tiles plus the pipeline's buffers must fit in
    # 101376 bytes on this device. At head_dim=128 in fp32 a 64x64 tiling asks for
    # 156672 and fails to compile; bf16 halves every tile and fits comfortably,
    # which is why the failure appeared only on the fp32 submission.
    #
    # Sized rather than guessed: roughly (BLOCK_M + 2 * BLOCK_N) * D * itemsize,
    # multiplied by the pipeline depth.
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
    _flash_fwd[grid](qf, kf, vf, of, 1.0 / (head_dim ** 0.5),
                     qf.stride(0), qf.stride(1),
                     SEQ=seq, D=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
                     num_warps=4, num_stages=stages)
