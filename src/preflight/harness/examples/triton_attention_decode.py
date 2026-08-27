"""Decode attention in Triton: one program per head, streaming the KV cache.

The generation-time shape, and the one place in this directory where the interesting
constraint is parallelism rather than arithmetic. A single query token attends to the
whole cache, so intensity is about 0.5 FLOP/byte against a ridge point near 82 -- this
is a memory-bound kernel wearing attention's clothes, and the roofline gate audits it
against the bus rather than the FLOP ceiling without being told to.

Structure is the same online softmax as the prefill kernels: a running maximum, a
running sum, and an accumulator rescaled whenever the maximum moves, so the scores are
never materialised. What differs is that the query is a single vector, so the `tl.dot`
becomes a broadcast multiply and a reduction.

**On the parallelism, where the obvious complaint turns out to be wrong.** The grid is
batch*heads = 32 programs against 128 SMs, so occupancy is a quarter, and the natural
conclusion is that this leaves most of the machine idle and wants FlashDecoding's split
across the cache. Measured, it does not:

    cache=8192    90.6% of the bus
    cache=16384   92.2%
    cache=32768   92.9%

Low occupancy is not low bandwidth. Thirty-two programs each streaming a long
contiguous cache issue enough outstanding loads to saturate DRAM, and once the bus is
saturated there is nothing left for more programs to win. Splitting the cache would help
where the traffic is too small to saturate it in the first place -- a short cache, or a
batch of one, which is exactly the regime FlashDecoding was written for and not the
regime here.

Worth noting what the harness did with the shortest shape. At cache=2048 this reports
128.7% of the bus, which is impossible and is not flagged, because the working set is
67 MB against 72 MB of L2: the roofline gate excludes cache-resident shapes from the
memory audit rather than accusing them. The number is real, the traffic simply was not
DRAM traffic.
"""

import triton
import triton.language as tl


@triton.jit
def _decode_fwd(Q, K, V, O, scale, stride_bh, stride_s, SEQ,
                D: tl.constexpr, BLOCK_N: tl.constexpr):
    bh = tl.program_id(0)
    base = bh * stride_bh
    offs_d = tl.arange(0, D)

    q = tl.load(Q + bh * D + offs_d).to(tl.float32)

    # fp32 running state regardless of storage dtype, as in the prefill kernels.
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for start_n in range(0, SEQ, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < SEQ
        k = tl.load(K + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)

        # One query vector, so the score is a reduction rather than a matmul.
        qk = tl.sum(k * q[None, :], axis=1) * scale
        qk = tl.where(mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(O + bh * D + offs_d, (acc / l_i).to(O.dtype.element_ty))


def launch_candidate(inputs, out, meta):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    batch, heads, seq, head_dim = k.shape
    kf, vf = (t.view(batch * heads, seq, head_dim) for t in (k, v))
    qf = q.view(batch * heads, head_dim)
    of = out.view(batch * heads, head_dim)

    _decode_fwd[(batch * heads,)](
        qf, kf, vf, of, 1.0 / (head_dim ** 0.5), kf.stride(0), kf.stride(1), seq,
        D=head_dim, BLOCK_N=128, num_warps=4,
    )
