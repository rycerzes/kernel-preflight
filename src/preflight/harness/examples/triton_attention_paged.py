"""Paged decode attention in Triton: walk the block table, never materialise the cache.

One program per (batch, head). It reads the sequence's block table entry by entry,
resolves each logical block to a physical one, streams `block_size` positions from
wherever that lands in the pool, and folds them into a running softmax. Nothing
contiguous is ever built, which is the entire point of paging and the entire difference
from torch_attention_paged.py.

The addressing is the work. A physical block holds `block_size` positions for *all*
heads interleaved, so the stride from one position to the next within a block is
`heads * head_dim`, not `head_dim`. Getting that wrong produces a kernel that is correct
when `heads == 1` and silently mixes heads otherwise -- the sort of bug a single-shape
test misses, and the reason the reference here builds the contiguous cache by an explicit
permute rather than a reshape that would hide the same mistake.

Online softmax over blocks rather than over tiles: `block_size` is 16, so the running
maximum is updated every 16 positions and the accumulator rescaled with it. That is more
rescaling than a prefill kernel does per element, and it is unavoidable -- the block
boundary is set by the allocator, not by the tile size that would suit the arithmetic.

Loads are masked on the head dimension only; every block in the table is full here
because sequence lengths are uniform, so there is no partial-block tail to handle. A
serving kernel needs one.
"""

import triton
import triton.language as tl


@triton.jit
def _paged_decode(Q, KC, VC, TABLE, O, scale,
                  stride_kb, stride_kp, stride_kh,
                  stride_tb, stride_qh, stride_oh,
                  NBLOCKS, HEADS: tl.constexpr, D: tl.constexpr,
                  BS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    offs_d = tl.arange(0, D)
    offs_p = tl.arange(0, BS)

    q = tl.load(Q + (b * HEADS + h) * stride_qh + offs_d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for i in range(NBLOCKS):
        physical = tl.load(TABLE + b * stride_tb + i)
        # Positions within a block are strided by heads * head_dim, because a block
        # stores every head for those positions.
        base = physical * stride_kb + h * stride_kh
        k = tl.load(KC + base + offs_p[:, None] * stride_kp + offs_d[None, :]).to(tl.float32)
        v = tl.load(VC + base + offs_p[:, None] * stride_kp + offs_d[None, :]).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1) * scale

        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(O + (b * HEADS + h) * stride_oh + offs_d, (acc / l_i).to(O.dtype.element_ty))


def launch_candidate(inputs, out, meta):
    q = inputs["q"]
    kc, vc = inputs["k_cache"], inputs["v_cache"]
    table = inputs["block_table"]
    batch, heads = meta["batch"], meta["heads"]
    head_dim, block_size = meta["head_dim"], meta["block_size"]
    blocks_per_seq = meta["blocks_per_seq"]

    qf = q.view(batch * heads, head_dim)
    of = out.view(batch * heads, head_dim)

    _paged_decode[(batch, heads)](
        qf, kc, vc, table, of, 1.0 / (head_dim ** 0.5),
        kc.stride(0), kc.stride(1), kc.stride(2),
        table.stride(0), qf.stride(0), of.stride(0),
        blocks_per_seq, HEADS=heads, D=head_dim, BS=block_size,
        num_warps=4,
    )
