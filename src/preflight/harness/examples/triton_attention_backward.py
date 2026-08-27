"""FlashAttention backward in Triton: three kernels, nothing seq x seq ever written.

The backward pass needs the attention probabilities P, and the forward did not keep
them. A materialising implementation stores the seq x seq matrix and reads it back; this
recomputes S from Q and K wherever it is needed, which is what makes it Flash and what
costs the extra 2*seq^2 terms the harness's `flops` already accounts for.

Three passes, because the data dependencies genuinely differ:

1. **preprocess** -- recompute the forward per query block to recover `L`, the row
   logsumexp, and `D = rowsum(dO * O)`. `L` is what lets the other two kernels rebuild P
   from S without re-deriving the row maximum, and `D` is the softmax Jacobian's coupling
   term. Both are one value per query row, so this is the only place seq^2 work produces
   a seq-sized result.
2. **dk/dv** -- outer loop over key blocks, inner over query blocks. dV and dK accumulate
   across queries, so the key block has to be the one held in registers.
3. **dq** -- outer loop over query blocks, inner over key blocks. dQ accumulates across
   keys, so the loops invert.

That inversion is the reason this is three kernels and not one. dV needs every query for
a fixed key; dQ needs every key for a fixed query. Nothing can hold both.

Which means this recomputes S three times where two would do, and the harness charges for
two -- folding dQ into the dk/dv pass with atomics is the standard way to avoid the third,
and this does not do it. So the number here is lower than a fully fused backward would
reach, by roughly the ratio of 18 recompute-units to 14. That gap is real and is meant to
be visible rather than absorbed into the cost model.

The maths, with S' = scale * Q K^T:

    P  = exp(S' - L)
    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - D)
    dQ = scale * dS K
    dK = scale * dS^T Q

`input_precision="ieee"` on every dot, so this honours an fp32 contract rather than
silently using TF32 tensor cores.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_preprocess(Q, K, V, DO, L, D, scale, stride_bh, stride_s,
                    SEQ: tl.constexpr, HD: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    base = tl.program_id(1) * stride_bh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)
    m_live = offs_m < SEQ

    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=m_live[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HD], dtype=tl.float32)
    for start_n in range(0, SEQ, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_live = offs_n < SEQ
        k = tl.load(K + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=n_live[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=n_live[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where(n_live[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    o = acc / l_i[:, None]
    do = tl.load(DO + base + offs_m[:, None] * stride_s + offs_d[None, :],
                 mask=m_live[:, None], other=0.0).to(tl.float32)

    bh = tl.program_id(1)
    tl.store(L + bh * SEQ + offs_m, m_i + tl.log(l_i), mask=m_live)
    tl.store(D + bh * SEQ + offs_m, tl.sum(do * o, 1), mask=m_live)


@triton.jit
def _bwd_dkdv(Q, K, V, DO, L, D, DK, DV, scale, stride_bh, stride_s,
              SEQ: tl.constexpr, HD: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * stride_bh
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HD)
    n_live = offs_n < SEQ

    k = tl.load(K + base + offs_n[:, None] * stride_s + offs_d[None, :],
                mask=n_live[:, None], other=0.0)
    v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :],
                mask=n_live[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HD], dtype=tl.float32)

    for start_m in range(0, SEQ, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_live = offs_m < SEQ
        q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                    mask=m_live[:, None], other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * stride_s + offs_d[None, :],
                     mask=m_live[:, None], other=0.0)
        l_m = tl.load(L + bh * SEQ + offs_m, mask=m_live, other=0.0)
        d_m = tl.load(D + bh * SEQ + offs_m, mask=m_live, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        p = tl.exp(s - l_m[:, None])
        p = tl.where(m_live[:, None] & n_live[None, :], p, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - d_m[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee") * scale

    tl.store(DK + base + offs_n[:, None] * stride_s + offs_d[None, :],
             dk.to(DK.dtype.element_ty), mask=n_live[:, None])
    tl.store(DV + base + offs_n[:, None] * stride_s + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=n_live[:, None])


@triton.jit
def _bwd_dq(Q, K, V, DO, L, D, DQ, scale, stride_bh, stride_s,
            SEQ: tl.constexpr, HD: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * stride_bh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)
    m_live = offs_m < SEQ

    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                mask=m_live[:, None], other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * stride_s + offs_d[None, :],
                 mask=m_live[:, None], other=0.0)
    l_m = tl.load(L + bh * SEQ + offs_m, mask=m_live, other=0.0)
    d_m = tl.load(D + bh * SEQ + offs_m, mask=m_live, other=0.0)

    dq = tl.zeros([BLOCK_M, HD], dtype=tl.float32)
    for start_n in range(0, SEQ, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_live = offs_n < SEQ
        k = tl.load(K + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=n_live[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :],
                    mask=n_live[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        p = tl.exp(s - l_m[:, None])
        p = tl.where(m_live[:, None] & n_live[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - d_m[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale

    tl.store(DQ + base + offs_m[:, None] * stride_s + offs_d[None, :],
             dq.to(DQ.dtype.element_ty), mask=m_live[:, None])


def launch_candidate(inputs, out, meta):
    q, k, v, do = inputs["q"], inputs["k"], inputs["v"], inputs["do"]
    batch, heads, seq, head_dim = q.shape
    bh = batch * heads

    qf, kf, vf, dof = (t.view(bh, seq, head_dim) for t in (q, k, v, do))
    dq, dk, dv = (out[i].view(bh, seq, head_dim) for i in range(3))

    # One value per query row, so this scratch is seq-sized rather than seq^2.
    l = torch.empty((bh, seq), device=q.device, dtype=torch.float32)
    d = torch.empty((bh, seq), device=q.device, dtype=torch.float32)

    # A tighter shared-memory budget than the forward kernels need, because the dk/dv
    # kernel holds far more live at once: k, v and the dk and dv accumulators over the
    # key block, plus q and do over the query block, plus the seq x seq intermediates
    # p, dp and ds. Sizing it as three tiles the way the forward does asked for 115712
    # bytes against this device's 101376 and failed to compile -- and only on fp32 at
    # head_dim 64, because bf16 halves every tile and fits either way.
    itemsize = q.element_size()
    block_m = block_n = 64
    while (2 * block_m + 4 * block_n) * head_dim * itemsize > 72_000:
        if block_n > 16:
            block_n //= 2
        elif block_m > 16:
            block_m //= 2
        else:
            break

    scale = 1.0 / (head_dim ** 0.5)
    stride_bh, stride_s = qf.stride(0), qf.stride(1)
    common = dict(SEQ=seq, HD=head_dim, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4)

    _bwd_preprocess[(triton.cdiv(seq, block_m), bh)](
        qf, kf, vf, dof, l, d, scale, stride_bh, stride_s, **common)
    _bwd_dkdv[(triton.cdiv(seq, block_n), bh)](
        qf, kf, vf, dof, l, d, dk, dv, scale, stride_bh, stride_s, **common)
    _bwd_dq[(triton.cdiv(seq, block_m), bh)](
        qf, kf, vf, dof, l, d, dq, scale, stride_bh, stride_s, **common)
