"""MoE grouped GEMM in Triton: permute by expert, one GEMM per tile, scatter back.

The shape every production MoE kernel has. Tokens are sorted so each expert's rows are
contiguous, the output is tiled so no tile straddles two experts, and each tile does an
ordinary GEMM against one weight matrix. The rows are gathered and scattered through the
permutation rather than materialised, so the token tensor is touched once on each side.

The routing plan is rebuilt on every call rather than cached, deliberately. `expert` does
not change between the harness's timed samples, so caching it would be both easy and
free -- and would measure a kernel that does not pay for its own routing while the torch
reference pays for its masks every call. Comparing those two numbers would be
meaningless. Building the plan is a handful of small tensor ops against a GEMM of several
hundred GFLOP, so honesty here costs almost nothing.

The plan itself is vectorised rather than looped over experts:

* `counts` and `offsets` locate each expert's contiguous run in the sorted order.
* `tiles_per_expert` pads each run up to a whole number of BLOCK_M tiles, which is what
  guarantees a tile never spans two experts -- the invariant that makes the inner loop an
  ordinary GEMM instead of a special case.
* `tile_count` carries how many rows of the last tile in each run are real, so the
  padding is masked rather than computed and discarded.

`input_precision="ieee"` on the dot, so this honours an fp32 contract. Left at the
default it would silently use TF32 tensor cores and be rejected against the fp32
tolerance -- which is what happens to triton_matmul_tf32.py, on purpose.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def _moe_gemm_kernel(X, W, OUT, PERM, TILE_E, TILE_START, TILE_COUNT,
                     stride_x, stride_we, stride_wk, stride_o,
                     K: tl.constexpr, N: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert = tl.load(TILE_E + pid_m)
    start = tl.load(TILE_START + pid_m)
    count = tl.load(TILE_COUNT + pid_m)

    offs_m = tl.arange(0, BM)
    live = offs_m < count
    # Token indices for this tile, read through the permutation.
    rows = tl.load(PERM + start + offs_m, mask=live, other=0)

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_live = offs_n < N

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_live = offs_k < K
        a = tl.load(X + rows[:, None] * stride_x + offs_k[None, :],
                    mask=live[:, None] & k_live[None, :], other=0.0)
        b = tl.load(W + expert * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :],
                    mask=k_live[:, None] & n_live[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")

    tl.store(OUT + rows[:, None] * stride_o + offs_n[None, :],
             acc.to(OUT.dtype.element_ty), mask=live[:, None] & n_live[None, :])


def _routing_plan(expert: torch.Tensor, experts: int, block_m: int):
    """Sorted order plus a per-tile (expert, start, valid-rows) table."""
    order = torch.argsort(expert, stable=True).to(torch.int32)
    counts = torch.bincount(expert, minlength=experts)
    offsets = torch.cumsum(counts, 0) - counts

    tiles_per_expert = (counts + block_m - 1) // block_m
    tile_expert = torch.repeat_interleave(
        torch.arange(experts, device=expert.device), tiles_per_expert
    )
    # Index of each tile within its own expert's run.
    first_tile = torch.cumsum(tiles_per_expert, 0) - tiles_per_expert
    tile_local = torch.arange(int(tiles_per_expert.sum()), device=expert.device) - first_tile[tile_expert]

    tile_start = offsets[tile_expert] + tile_local * block_m
    tile_count = torch.clamp(counts[tile_expert] - tile_local * block_m, max=block_m)
    return order, tile_expert.to(torch.int32), tile_start.to(torch.int32), tile_count.to(torch.int32)


def launch_candidate(inputs, out, meta):
    x, w, expert = inputs["x"], inputs["w"], inputs["expert"]
    tokens, hidden = x.shape
    n = meta["n"]

    order, tile_e, tile_start, tile_count = _routing_plan(
        expert, meta["experts"], BLOCK_M
    )
    grid = (tile_e.numel(), triton.cdiv(n, BLOCK_N))
    _moe_gemm_kernel[grid](
        x, w, out, order, tile_e, tile_start, tile_count,
        x.stride(0), w.stride(0), w.stride(1), out.stride(0),
        K=hidden, N=n, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K,
        num_warps=4, num_stages=3,
    )
