"""MoE grouped GEMM the reference way: one GEMM per expert, boolean-masked.

How the operation is written before anyone optimises it, and it is not a straw man --
the per-expert GEMMs themselves go to cuBLAS. What it pays for is the routing: eight
boolean masks over the token dimension, eight gathers to collect each expert's rows, and
eight scatters to put the results back, all of which touch the full token tensor.
"""

import torch


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    w = inputs["w"]
    expert = inputs["expert"]
    for e in range(meta["experts"]):
        rows = expert == e
        if bool(rows.any()):
            out[rows] = x[rows] @ w[e]
