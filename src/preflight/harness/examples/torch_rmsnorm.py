"""Torch-native RMSNorm. Not a kernel — the harness's own control.

Its job is to prove the Python harness agrees with the CUDA harness on the same
operation. If the two disagree on a correct implementation, the harness is wrong
and no verdict it produces means anything.
"""

import torch


def launch_candidate(inputs, out, meta):
    x, w = inputs["x"], inputs["w"]
    eps = meta["eps"]
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    torch.mul(x * scale, w, out=out)
