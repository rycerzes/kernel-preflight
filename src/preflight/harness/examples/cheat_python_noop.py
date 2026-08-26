"""Adversarial: does nothing. The Python-backend twin of cheat_noop.cu.

Present to confirm the gates are backend-agnostic — they adjudicate a measurement
schema, so a cheat should be caught identically whether it was written in CUDA,
Triton or torch.
"""


def launch_candidate(inputs, out, meta):
    return
