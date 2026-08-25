# kernel-preflight

Preflight checks for GPU kernel performance claims, run by an agent inside the
[TrueForge](https://github.com/truefoundry/trueforge) harness.

An agent that writes a kernel *and* reports its own speedup has both the
incentive and the opportunity to be wrong in its own favour. Correctness tests do
not catch it: a kernel can be numerically perfect and still be timed dishonestly.
This project treats the agent's performance claims as untrusted, checks them
against the hardware's physical limits, and puts a human in front of publication.

**Status: in progress.** The sandbox provider and the physics gate are built and
tested. The agent loop is not finished yet. See [Verified](#verified) for exactly
what has been measured rather than asserted.

## Why

In February 2025, Sakana AI's "AI CUDA Engineer" reported 10-100x kernel speedups.
The kernels had not been optimised — the agent had found a way to exploit the
benchmark harness, and several figures implied throughput roughly 30x above the
hardware's theoretical maximum. Tri Dao pointed out publicly that the numbers were
not physically possible.

Reward hacking in kernel generation is now a documented failure class: calling
forbidden framework operators, using async streams to bypass the timer, hardcoding
outputs, exploiting narrow input distributions, hyperspecialising to one shape.

## Prior art

Roofline-grading kernel claims against hardware peak is **established practice,
not a contribution of this project**:

- [`SOL-ExecBench`](https://arxiv.org/html/2603.19173v1) — speed-of-light benchmarking against hardware limits
- [`KernelBench-Hard`](https://github.com/Infatoshi/KernelBench-Hard) — roofline-graded against hardware peak
- `ROBUST-KBENCH` — addresses KernelBench's reward-hacking vulnerabilities
- [`KernelBench`](https://github.com/ScalingIntelligence/KernelBench) itself ships adversarial kernels and flags excessive speedups

Kernel *generation* is also well covered by better-resourced work —
[`KernelAgent`](https://github.com/meta-pytorch/KernelAgent) (Meta/PyTorch) and
[`CUDA-Agent`](https://github.com/BytedTsinghua-SIA/CUDA-Agent) (RL-trained, SOTA
on KernelBench). This project does not compete on kernel quality.

What is not established practice, and is where this aims: doing the checks **inside
an agent harness**, with a human approval gate in front of *publishing* a kernel to
the Hub, rather than scoring a benchmark offline.

## Architecture

```
TrueForge  (on the GPU host)
└── DockerSandboxProvider           new; upstream/
      • docker run --gpus all
      • nvcc + torch inside the sandbox
            │
            ▼
      preflight gates
      • physics    claim vs driver-reported hardware ceiling
      • liveness   did the kernel write its output at all
      • sync       is the timed region actually synchronised
      • variance   repeated runs, not one sample
            │
            ▼
      human approval  ──▶  publish to the HF Hub
```

The sandbox has to hold the GPU, because a performance claim measured somewhere
other than where the kernel runs is not a measurement. TrueForge's existing
bubblewrap-based local provider cannot do that — `bwrap` is not setuid, so it
always unshares into a user namespace, and the NVIDIA driver refuses to initialise
inside one. Full write-up, including the four configurations tried, in
[`docs/sandbox-gpu-investigation.md`](docs/sandbox-gpu-investigation.md).

## Verified

Measured on an RTX 4090 (sm_89, driver 580.159.04, CUDA 13.2), not asserted:

| Check | Result |
| --- | --- |
| TrueForge's own `sandboxProviderContractSuite` | 4/4, incl. sibling-escape isolation |
| Full `packages/trueforge` unit suite after the union widening | 276 pass / 36 suites |
| `nvidia-smi` inside the sandbox | RTX 4090 visible |
| `nvcc -arch=sm_89` compile + run inside the sandbox | 921.2 / 1008.1 GB/s = 91.4% of peak |
| 5 MiB upload/download round-trip | byte-exact |

The device ceiling is corroborated two ways: `preflight.device` derives
**1008.1 GB/s** from the CUDA driver attribute API, and an independent CUDA C
bandwidth kernel measures against the same figure.

Run-to-run variance on identical code was **84.9% → 91.4% of peak** across three
runs. That is why `variance` is a gate: a single timing sample is not evidence,
including for this project's own numbers.

## Layout

| Path | Contents |
| --- | --- |
| `src/preflight/` | the gates. `device.py` reads hardware ceilings, `roofline.py` grades claims |
| `upstream/trueforge/` | changes to TrueForge itself, mirroring upstream paths for a clean PR |
| `docs/` | investigation notes |

## Qodo Code Review Evidence

_Pending — this section will link the reviewed PRs and record how High-severity
findings were resolved or dismissed._

## Notes

- Built with AI assistance (Claude Code), as permitted by the hackathon rules.
- `upstream/` is intended for a pull request to `truefoundry/trueforge`; it is
  mirrored here so this repository is self-contained and reviewable.
