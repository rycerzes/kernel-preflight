# kernel-preflight

Preflight checks for GPU kernel performance claims, run by an agent inside the
[TrueForge](https://github.com/truefoundry/trueforge) harness.

An agent that writes a kernel *and* reports its own speedup has both the
incentive and the opportunity to be wrong in its own favour. Correctness tests do
not catch it: a kernel can be numerically perfect and still be timed dishonestly.
This project treats the agent's performance claims as untrusted, checks them
against the hardware's physical limits, and puts a human in front of publication.

**Status: working end to end.** The agent reads a skill, drafts in a GPU sandbox,
submits to the gates and stops for approval before publishing. See
[Verified](#verified) for what has been measured rather than asserted.

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
├── DockerSandboxProvider ──── the agent's workspace: GPU, nvcc, skills
│                              (new; contributed in upstream/)
└── MCP: kernel-preflight ──── the adjudicator, outside the agent's control
      ├── device_spec          hardware ceilings from the CUDA driver
      ├── preflight_kernel     compile → measure → gate, in isolation
      └── publish_kernel       approval-gated; re-verifies before publishing
```

**The agent submits source and never submits a number.** A candidate is compiled
*into* a fixed harness it cannot see or modify, which owns the allocation, the
input distribution, the timing loop, the reference and the tolerances. The
documented ways of faking a kernel speedup are all properties of measurement code;
making that code ours does not detect them, it makes them unrepresentable.

The gates:

| gate | catches |
| --- | --- |
| `wellformed` | the measurement did not come from the harness |
| `provenance` | the result was fabricated before the harness ran |
| `correctness` | it does not compute the operation |
| `timed_work` | it computed during warmup and skipped the measured calls |
| `liveness` | it never wrote the output — instant, and otherwise brilliant |
| `input_sensitivity` | it writes a constant, ignoring its input |
| `shape_consistency` | it is tuned to one shape, i.e. a lookup table |
| `variance` | the timing is too unstable to mean anything |
| `roofline` | the implied throughput is physically impossible |

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
| Docker provider suites (contract + GPU + hardening) | 16/16 |
| Default `packages/trueforge` unit suite, unaffected | 273/273, 35 suites |
| `nvidia-smi` inside the sandbox | RTX 4090 visible |
| `nvcc -arch=sm_89` compile + run inside the sandbox | 921.2 / 1008.1 GB/s = 91.4% of peak |
| 5 MiB upload/download round-trip | byte-exact |
| Agent end to end: skill → sandbox → gates | admitted at 91.1% of DRAM peak, first submission |
| Adversarial kernels (`noop`, `cached`, `forge`) | all three rejected, each by a different gate |

The device ceiling is corroborated two ways: `preflight.device` derives
**1008.1 GB/s** from the CUDA driver attribute API, and an independent CUDA C
bandwidth kernel measures against the same figure.

Run-to-run variance on identical code was **84.9% → 91.4% of peak** across three
runs. That is why `variance` is a gate: a single timing sample is not evidence,
including for this project's own numbers.

## Layout

| Path | Contents |
| --- | --- |
| `src/preflight/` | `harness/driver.cu` measures, `gates.py` adjudicates, `runner.py` isolates, `mcp_server.py` serves |
| `src/preflight/device.py` | hardware ceilings read from the CUDA driver |
| `src/preflight/harness/examples/` | 17 honest kernels across five toolchains and 5 adversarial ones, kept as regression tests |
| `agent/` | the saved TrueForge agent manifest |
| `docker/` | the sandbox image: CUDA plus the Python the bootstrap needs |
| `upstream/trueforge/` | changes to TrueForge itself, mirroring upstream paths for a clean PR |
| `docs/` | investigation notes |

## Qodo Code Review Evidence

Every substantive change goes through a pull request reviewed by Qodo before merge.

| Round | Findings | Resolution |
| --- | --- | --- |
| [#1, first pass](https://github.com/rycerzes/kernel-preflight/pull/1) | 5 High, 3 Medium | 7 fixed, 1 partially fixed with a documented scope note. [Full response](https://github.com/rycerzes/kernel-preflight/pull/1#issuecomment-5414904562) |
| [#1, review of the fixes](https://github.com/rycerzes/kernel-preflight/pull/1) | 1 High, 2 Medium | all 3 fixed |
| [#2, preflight agent](https://github.com/rycerzes/kernel-preflight/pull/2) | 6 findings | all 6 fixed. [Full response](https://github.com/rycerzes/kernel-preflight/pull/2#issuecomment-5420739404) |

The second round reviewed the fixes themselves and found three second-order bugs,
which is the review loop doing its job:

- **Test deletes active sandboxes** (High) — the reaper test marked every labelled
  container stale, so it would delete live sandboxes belonging to another test
  worker or a running dev server on the same daemon. Fixed by making ownership
  scoped: containers carry a scope label, `reapStale` requires one, and a negative
  cutoff is now rejected outright. The regression test proves a bystander scope
  survives a reap.
- **Exit 124 conflated with timeout** (Medium) — GNU `timeout` exits 124 when it
  fires *and* passes a command's own 124 straight through, confirmed by
  experiment. Exit codes now ride in the successful envelope per the provider
  contract, and `timeout --verbose` supplies the stderr diagnostic that
  distinguishes a real timeout.
- **Oversized download reported missing** (Medium) — collapsing the size check and
  the read into one invocation removed a check/use window, but a file can still
  grow mid-`cat`, making the streaming cap the last line of defence. It killed the
  child without recording why, so an oversized file surfaced as missing.

Checking that round also surfaced a packaging problem Qodo had not raised: the GPU
test did not match upstream's `contract.test.ts$` ignore pattern, so it sat in the
default unit run and would have failed on any machine without Docker. Renamed to
follow the convention — infra-dependent tests are opt-in.

Findings were verified independently before being acted on, rather than accepted
on the reviewer's word. Two of the eight needed the claim narrowed:

- **Timeout leaves workload running** — real, and reproduced by hand
  (`timeout 2s sh -c "sleep 120"` → exit 124 in 2.0s, workload gone from `ps`).
  Fixed for the foreground workload; a deliberately detached child still outlives
  the bound, because `timeout` signals its child and not the process group. That
  residual is bounded by container removal and `reapStale`, and is documented at
  the call site rather than papered over.
- **Code mode admitted then crashes** — narrower than described, because
  `Sandbox.ts` gates Code Mode lazily via `codeModeEnv`. The failure is now a
  typed `CodeModeUnsupportedError`; full capability negotiation is out of scope
  for this PR.

The highest-impact finding was **cross-turn sandbox reuse**: the server builds a
fresh provider per turn and hands it a sandbox id carried over from an earlier
turn, so the original instance-local container map was empty exactly when it was
needed. Container names are now derived from the sandbox id.

Verifying the fixes also caught two bugs in the fixes themselves — a lookup left
keyed on the wrong tuple field, and a regression test whose `grep` matched its own
argv. Both are recorded in the PR response, because "the test passed" and "the fix
works" are different claims.

### The finding that invalidated the premise

Review of the agent PR found that a candidate could **forge its own verdict**. The
harness prints its measurement, but the candidate is linked into the same binary
and C++ static constructors run before `main`. A kernel with an empty body, whose
constructor printed a fabricated measurement and exited, was **admitted at a
plausible 91% of peak** — six gates, all green, for code that never launched
anything.

That is not a bug in a gate. It falsified the sentence at the top of this file.

It is fixed by refusing to take stdout at face value: the harness must echo a
nonce chosen after the candidate's source was fixed, and must account for
wall-clock time the caller independently observed. Fabricated work cannot pay for
time that never elapsed. The attack is committed as
`src/preflight/harness/examples/cheat_forge.cu` so it stays a regression test.

The same review found that candidate code was being compiled and executed **on the
host**, in the process environment holding the publishing token — while this
repository was contributing a GPU container sandbox to TrueForge. Execution now
happens in a container with no network and no inherited environment.

## Notes

- Built with AI assistance (Claude Code), as permitted by the hackathon rules.
- `upstream/` is intended for a pull request to `truefoundry/trueforge`; it is
  mirrored here so this repository is self-contained and reviewable.
