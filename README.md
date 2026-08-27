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
      └── publish_kernel       approval-gated; re-verifies under the same contract
```

**The agent submits source and never submits a number.** A candidate is compiled
*into* a fixed harness it cannot see or modify, which owns the allocation, the
input distribution, the timing loop, the reference and the tolerances. The
documented ways of faking a kernel speedup are all properties of measurement code;
making that code ours does not detect them, it makes them unrepresentable.

Inside the container that is two processes, not one:

```
supervisor.py ──── holds the nonce and the output path (from stdin, never argv)
  │                the only writer of a verdict; never loads candidate code
  └── worker ───── driver.cu or driver.py, linked to or importing the candidate
                   reports numbers on a descriptor; told neither secret
```

The split is the whole guarantee rather than a detail of it. A single process cannot
both execute a candidate and be trusted to report on it, and while this was one
process, two different candidates forged a 92%-of-peak verdict without launching a
kernel — see
[the finding that invalidated the premise, twice](#the-finding-that-invalidated-the-premise-twice).

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

Every number here is the harness's. A candidate submits source and never a number:
the harness owns allocation, the input distribution, timing, the reference and the
tolerances, and the candidate cannot reach any of them.

### Infrastructure

Measured on an RTX 4090 (sm_89, driver 580.159.04, CUDA 13.2):

| Check | Result |
| --- | --- |
| TrueForge's own `sandboxProviderContractSuite` | 4/4, incl. sibling-escape isolation |
| Docker provider suites (contract + GPU + hardening) | 16/16 |
| Default `packages/trueforge` unit suite, unaffected | 273/273, 35 suites |
| `nvidia-smi` inside the sandbox | RTX 4090 visible |
| `nvcc -arch=sm_89` compile + run inside the sandbox | 921.2 / 1008.1 GB/s = 91.4% of peak |
| 5 MiB upload/download round-trip | byte-exact |
| Agent end to end: skill → sandbox → gates | admitted at 91.1% of DRAM peak, first submission |
| Gate unit suite | 35/35 |

The device ceiling is corroborated two ways: `preflight.device` derives
**1008.1 GB/s** from the CUDA driver attribute API, and an independent CUDA C
bandwidth kernel measures against the same figure.

### The kernel matrix

28 cases in one session on one GPU, with a thermal cooldown between each so the
figures are comparable — `benchmark/full_matrix.py`. A sweep run straight after
Helion's autotuner once read a third of the throughput of the same kernel idle,
which is why Helion is scheduled last and why the harness samples the SM clock and
says so when it ran below peak.

**17 admitted, 11 not, and every one of the 8 adversarial candidates rejected.**

One op, five toolchains — the comparison the gates are indifferent to, because they
adjudicate a measurement schema rather than a toolchain:

| rmsnorm | fp32, % of the 1008.1 GB/s bus |
| --- | --- |
| hand-written CUDA | 90.1% |
| TileLang | 90.7% |
| Triton | 89.6% |
| Helion (autotuned) | 89.1% |
| CuTe DSL, one warp per row | 56.4% |
| eager PyTorch | 26.2% |

Memory-bound, other ops: CUDA softmax 90.2%, CUDA silu 91.2%, TileLang silu 90.6%,
CuTe silu 88.6%, CUDA transpose 30.4%.

Compute-bound, audited against the arithmetic ceiling rather than the bus:

| kernel | declared | verdict |
| --- | --- | --- |
| Triton matmul, `input_precision="ieee"` | fp32 | 58.2% of fp32 pipelines |
| Triton matmul, TF32 | fp32 | **rejected** — correctness, timed_work |
| the same TF32 kernel | tf32 | 89.2% of tf32 pipelines |
| torch SDPA | fp32 | 29.2% |
| torch SDPA | bf16 | 93.1% |
| Triton FlashAttention | fp32 | **rejected** — correctness, timed_work |
| the same kernel | tf32 | 76.7% |
| the same kernel | bf16 | 88.9% |

Two independent sweeps of this matrix agree to within **0.3 points** on every
memory-bound case and disagree by up to **6.8 points** on the compute-bound
attention ones, which pick a different winning shape from run to run. That gap is
the argument for `variance` being a gate rather than a note, and it is the reason a
single number from a single run is not evidence — including for the numbers above.

### Five results worth reading

**One true value passes a check built to need the whole output.** `timed_work`
exists to prove the *measured* calls did the work, and it asks two things: that
something was written, and that what was written matches. Both are satisfiable by a
single element, because the error is computed over the finite elements only — a NaN
has no distance from anything. A candidate that computed correctly through warmup
and then wrote one true value with NaN everywhere else, on every timed call, was
admitted with all nine gates green. Both harnesses were already computing the flag
that catches it and throwing it away.

**Physics disproves a precision claim without needing the reference.**
`cheat_silent_bf16.py` declares fp32 and quietly computes in bf16. It is caught
three times over: 594x past the fp32 tolerance, wrong again when the output is
re-read after timing, and at 100.9 TFLOP/s against an 83.1 TFLOP/s fp32 ceiling.
The last catch needs no reference output at all — the arithmetic is impossible at
the precision claimed.

**The same kernel is honest at one precision and dishonest at another.** The Triton
FlashAttention kernel is rejected as fp32 and admitted at tf32 and bf16, because
`tl.dot` silently uses TF32. Nothing about the kernel changes. Declaring what you
actually compute is the whole difference.

**A toolchain's config is what gets measured.** A Helion config sweep on one kernel
spans **4.7x** — `block_sizes=[1]` reaches 90% of the bus, `[32]` reaches 47%.
Pinning a config measures that config.

**The harness caught a measurement bug in my own candidate.** `cute_silu.py`
reported 7.0% of the bus. That was `@cute.jit` re-tracing on every launch: the
number was JIT dispatch, not the kernel. Hoisting the compile reports **88.6%**, a
12.7x correction. A compiled CuTe function also silently accepts shapes it was not
compiled for, so the caches are keyed on shape.

### What it does not claim

- **Hopper compute ceilings are absent on purpose.** The published figures do not
  divide cleanly into a per-SM-per-clock constant that can be defended, so the gate
  returns `UNVERIFIABLE_COMPUTE` rather than a guess. An unverifiable ceiling is
  non-blocking and says so.
- **Tensor-core ceilings are a floor, not a limit.** Within one compute capability
  NVIDIA ships parts differing by exactly 2x — the RTX 3090 at 35.6 dense TF32
  TFLOPS against the A40 at 74.8, both sm_86 — and
  [states that the reason is unpublished](https://forums.developer.nvidia.com/t/tf32-tflops-of-geforce-rtx-3090-vs-a40/265828).
  Utilisation is reported against the rate verified on this hardware, but a claim is
  only refused past the widest rate any part of that capability is rated for. fp32
  keeps a hard ceiling, and that is the axis the bf16 catch above runs on.
- **fp16 is implemented and untested.** Only sm_89 hardware was available.
- **The worker's numbers are bounded, not proven.** See
  [the honest limits](#the-limit-this-does-not-clear).

### Five false accusations

A gate that flags correct work is worse than no gate, and this project has
built that gate five times. Each of these rejected an honest kernel before it was
fixed:

| what it flagged | why it was wrong |
| --- | --- |
| baseline at 1820 GB/s on a 1008 GB/s bus | 34 MB fits in 72 MB of L2; DRAM does not bound cache-resident traffic |
| fp32 attention diverging from fp64 | the anti-narrow-range input distribution spanned 2^24 in the softmax tail |
| an honest kernel's timing variance | at `repeats=10`, `(10*9)//10` indexes the last sample, so p90 *was* max |
| torch's own matmul at 2.8e-2 | pure relative error explodes where cancellation puts the reference near zero |
| a kernel at a third of its throughput | measured after 900 s of autotuning, against a ceiling that assumes peak clock |

### The limit this does not clear

Two candidates in `examples/` forged the entire measurement and were admitted at
92% of the bus without launching a kernel: `cheat_python_forge.py` read `--out` and
`--nonce` from `sys.argv` at import, and `cheat_forge_proc.cu` recovered the same
two values from `/proc/self/cmdline` inside a C++ static constructor. One process
cannot both execute a candidate and be trusted to report on it.

`supervisor.py` now takes those secrets on stdin, spawns the worker with neither,
and is the only writer of the verdict — so a worker that exits early leaves nothing
behind. Hiding the values better would not have worked: `/proc/<ppid>/cmdline` and
`/proc/<ppid>/environ` are readable from inside the container, while
`/proc/<ppid>/mem` and `ptrace` are refused, which is what makes a sibling process
the right boundary rather than a more careful flag.

What remains: the worker still reports its own numbers on a descriptor. They are
bounded — by a duration the supervisor observed from outside, and by the roofline,
variance and timed-work gates — but bounded is not proven. Proving them would move
the timing authority outside the worker entirely, measuring a known quantity of
work from the other side of the process boundary. That is a larger change than this,
and it is not done.

## Layout

| Path | Contents |
| --- | --- |
| `src/preflight/harness/supervisor.py` | the only process that never runs candidate code, and the only writer of a verdict |
| `src/preflight/harness/driver.cu`, `driver.py` | the measurement workers, one per backend, emitting the same schema |
| `src/preflight/gates.py` | adjudicates that schema — nine gates, indifferent to the toolchain |
| `src/preflight/runner.py` | isolation: container, no network, no inherited environment |
| `src/preflight/mcp_server.py` | the three tools TrueForge sees |
| `src/preflight/device.py` | hardware ceilings read from the CUDA driver attribute API |
| `src/preflight/harness/examples/` | 17 honest kernels across five toolchains, 8 adversarial, 1 merely wrong — all regression tests |
| `benchmark/full_matrix.py` | the 28-case sweep behind the table above |
| `tests/` | 35 gate tests, including the ones that fail against the pre-fix code |
| `agent/` | the saved TrueForge agent manifest |
| `docker/` | the sandbox images: CUDA, plus torch/Triton/Helion/CuTe/TileLang |
| `upstream/trueforge/` | changes to TrueForge itself, submitted upstream as [truefoundry/trueforge#467](https://github.com/truefoundry/trueforge/pull/467) |
| `docs/` | investigation notes |

## Qodo Code Review Evidence

Every substantive change goes through a pull request reviewed by Qodo before merge.

| Round | Findings | Resolution |
| --- | --- | --- |
| [#1, first pass](https://github.com/rycerzes/kernel-preflight/pull/1) | 5 High, 3 Medium | 7 fixed, 1 partially fixed with a documented scope note. [Full response](https://github.com/rycerzes/kernel-preflight/pull/1#issuecomment-5414904562) |
| [#1, review of the fixes](https://github.com/rycerzes/kernel-preflight/pull/1) | 1 High, 2 Medium | all 3 fixed |
| [#2, preflight agent](https://github.com/rycerzes/kernel-preflight/pull/2) | 6 findings | all 6 fixed. [Full response](https://github.com/rycerzes/kernel-preflight/pull/2#issuecomment-5420739404) |
| [#3, benchmark and ops guide](https://github.com/rycerzes/kernel-preflight/pull/3) | 7 findings across two rounds | all 7 fixed |
| [#4, six ops and six backends](https://github.com/rycerzes/kernel-preflight/pull/4) | 9 findings | all 9 fixed; one of them falsified the premise a second time |

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

### The finding that invalidated the premise, twice

Review of the agent PR found that a candidate could **forge its own verdict**. The
harness printed its measurement to stdout, the candidate was linked into the same
binary, and C++ static constructors run before `main`. A kernel with an empty body,
whose constructor printed a fabricated measurement and exited, was **admitted at a
plausible 91% of peak** — six gates, all green, for code that never launched
anything.

That is not a bug in a gate. It falsified the sentence at the top of this file.

The first fix was to stop taking stdout at face value: the harness had to echo a
nonce chosen after the candidate's source was fixed, and account for wall-clock time
the caller independently observed. Fabricated work cannot pay for time that never
elapsed. Later the measurement moved out of stdout entirely, into a file, which
removed the channel `cheat_forge.cu` had used.

**That fix was not enough, and review of PR #4 found why.** The nonce and the output
path were still passed in `argv`, and the candidate was still running inside the
process that wrote the verdict. Two new candidates read them straight back out:

- `cheat_python_forge.py` — the Python harness imports candidates with
  `exec_module`, so module-level code runs before anything is measured, with
  `sys.argv` intact.
- `cheat_forge_proc.cu` — a C++ static constructor is not passed `argv`, but
  `/proc/self/cmdline` contains it.

Both wrote a measurement claiming 92% of the memory bus and exited. Both were
admitted, all nine gates green. The first fix had made forgery harder to reach
without making it impossible, which is the difference between a mitigation and a
boundary.

Hiding the values better does not work, and this was measured rather than assumed:
inside the container `/proc/<ppid>/cmdline` and `/proc/<ppid>/environ` are readable,
while `/proc/<ppid>/mem` and `ptrace(PTRACE_ATTACH)` are refused with `EPERM`. So no
flag and no environment variable can carry a secret past code running there — but a
*sibling process* can hold one safely.

That is the fix. `supervisor.py` takes the nonce and the output path on stdin,
spawns the worker with neither, and is the only writer. It also stamps the wall time
it observed itself, so the provenance check that the timed loops fit inside the run
is finally made against a duration the measured code does not control. A worker that
exits early now leaves nothing behind — which the runner already treats as a
rejection, and which is exactly what both forges now do.

Both attacks are committed as regression tests. The residual limit is stated in
[what it does not claim](#what-it-does-not-claim): the worker still reports its own
numbers, bounded by an observed duration and by the roofline, variance and
timed-work gates, but bounded is not proven.

The same review round also found that the tf32 correctness bar was 128x looser than
the contract, that five of six input builders silently ignored a declared reduced
precision, and that a merely *wrong* transpose came back as a JSON parse error
instead of a correctness verdict. Nine findings, all fixed, each verified against
the code or the hardware before being acted on — one of them,
[a claim about GA10x tensor ceilings](https://forums.developer.nvidia.com/t/tf32-tflops-of-geforce-rtx-3090-vs-a40/265828),
turned out to be a real risk of *falsely accusing* correct kernels.

The earlier review also found that candidate code was being compiled and executed
**on the host**, in the process environment holding the publishing token — while
this repository was contributing a GPU container sandbox to TrueForge. Execution now
happens in a container with no network and no inherited environment.

## Notes

- Built with AI assistance (Claude Code), as permitted by the hackathon rules.
- `upstream/` is mirrored here so this repository is self-contained and reviewable.
  It is submitted upstream as
  [truefoundry/trueforge#467](https://github.com/truefoundry/trueforge/pull/467),
  raised behind [#466](https://github.com/truefoundry/trueforge/issues/466) because
  TrueForge asks for maintainer approval before code contributions. Whether they take
  it is theirs to decide; the analysis in the issue stands either way.
