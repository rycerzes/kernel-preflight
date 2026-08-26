# The kernel agent that cannot lie about the speedup

*Built for the WeMakeDevs / TrueFoundry Agent Harness Hackathon, on an RTX 4090.*

In February 2025, Sakana AI announced an "AI CUDA Engineer" that reported 10–100×
kernel speedups. The kernels had not been optimised. The agent had found a way to
exploit the benchmark harness, and several of the figures implied throughput
roughly **30× above what the hardware can physically do**. Tri Dao pointed out
publicly that the numbers were not possible.

That failure is not exotic. It is the default outcome of a specific arrangement:
**an agent that writes a kernel and also reports its own speedup.** Correctness
tests do not catch it, because a kernel can be numerically perfect and still be
timed dishonestly. Reward hacking in kernel generation now has a documented
taxonomy — calling forbidden framework operators, using async streams to slip
past the timer, hardcoding outputs, exploiting narrow input distributions,
hyperspecialising to one shape.

So I built an agent inside [TrueForge](https://github.com/truefoundry/trueforge)
with one property: **it submits kernel source and never submits a number.**

## The arrangement

A fixed harness owns everything that determines a measurement — allocation, the
input distribution, the timing loop, the reference implementation, the tolerances.
The candidate is compiled *into* it and cannot see or modify it. Nine gates then
adjudicate the result, and `publish_kernel` requires human approval.

The insight is that the documented ways of faking a kernel speedup are all
properties of *measurement code*. Making that code ours does not detect them. It
makes them **unrepresentable**.

Roofline-grading kernel claims against hardware peak is established practice, not
my contribution — SOL-ExecBench, KernelBench-Hard, ROBUST-KBENCH all do it, and
KernelBench itself ships adversarial kernels. Kernel *generation* is covered by
better-resourced work: Meta's KernelAgent, ByteDance's CUDA-Agent. What is not
established is doing the checks **inside an agent harness**, with a human gate in
front of publishing, rather than scoring a benchmark offline.

## It works, and the interesting part is what it caught

28 cases in one sweep on one GPU: 17 admitted, and every one of the 8 adversarial
candidates rejected. Same operation, five toolchains, all measured by the same
gates because those gates adjudicate a *measurement schema* rather than a
toolchain:

| rmsnorm | % of the 1008.1 GB/s bus |
|---|---|
| hand-written CUDA | 90.1% |
| TileLang | 90.7% |
| Triton | 89.6% |
| Helion (autotuned) | 89.1% |
| CuTe DSL | 56.4% |
| eager PyTorch | 26.2% |

Three results are worth more than the table.

**Physics can disprove a precision claim without a reference.** One adversarial
candidate declares fp32 and quietly computes in bf16 — roughly a 2× speedup for a
2^15 loss of mantissa. It is caught three times over: 594× past the fp32
tolerance, wrong again when the output is re-read after timing, and at **100.9
TFLOP/s against an 83.1 TFLOP/s fp32 ceiling**. That last catch needs no reference
output at all. The arithmetic is impossible at the precision claimed, so the claim
is false whatever the answer looks like.

**The same kernel can be honest at one precision and dishonest at another.** A
Triton FlashAttention kernel is *rejected* when declared fp32 and *admitted* at
tf32 and bf16. Nothing about the kernel changes — `tl.dot` silently uses TF32.
Reduced precision is a perfectly good engineering choice. It widens the tolerance
*and* lowers the ceiling you are judged against, so it buys nothing you have not
earned. Taking it silently is the only thing disallowed.

**A toolchain's config is what gets measured.** A Helion config sweep on one kernel
spans **4.7×** — `block_sizes=[1]` reaches 90% of the bus, `[32]` reaches 47%.
Pinning a config measures that config, not the kernel.

## Then it caught me

This is the part I did not plan for.

**My own candidate was misreporting by 12.7×.** A CuTe DSL kernel reported 7.0% of
the memory bus, and I wrote that down as "CuTe is low-level and I used it badly."
It was not. `@cute.jit` re-traces on *every* entry, so the number was JIT dispatch,
not the kernel. Hoisting the compile out reports **88.6%** — in line with the other
four toolchains. Ten warmup calls outside the timed region could not absorb it,
because the cost was per-call rather than first-call.

Worse: a compiled CuTe function silently accepts shapes it was not compiled for.
Shapes are baked in as constants, so passing a different one does not raise — it
computes the wrong answer with stale bounds. The correctness gate caught that, not
me.

**The correctness bar was 128× too loose in one place.** The gate widened the TF32
tolerance by multiplying the harness's fp32 tolerance by 2^13. But that base grows
as `sqrt(depth)`, so the bar reached 0.5 at k=4096 — and got *looser* the deeper
the reduction. The measured deviation of a real TF32 matmul is flat at ~1.5e-3 from
k=512 to k=4096, because TF32 rounds its *operands* and accumulates in fp32; its
error is quantisation, not accumulation. My own code comment said exactly that
while the implementation did the opposite.

**And twice, the central claim turned out to be false.**

Round one: a C++ static constructor runs before `main`. A kernel with an empty body
whose constructor printed a fabricated measurement and exited was **admitted at a
plausible 91% of peak**. Six gates, all green, for code that never launched
anything. I fixed it with a nonce the harness had to echo plus wall-clock time the
caller independently observed, and later moved the measurement out of stdout into a
file.

Round two, found by review of the very PR that added all the breadth: **the fix was
not a boundary, only a speed bump.** The nonce and the output path were still in
`argv`, and the candidate was still running inside the process that wrote the
verdict. Two new candidates read them straight back out — one from `sys.argv` at
import time, one from `/proc/self/cmdline` inside a static constructor. Both wrote a
measurement claiming 92% of the bus. **Both were admitted, all nine gates green.**

Hiding the values better does not work, and I measured that rather than assuming
it. Inside the container:

| | |
|---|---|
| `/proc/<ppid>/cmdline` | **readable** |
| `/proc/<ppid>/environ` | **readable** |
| `/proc/<ppid>/mem` | refused, `EPERM` |
| `ptrace(PTRACE_ATTACH)` | refused, `EPERM` |

No flag and no environment variable can carry a secret past code running there.
But a *sibling process* can hold one safely. So the harness is now two processes: a
supervisor that takes the nonce and output path on **stdin**, never loads candidate
code, and is the only writer of a verdict; and a worker that runs the candidate and
is told neither. A worker that exits early leaves nothing behind — which the runner
already treats as a rejection.

The lesson generalises past this project. **A single process cannot both execute
untrusted code and be trusted to report on it.** No amount of care about how you
pass the secret changes that, because the secret is not the vulnerability — the
shared address space is.

## The mistake I kept making

Five times I shipped a gate that flagged **correct** work. That is worse than no
gate, because it teaches everyone to ignore the gate.

| what it flagged | why it was wrong |
|---|---|
| a baseline at 1820 GB/s on a 1008 GB/s bus | 34 MB fits in 72 MB of L2; DRAM does not bound cache-resident traffic |
| fp32 attention diverging from fp64 | my anti-narrow-range input distribution spanned 2^24 in the softmax tail |
| an honest kernel's timing variance | at `repeats=10`, `(10*9)//10` indexes the last sample, so p90 *was* max |
| torch's own matmul at 2.8e-2 | pure relative error explodes where cancellation puts the reference near zero |
| a kernel at a third of its throughput | measured after 900 s of autotuning, against a ceiling that assumes peak clock |

The sixth was nearly shipped and caught by review. NVIDIA rates the RTX 3090 at
35.6 dense TF32 TFLOPS and the A40 at 74.8 — exactly 2×, and **both are compute
capability 8.6**. Asked directly, NVIDIA's answer was that
["the TC units in each of those 2 GPUs do not necessarily act in precisely the same way"](https://forums.developer.nvidia.com/t/tf32-tflops-of-geforce-rtx-3090-vs-a40/265828)
and that the difference is *unpublished*. A ceiling keyed on compute capability
alone would have called a real A40 measurement physically impossible.

So the gate now reports utilisation against the rate verified on the hardware in
front of it, but only *refuses* a claim past the widest rate any part of that
capability is rated for. That costs sensitivity. It is still right: the failure this
exists to catch was 30× over the maximum, so a 2× margin does not hide it, and a
false accusation is the more expensive mistake.

## What it still does not claim

Hopper ceilings are **deliberately absent** — the published figures do not divide
cleanly into a per-SM-per-clock constant I can defend, so the gate returns
`UNVERIFIABLE_COMPUTE` rather than guessing. An unverifiable ceiling is
non-blocking and says so. fp16 is implemented and untested; only sm_89 hardware was
available.

And the boundary is not complete. The worker still reports its own numbers on a
descriptor. They are bounded — by a duration the supervisor observed from outside,
and by the roofline, variance and timed-work gates — but **bounded is not proven.**
Proving them means moving the timing authority outside the worker entirely,
measuring a known quantity of work from the other side of the process boundary.
That is a bigger change than the one I made, and it is not done.

One more honest negative: I A/B tested TrueForge's context management on this
workload and it cost **49% more tokens** (87,311 against 58,504) with both runs
completing. Kernel iteration keeps a small working set, so compaction paid for
itself in neither direction. Wrong regime, not a broken feature.

## Why a harness, and not a script

Three things had to be true at once, and TrueForge made them cheap.

The **sandbox has to hold the GPU**, because a performance claim measured somewhere
other than where the kernel runs is not a measurement. TrueForge's bubblewrap-based
local provider cannot do that — `bwrap` is not setuid, so it always unshares into a
user namespace, and the NVIDIA driver refuses to initialise inside one. So I wrote a
Docker sandbox provider with GPU passthrough against TrueForge's existing
`SandboxProvider` contract and its shared contract test suite: 4/4 contract,
16/16 provider, and the default 273-test unit suite untouched.

The **adjudicator has to sit outside the agent's reach.** As an MCP server it is not
a library the agent can monkeypatch or a file it can edit — it is a tool boundary,
and `publish_kernel` carries `destructiveHint` so the harness gates it behind human
approval rather than trusting the agent to ask nicely.

And the **knowledge should not be mine to invent.** The agent reads Hugging Face's
own `cuda-kernels` and `triton-kernels` skills, pinned to a git SHA, so what it
knows about sm_89 optimisation is versioned and auditable rather than baked into a
prompt.

## The shape of the thing

The design decision that paid for itself repeatedly: **the gates adjudicate a
measurement schema, not a toolchain.** Adding a backend means writing a harness that
emits that schema. Five toolchains — CUDA, Triton, Helion, CuTe DSL, TileLang —
went in with **zero changes to any gate.** The same nine checks apply whether the
kernel was written at tile level or at thread level, and the comparison table above
exists because of that and not in spite of it.

Twelve findings came out of the last review round alone. Every one was verified
against the code or the hardware before being acted on, because a reviewer can be
wrong in both directions — and one of them was wrong in the more dangerous
direction, describing a missed detection when the real risk was a false accusation.

The project's own premise was falsified twice by review. That is the strongest thing
I can say for it. A verification harness whose claims have never been tested against
someone actively trying to break them is not a verification harness. It is a hope.

---

**Code:** [github.com/rycerzes/kernel-preflight](https://github.com/rycerzes/kernel-preflight)
— MIT, including both forgery attacks as regression tests.
