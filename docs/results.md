# Results

Every number here is the harness's. A candidate submits source and never a number: the
harness owns allocation, the input distribution, timing, the reference and the tolerances,
and the candidate cannot reach any of them.

Measured on an RTX 4090 (sm_89, driver 580.159.04, CUDA 13.2). Reproduce with
`benchmark/full_matrix.py`.

## Infrastructure

Measured on an RTX 4090 (sm_89, driver 580.159.04, CUDA 13.2):

| Check | Result |
| --- | --- |
| TrueForge's own `sandboxProviderContractSuite` | 4/4, incl. sibling-escape isolation |
| Docker provider suites (contract + GPU + hardening) | 16/16 |
| `packages/trueforge` unit suite | 274/274, 36 suites (273 upstream plus one this adds) |
| `trueforge-core` unit suite, unaffected | 388/389, 39 suites (1 skipped upstream) |
| `nvidia-smi` inside the sandbox | RTX 4090 visible |
| `nvcc -arch=sm_89` compile + run inside the sandbox | 921.2 / 1008.1 GB/s = 91.4% of peak |
| 5 MiB upload/download round-trip | byte-exact |
| Agent end to end: skill → sandbox → gates | rmsnorm admitted at 91.1% of the bus, first submission |
| Agent on a compute-bound op it was not tuned for | Triton matmul admitted at 60.9% of the fp32 ceiling, beating the hand-written `triton_matmul.py` at 58.2% |
| Gate unit suite | 39/39 |

The device ceiling is corroborated two ways: `preflight.device` derives
**1008.1 GB/s** from the CUDA driver attribute API, and an independent CUDA C
bandwidth kernel measures against the same figure.

## The kernel matrix

67 cases in one session on one GPU, with a thermal cooldown between each so the
figures are comparable — `benchmark/full_matrix.py`. A sweep run straight after
Helion's autotuner once read a third of the throughput of the same kernel idle,
which is why Helion is scheduled last and why the harness samples the SM clock and
says so when it ran below peak.

**51 admitted, 16 not, and every one of the 12 adversarial candidates rejected.**

One op, five toolchains — the comparison the gates are indifferent to, because they
adjudicate a measurement schema rather than a toolchain:

| rmsnorm | fp32, % of the 1008.1 GB/s bus |
| --- | --- |
| TileLang | 90.5% |
| hand-written CUDA | 89.7% |
| Triton | 89.0% |
| Helion (autotuned) | 88.7% |
| CuTe DSL, block per row via shared memory | 84.9% |
| eager PyTorch | 26.1% |

Memory-bound, other ops: CUDA silu 91.2%, CUDA softmax 88.9%, TileLang silu 90.5%,
CuTe silu 88.1%, CUDA transpose 30.9%.

Reduced storage precision, on the backends that can honour it: Triton rmsnorm fp16
89.6%, TileLang silu fp16 90.6%, CuTe rmsnorm bf16 82.5% and fp16 82.4%, Triton
FlashAttention fp16 88.3% of the fp16 ceiling.

## Eighteen operations

Best and worst admitted result per op, from one sweep. The spread within a row is the
useful part: it is the same operation, the same reference and the same gates, so the gap
is the kernel.

| op | best | worst | what it exercises |
| --- | --- | --- | --- |
| `silu` | 91.2% cuda | 88.1% cute | elementwise, the control |
| `matmul` | 91.5% triton/tf32 | 57.8% triton/fp32 | compute-bound; tf32 vs pinned ieee |
| `swiglu` | 91.2% tilelang | 39.0% torch | Fusion — 72% of KernelBenchX's hardest category fails |
| `rmsnorm` | 90.4% tilelang | 26.1% torch | one reduction, five toolchains |
| `rope` | 89.7% triton | 28.9% torch | element `i` needs element `i + d/2` |
| `attention_decode` | 92.9% triton | 19.6% torch | the only attention shape on the *memory* side |
| `quantize` | 88.7% triton | 14.0% torch | Quantization — 0 of 30 solved in KernelBenchX |
| `layernorm` | 88.5% triton | 42.9% torch | mean before variance |
| `attention` | 88.5% triton/fp16 | 29.3% torch | FlashAttention forward, online softmax |
| `softmax` | 88.8% cuda | — | numerically delicate reduction |
| `cross_entropy` | 88.0% triton | 44.8% torch | Loss; output smaller than input |
| `attention_gqa` | 82.7% triton/bf16 | 28.6% torch | index the shared KV head, don't expand it |
| `gather` | 78.0% triton | 38.2% torch | Index; irregular, cannot reach peak |
| `attention_backward` | 70.3% triton/bf16 | 11.1% triton/fp32 | dQ, dK, dV; three kernels |
| `attention_causal` | 67.3% triton/bf16 | 21.1% torch | skip tiles, mask only the diagonal |
| `attention_paged` | 44.8% triton | 13.4% torch | block table indirection |
| `moe_gemm` | 32.6% triton | 21.6% torch | grouping, not arithmetic |
| `transpose` | 31.0% cuda | — | pure movement, bit-exact |

Three of those numbers are low for reasons that are the operation rather than the code,
and having them in the set is what stops 90% reading as the pass mark. `gather` is
irregular by construction. `moe_gemm` reads every expert's weights whichever tokens
arrive. `attention_paged` chases a block table through a pool. A harness whose every op
streams teaches the wrong lesson.

`attention_backward` at fp32 is the one case where a hand-written kernel *loses* to
torch — 11.1% against 22.3% — because it pins `input_precision="ieee"` and recomputes the
scores a third time to get dQ. At bf16 it wins, 70.3%. Both are reported.

## How the operations were chosen

The first six were picked for coverage. The rest were not picked by taste.

Six came from [KernelBenchX](https://arxiv.org/abs/2605.04956), which grades 176 tasks
across 15 categories and reports where LLM-written kernels actually break. Six more came
from what a serving stack runs: [FlashInfer](https://arxiv.org/pdf/2501.01005) reports
28–30% latency reduction from fusing RoPE into attention, and paged attention, GQA and
grouped-expert GEMM are the kernels a deployment is built on. The remaining ones —
`attention_backward` for training, `attention_decode` for generation — cover the halves
of attention that a single non-causal prefill kernel does not.

The first six operations here were picked for coverage. The next three were picked from
[KernelBenchX](https://arxiv.org/abs/2605.04956), which grades 176 tasks across 15
categories and reports where LLM-written kernels actually break: **Fusion is the
largest category and 72% of it fails** across every method they tested, **Quantization
is 0 of 30**, and Math and Activation are solved consistently. This harness had the
easy ones.

| op | category | eager torch | best hand-written |
| --- | --- | --- | --- |
| `swiglu` — `silu(a) * b` | Fusion (72% fail) | 39.0% | **91.4%** TileLang |
| `quantize` — per-row int8 round trip | Quantization (0/30) | 14.0% | **88.7%** Triton |
| `rope` — split-half rotary embedding | — | 28.8% | **89.7%** Triton |
| `cross_entropy` — one loss per row | Loss | 44.8% | **88.1%** Triton |
| `layernorm` — mean before variance | Normalization | 42.9% | **88.4%** Triton |
| `gather` — `table[idx]` | Index | 38.1% | **78.0%** Triton |

RoPE is not from that taxonomy. It is here because essentially every deployed
transformer runs it and because fusing it with attention is where inference engines find
their wins — [FlashInfer](https://arxiv.org/pdf/2501.01005) reports 28–30% latency
reduction from exactly that fusion. Structurally it is the odd one out: output element
`i` depends on input element `i + d/2`, so the row cannot be split into independent
tiles the way an elementwise kernel can.

SwiGLU across four toolchains, since fusion is where the failures concentrate: TileLang
91.4%, Triton 90.2%, CuTe DSL 88.4%, eager torch 39.0%. The arithmetic is five FLOP an
element and irrelevant; the entire question is whether the kernel touches each input
once instead of materialising the intermediate.

**`gather` is the one op here that should not reach 90%, and that is the point of
including it.** Its access pattern is irregular — 512-byte rows fetched from
unpredictable places — so the achievable fraction of peak is set by the pattern rather
than by the code. Triton reaches 78.0% and torch 38.1%; neither is a bad kernel. Every
other op in the set streams, and without one that cannot, 90% quietly becomes the pass
mark.

It also caught an error in its own cost model, which is the roofline gate working on
me rather than on a candidate. Indices are drawn with replacement, so about 1 − 1/e of
the rows are distinct and the rest are cache hits. Charging for all of them overstated
the traffic by a fifth, and a correct Triton kernel came out at **95.5% of the bus** —
refused as pointing at the measurement rather than the kernel. It was right.

**`quantize` could not be graded at all as first specified,** and that is the more
interesting half of the story. With a scale of `absmax / 127` the operation is
discontinuous, so a value within an ulp of a rounding boundary lands on a different
integer in fp32 than in the float64 reference — 7.7% relative error on a *correct*
kernel. No tolerance fixes it: one wide enough to admit the boundary cases also admits
truncation, which is the error the op exists to catch.

A power-of-two scale removes the ambiguity rather than tolerating it. `absmax` is exact
because a max reduction is exact, its binary exponent is exact, and dividing by 2^k is
an exponent shift — so the round trip is bit-exact and both candidates measure
**violation 0.000000** against float64. Scaling by a power of two is also what several
real quantisation schemes do, for the same reason.

Two details had to be right for that to hold. The reference takes `frexp`, so the
Triton kernel reads the IEEE-754 exponent field directly: `ceil(log2(peak))` disagrees
at every exact power of two — `log2(2^m)` is `m` where `frexp` reports `m+1` — and can
flip just below one when fp32 `log2` rounds up. And rounding needs `libdevice.rint` for
round-half-even, because `tl.math` has only `floor` and `ceil` in Triton 3.7 and
`floor(v + 0.5)` is half-up, which differs on exactly the ties a power-of-two scale
makes reachable.

Compute-bound, audited against the arithmetic ceiling rather than the bus:

| kernel | declared | verdict |
| --- | --- | --- |
| Triton matmul, `input_precision="ieee"` | fp32 | 57.9% of fp32 pipelines |
| Triton matmul, TF32 | fp32 | **rejected** — correctness, timed_work |
| the same TF32 kernel | tf32 | 90.9% of tf32 pipelines |
| torch SDPA | fp32 | 29.2% |
| torch SDPA | bf16 | 92.7% |
| Triton FlashAttention | fp32 | **rejected** — correctness, timed_work |
| the same kernel | tf32 | 82.8% |
| the same kernel | bf16 | 82.8% |

Two independent sweeps of this matrix agree to within **0.3 points** on every
memory-bound case and disagree by up to **6.8 points** on the compute-bound
attention ones, which pick a different winning shape from run to run. That gap is
the argument for `variance` being a gate rather than a note, and it is the reason a
single number from a single run is not evidence — including for the numbers above.

## Seven results worth reading

**One axis was covering for the other, and an attack got through.** A candidate that
computes once and serves the answer from cache for the rest of the timed calls is
refused on matmul at **18.4x the fp32 ceiling** — skipping a 4096-cubed GEMM for a
64 MiB copy implies 1529 TFLOP/s. The same trick on rmsnorm was **admitted at 89.7% of
the memory bus**, replacing an honest kernel that measures 26%. Nothing saw it:
correctness held because a cached answer is right for the inputs it came from, the
externally observed timing was satisfied because those calls really were that fast,
and the roofline could not help because a copy moves exactly the traffic the harness
charges while four FLOP an element sits far below the ridge point.

The fix is that the answer now moves: one input is rotated by a single element before
every timed sample, and the output is checked against the reference for the final input
state. Nothing computed earlier stays valid and no candidate can tell which sample is
last, so every sample has to do the work. It now fails `timed_work` by four orders of
magnitude on both backends. Rotation rather than an offset for a reason — see the false
accusations below.


**The agent declared the precision it actually computed in, unprompted.** Asked for
"the fastest fp32 matmul you can in Triton", it pinned `input_precision="ieee"` rather
than letting `tl.dot` quietly give it TF32 — and was admitted at 60.9% of the fp32
ceiling, ahead of the hand-written baseline in this repo at 58.2%. Taking the default
would have been faster and would have been rejected on correctness.


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

## What it does not claim

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
- **fp16 is exercised now**, which it previously was not, and the reason given for
  that was wrong: sm_89 has fp16 tensor cores, so this hardware could always test it.
  Six fp16 cases run in the matrix. Testing it also found two candidates hardcoding
  `float32` and a shared-memory allocation that could not honour anything else.
- **The worker's numbers are bounded, not proven.** See
  [the honest limits](#the-limit-this-does-not-clear).

## Six false accusations

A gate that flags correct work is worse than no gate, and this project has built that
gate six times. Each of these rejected an honest kernel before it was fixed, and the
last one shipped *inside* another fix — which is the more useful lesson: the dangerous
moment is not writing a new check, it is changing what the harness feeds the kernel.

| what it flagged | why it was wrong |
| --- | --- |
| baseline at 1820 GB/s on a 1008 GB/s bus | 34 MB fits in 72 MB of L2; DRAM does not bound cache-resident traffic |
| fp32 attention diverging from fp64 | the anti-narrow-range input distribution spanned 2^24 in the softmax tail |
| an honest kernel's timing variance | at `repeats=10`, `(10*9)//10` indexes the last sample, so p90 *was* max |
| torch's own matmul at 2.8e-2 | pure relative error explodes where cancellation puts the reference near zero |
| a kernel at a third of its throughput | measured after 900 s of autotuning, against a ceiling that assumes peak clock |
| a correct TF32 FlashAttention kernel | the fix above first offset the input by a constant, which shifted the softmax into a more saturated regime and raised the error 1.7x |

And one claim withdrawn rather than a gate fixed. I wrote that `cross_entropy` without
the row-max subtraction overflows, added an adversarial candidate for it, and the
harness admitted that candidate — correctly. Measured: logits span ±80, the largest row
sum of exponentials is 4.3e35, fp32 holds 3.4e38. Nothing overflows, so the unstable
kernel's answer is right and the docstrings claiming otherwise were wrong. What does
break is the *scope* of the reduction: `cheat_blockwise_cross_entropy.py` shifts each
tile by that tile's own maximum and adds the partials without rescaling, which gives no
NaN, the right sign, a plausible magnitude, and an answer wrong by 227× tolerance.

## The limit this does not clear

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

What remains: the worker still reports its own numbers on a descriptor, so they are
bounded rather than proven — but the bound is now tight. Both harnesses write a byte
either side of every timing loop on a descriptor the supervisor owns, so those loops
are timed on a clock the measured code does not control. `harness_wall_ms` used to be
the worker's whole lifetime, several seconds of which is torch import and float64
reference computation, so a forgery claiming 30 ms of work sat comfortably inside a
5-second process. Counting `inner_iters` on the claim side closed the rest: the old
formula understated the claim by up to 40x on the small shapes.

Honest kernels now sit at **1.05x–1.27x** of the interval the supervisor observed,
against roughly 43x of slack before. To claim near-peak throughput, a worker has to
actually spend that wall time inside the loops.

Spending it without doing the work is the residual, and `correctness`,
`input_sensitivity` and `timed_work` are what stand against that. Removing it entirely
means the supervisor owning allocation and verification too — plausibly over CUDA IPC
— which is a redesign rather than a fix, and is not done.

## Prior art

Roofline-grading kernel claims against hardware peak is **established practice, not a
contribution of this project**:

- [`SOL-ExecBench`](https://arxiv.org/html/2603.19173v1) — speed-of-light benchmarking against hardware limits
- [`KernelBench-Hard`](https://github.com/Infatoshi/KernelBench-Hard) — roofline-graded against hardware peak
- `ROBUST-KBENCH` — addresses KernelBench's reward-hacking vulnerabilities
- [`KernelBench`](https://github.com/ScalingIntelligence/KernelBench) itself ships adversarial kernels and flags excessive speedups

Kernel *generation* is also well covered by better-resourced work —
[`KernelAgent`](https://github.com/meta-pytorch/KernelAgent) (Meta/PyTorch) and
[`CUDA-Agent`](https://github.com/BytedTsinghua-SIA/CUDA-Agent) (RL-trained, SOTA on
KernelBench). This project does not compete on kernel quality. What is not established
practice is doing the checks **inside an agent harness**, with a human approval gate in
front of publishing to the Hub, rather than scoring a benchmark offline.
