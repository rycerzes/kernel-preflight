# kernel-preflight

**A GPU kernel agent that cannot lie about the speedup.** It submits kernel source and
never submits a number.

Built on [TrueForge](https://github.com/truefoundry/trueforge). An agent that writes a
kernel *and* reports its own speedup has both the incentive and the opportunity to be
wrong in its own favour, and correctness tests do not catch it — a kernel can be
numerically perfect and still be timed dishonestly. So the agent here never touches the
measurement: a fixed harness it cannot see owns the allocation, the input distribution,
the timing loop, the reference and the tolerances, and nine gates adjudicate the result
against the hardware's physical limits.

In February 2025 Sakana AI's "AI CUDA Engineer" reported 10–100× speedups. The kernels
were not faster; it had exploited the benchmark harness, and several figures implied
throughput ~30× above what the hardware can physically deliver. Every documented way of
faking a kernel speedup is a property of measurement code. Making that code ours does not
detect them — it makes them unrepresentable.

## Demo

**[▶ 2m45s demo video](video/out/kernel-preflight-demo.mp4)** — four of its beats are a
recorded TrueForge session replayed verbatim: the agent finding a TF32 trap unprompted,
failing two submissions, reading the traceback, and earning nine green gates. A fifth is a
live run against a kernel that serves a cached answer while it is being timed.

## How it works

```
TrueForge  (on the GPU host)
├── DockerSandboxProvider ──── the agent's workspace: GPU, nvcc, skills
│                              (new; contributed upstream)
└── MCP: kernel-preflight ──── the adjudicator, outside the agent's control
      ├── device_spec          hardware ceilings from the CUDA driver
      ├── preflight_kernel     compile → measure → gate, in isolation
      └── publish_kernel       approval-gated; re-verifies under the same contract
```

Inside the container that is **two processes, and the split is the whole guarantee**:

```
supervisor.py ──── holds the nonce and output path (stdin, never argv)
  │                the only writer of a verdict; never loads candidate code
  └── worker ───── driver.cu / driver.py, linked to the candidate
                   reports on a descriptor; told neither secret
```

A single process cannot both execute a candidate and be trusted to report on it. While
this was one process, two candidates forged a 92%-of-peak verdict without launching a
kernel. Both are regression tests now.

## The gates

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

The gates adjudicate a **measurement schema, not a toolchain** — which is why five
toolchains were added after they were written, with zero gate changes.

## Quickstart

Needs an NVIDIA GPU, Docker with the container toolkit, and Node 22+.

```bash
docker build -t kernel-preflight-sandbox:cuda13 -f docker/Dockerfile.sandbox docker/
PREFLIGHT_MCP_PORT=8791 PYTHONPATH=src .venv/bin/python -m preflight.mcp_server
TRUEFORGE_DIR=/path/to/trueforge ./run-trueforge.sh
```

Then register the sandbox provider, the MCP server, the skills and the agent — full
commands and the operational gotchas are in [`docs/running.md`](docs/running.md).

Ask the agent for a kernel and it returns a verdict it did not author:

```
> Write the fastest fp32 matmul you can in Triton and get it admitted.

ADMITTED   NVIDIA GeForce RTX 4090
  [pass] provenance         nonce echoed; 25470 ms of work inside a 25761 ms process
  [pass] correctness        worst deviation 0.10x of the fp32 tolerance at 4096x4096
  [pass] timed_work         the measured calls produced correct output
  [pass] roofline           peak fp32 pipelines utilisation 60.9% at 4096x4096
  ...
```

## Results

**67 cases, 51 admitted, and every one of 12 adversarial candidates rejected.** 18
operations across 6 toolchains, one sweep on one RTX 4090 with a thermal cooldown between
cases.

One op, five toolchains — the comparison the gates are indifferent to:

| rmsnorm | fp32, % of the 1008.1 GB/s bus |
| --- | --- |
| TileLang | 90.5% |
| hand-written CUDA | 89.7% |
| Triton | 89.0% |
| Helion (autotuned) | 88.7% |
| CuTe DSL | 84.9% |
| eager PyTorch | 26.1% |

Not everything reaches 90%, and that is deliberate — `gather` (78.0%) is irregular by
construction, `moe_gemm` (32.6%) reads every expert's weights whichever tokens arrive. A
harness whose every op streams teaches the wrong lesson.

Full tables, per-op spreads, precision contracts and how the operations were chosen:
**[`docs/results.md`](docs/results.md)**.

## What it caught

- **A cached-answer kernel admitted at 89.7% of the bus.** Correctness held, the timing
  was real, and the roofline could not help. Fixed by rotating one input before every
  timed sample, so nothing computed earlier stays valid.
- **A kernel declaring fp32 while computing bf16**, caught three ways — the sharpest being
  100.9 TFLOP/s against an 83.1 TFLOP/s ceiling, which needs no reference output at all.
- **A measurement bug in my own candidate**: `cute_silu.py` reported 7.0% of the bus, which
  was `@cute.jit` re-tracing per launch. Hoisting the compile reports 88.6% — a 12.7×
  correction.
- **Six times a gate rejected correct work**, which is worse than shipping no gate. All six
  are written up rather than quietly fixed.

The narrative version, including the three attacks that got through first and the two
claims I withdrew publicly: **[`BLOG.md`](BLOG.md)**.

## Layout

| Path | Contents |
| --- | --- |
| `src/preflight/harness/supervisor.py` | the only process that never runs candidate code, and the only writer of a verdict |
| `src/preflight/harness/driver.cu`, `driver.py` | the measurement workers, one per backend, emitting the same schema |
| `src/preflight/gates.py` | adjudicates that schema — nine gates, indifferent to the toolchain |
| `src/preflight/runner.py` | isolation: container, no network, no inherited environment |
| `src/preflight/mcp_server.py` | the three tools TrueForge sees |
| `src/preflight/harness/examples/` | 42 honest kernels, 12 adversarial, 1 merely wrong — all regression tests |
| `benchmark/full_matrix.py` | the 67-case sweep |
| `tests/` | 39 gate tests, including ones that fail against the pre-fix code |
| `docker/`, `agent/` | sandbox images and the saved TrueForge agent manifest |
| `upstream/trueforge/` | the container sandbox provider, submitted as [truefoundry/trueforge#467](https://github.com/truefoundry/trueforge/pull/467) |
| `video/` | the demo: an HTML composition, narration script and build |

## Docs

| | |
| --- | --- |
| [`BLOG.md`](BLOG.md) | the write-up: what it caught, and what caught me |
| [`docs/results.md`](docs/results.md) | the full 67-case matrix and how the ops were chosen |
| [`docs/running.md`](docs/running.md) | running it, and the operational gaps |
| [`docs/review.md`](docs/review.md) | Qodo review rounds, and the finding that invalidated the premise twice |
| [`docs/sandbox-gpu-investigation.md`](docs/sandbox-gpu-investigation.md) | whether bubblewrap can reach a GPU (it can) |

## Code review

Every substantive change went through a pull request reviewed by Qodo before merge —
**5 PRs, 33 findings, all fixed**, each verified against the code or the hardware before
being acted on rather than accepted on the reviewer's word.

Two rounds mattered more than the rest. Review of the agent PR found a candidate could
**forge its own verdict** — a kernel with an empty body whose C++ static constructor
printed a fabricated measurement was admitted at 91% of peak. Review of PR #4 then found
the fix was insufficient: the nonce still travelled in `argv`, and two new candidates read
it back out of `sys.argv` and `/proc/self/cmdline`. That is what forced the
supervisor/worker split. Details: [`docs/review.md`](docs/review.md).

## Notes

- Built for the WeMakeDevs / TrueFoundry Agent Harness Hackathon, on a single RTX 4090.
- Built with AI assistance (Claude Code), as permitted by the hackathon rules.
- `upstream/` is mirrored here so this repository is self-contained. It is submitted as
  [truefoundry/trueforge#467](https://github.com/truefoundry/trueforge/pull/467), behind
  [#466](https://github.com/truefoundry/trueforge/issues/466), because TrueForge asks for
  maintainer approval before code contributions.
