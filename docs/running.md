# Running kernel-preflight

Two long-lived processes on the GPU host, plus a container image. Order matters
in one place, noted below.

## 1. Sandbox image

```bash
docker build -t kernel-preflight-sandbox:cuda13 -f docker/Dockerfile.sandbox docker/
```

The stock `nvidia/cuda:*-devel` image is not sufficient. TrueForge's sandbox
bootstrap runs a Python script to materialise git-backed skills, so the image
needs `python3`, `git` and the `pydantic` pin the bootstrap declares. Without them
sandbox initialisation fails with `python3: not found` and no skill ever mounts —
the agent still runs, which makes the failure easy to miss.

The image also clears `ENTRYPOINT`. The NVIDIA base prints a licence banner on
stdout, and stdout is a shared channel — TileLang logs to it too. The measurement
no longer travels that way (see below), but a clean stdout is still what makes a
harness failure readable, so the noise is removed at the source.

## 2. Preflight MCP server

```bash
PREFLIGHT_MCP_PORT=8791 PYTHONPATH=src .venv/bin/python -m preflight.mcp_server
```

Localhost only. It compiles and runs model-authored CUDA and must never be
exposed off the host, though the execution itself is containerised (see
`runner.py`).

Inside that container there are two processes, and the split is the guarantee
rather than a detail of it:

```
supervisor.py ──── given the nonce and output path on stdin, never in argv
  │                the only writer of a verdict; never loads candidate code
  └── worker ───── driver.cu or driver.py, linked to or importing the candidate
                   reports on a descriptor; told neither secret
```

A single process cannot both execute a candidate and be trusted to report on it.
While this was one process, two candidates forged a 92%-of-peak verdict without
launching a kernel, by reading the nonce and output path out of `sys.argv` and
`/proc/self/cmdline`. Both are kept in `examples/` as regression tests.

> **Restart it after any change under `src/preflight/`.**
>
> Python imports at startup, so the running server keeps serving the code it was
> launched with. This cost an hour: the variance gate had already been changed
> from max/min to p90/median, but the server was still enforcing the old one, and
> three subagents spent that hour writing kernels to satisfy a gate that no longer
> existed. One of them diagnosed it correctly — *"this is clearly a systematic
> issue with the test"* — which was right, and the test had already been fixed on
> disk.

## 3. TrueForge

```bash
TRUEFORGE_DIR=/path/to/trueforge ./run-trueforge.sh   # localhost:8790, SQLite
```

Two environment settings in that script are not optional on this host:

- `XDG_DATA_HOME` — `env-paths` otherwise puts the database in `~/.local/share`.
- `NODE_EXTRA_CA_CERTS` — the host sits behind a TLS-intercepting proxy whose
  root CA is in the system store but not in Node's bundled one, so every model
  call fails with `UNABLE_TO_GET_ISSUER_CERT_LOCALLY`. `--use-system-ca` would
  also work, but the package script hardcodes `NODE_OPTIONS`, so it would be
  discarded.

## 4. Configuration

```bash
# sandbox provider
curl -X PUT localhost:8790/api/v1/settings/sandbox-providers -H 'Content-Type: application/json' \
  -d '{"manifest":{"type":"docker","image":"kernel-preflight-sandbox:cuda13","exec_timeout_ms":600000,"gpus":"all"}}'

# the preflight tools
curl -X POST localhost:8790/api/v1/settings/mcp-servers -H 'Content-Type: application/json' \
  -d '{"manifest":{"type":"remote","name":"kernel-preflight","url":"http://127.0.0.1:8791/mcp","description":"Preflight gates for GPU kernel performance claims."}}'

# the skills: Hugging Face's own, from one repo at one pinned SHA
for SKILL in cuda-kernels triton-kernels; do
  curl -X POST localhost:8790/api/v1/settings/skills -H 'Content-Type: application/json' \
    -d "{\"manifest\":{\"type\":\"git\",\"name\":\"$SKILL\",
         \"url\":\"https://github.com/huggingface/kernels\",
         \"path\":\"kernel-builder/skills/$SKILL\",
         \"ref\":\"3b21db391253ac5a75203482a2031811115494a0\"}}"
done

# the agent
curl -X POST localhost:8790/api/v1/agents -H 'Content-Type: application/json' \
  --data @agent/kernel-preflight.agent.json
```

Both skills are pinned to a commit SHA rather than a branch, which is what the
upstream skills documentation recommends and what makes a run reproducible. They
are not mine: `cuda-kernels` and `triton-kernels` are Hugging Face's, so what the
agent knows about sm_89 optimisation is versioned and auditable rather than baked
into a prompt.

## Known operational gaps

- **Nothing calls `reapStale` on a schedule.** The container sandbox provider can
  reap stale sandboxes by label, but TrueForge has no periodic hook to invoke it,
  so containers accumulate across runs until something clears them. Reaping is
  scoped, so it is safe to run by hand:
  `docker rm -f $(docker ps -aq --filter label=com.truefoundry.trueforge.sandbox)`
- **Code Mode is unavailable** on the container provider. It needs a bidirectional
  transport and the local provider's unix socket has no container equivalent. The
  session degrades to ordinary tool calls rather than failing.
- **`max_tokens` is provider-capped.** GLM-5.2 via the HF router rejects anything
  above 16384 with a 400. A model that writes long responses will truncate and
  fail the turn; the agent instructions carry explicit brevity rules for that
  reason.

## A limitation the fan-out exposed

Three subagents explored three block sizes in parallel and returned 90.3%, 90.7%
and 91.0% of DRAM peak. The agent declared the third the winner "by 0.7
percentage points".

That conclusion is not supported. Run-to-run variance on *identical* code was
measured at 84.9%–91.4% on this host, which is wider than the entire spread
between the three variants. Every measurement passed its own variance gate,
because that gate asks whether a single kernel's timing is stable enough for its
own median to mean something — not whether two medians differ.

Ranking candidates is a different statistical question from admitting one, and
the gates currently answer only the second. Comparing variants honestly needs
repeated independent runs per variant and a confidence interval on the
difference. Until that exists, treat the fan-out as a way to explore the design
space in parallel, not as a way to pick a winner by a fraction of a percent.

## Context-management A/B: a negative result

`benchmark/context_ab.py`, four trials per arm, arm order alternating:

| arm | completed | admitted | median tokens | median input | median wall |
| --- | --- | --- | --- | --- | --- |
| managed | 100% | 100% | 87,311 | 84,198 | 65s |
| raw | 100% | 100% | 58,504 | 56,219 | 55s |

Compaction and large-response offload cost **~49% more tokens** on this task and
finished slower. Every trial in both arms completed and was admitted, so this is
not a reliability trade being paid for in tokens — it is pure overhead here.

This is the wrong regime for the feature, not evidence the feature is broken. The
task's input sits around 56k tokens without management; the compaction threshold
is 50k, so compaction fires and pays for an extra summarising model call, while
offload converts one large tool result into a sandbox write plus follow-up reads.
Both costs land, and the run ends before either can amortise.

It does not contradict TrueForge's published figures, which came from
Enterprise-Bench across three MCP servers — a far longer, more tool-heavy
workload where a context window would actually be exhausted. It qualifies *when*
the machinery pays for itself, which is a more useful claim than a percentage.

Demonstrating the benefit needs a task that would otherwise overflow: many more
iterations, several skills, or a much larger operator set. That measurement is
not in this repository yet, and the honest summary is that on a task this size
you should leave context management off.

## Two traps that cost real time

**A shadow package silently shadowed the real one.** An `rsync -az src/ tests/ host:repo/`
with two sources copies the *contents* of each into the destination, so `src/preflight/`
landed as `repo/preflight/`. Under `python3 -c` the working directory precedes
`PYTHONPATH` on `sys.path`, so that stale copy won every import. The symptom was a
backend reported as unknown while being present in both copies of the file.

Fixed twice over: `sync.sh` purges `__pycache__` and syncs each tree to its own
destination, and the matrix script asserts the module it imported is the one under
`src/` before measuring anything. Any harness that can silently measure the wrong
code is worse than no harness.

**`rsync -a` preserves mtimes, so bytecode can look newer than source.** A synced
`.py` older than the `.pyc` beside it leaves Python running the cache. Same family
as forgetting to restart the MCP server, and the same fix: purge on sync.

## Known flakiness

The variance gate asks whether a kernel's own median means anything, and it has
been retuned twice for exactly this. It compared max/min, then p90/median — which
at `repeats=10` indexes the last sample, so p90 *was* max and honest kernels were
rejected — and now compares **p75/p25 against 1.5**, which is unmoved by a handful
of spikes by construction. Combined with batched timing, an eighteen-case sweep no
longer trips it.

What is still not implemented is the harder statistical question. The gate decides
whether to *admit* one kernel; it does not decide whether one kernel is faster than
another. Two independent sweeps of the full matrix agree to within 0.3 points on
every memory-bound case but disagree by up to 6.8 points on the compute-bound
attention ones. Ranking variants honestly needs repeated independent runs per
variant and a confidence interval on the difference, which is why the fan-out below
should be read as a way to explore the design space, not to pick a winner by a
fraction of a percent.
