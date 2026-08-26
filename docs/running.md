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
stdout, and the preflight harness communicates over stdout in JSON. Strict parsing
is deliberate on the reading side, so the noise is removed at the source.

## 2. Preflight MCP server

```bash
PREFLIGHT_MCP_PORT=8791 PYTHONPATH=src .venv/bin/python -m preflight.mcp_server
```

Localhost only. It compiles and runs model-authored CUDA and must never be
exposed off the host, though the execution itself is containerised (see
`runner.py`).

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
./run-trueforge.sh          # standalone mode, SQLite, localhost:8790
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

# the agent
curl -X POST localhost:8790/api/v1/agents -H 'Content-Type: application/json' \
  --data @agent/kernel-preflight.agent.json
```

The skill is pinned to a commit SHA rather than a branch, which is what the
upstream skills documentation recommends and what makes a run reproducible.

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
