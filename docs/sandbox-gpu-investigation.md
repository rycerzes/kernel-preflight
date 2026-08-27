# Getting a GPU inside a TrueForge sandbox

Kernel Preflight needs the agent to compile and benchmark CUDA *inside the harness's
sandbox primitive*, not beside it. This note records what was tried, measured on an
RTX 4090 (sm_89, driver 580.159.04, CUDA 13.2), and why the design landed where it did.

## What TrueForge ships today

Two sandbox providers exist in the tree:

| Provider | Location | Isolation |
| --- | --- | --- |
| `DaytonaProvider` | `trueforge-core/src/core/sandbox/provider/` | Daytona cloud |
| `TFYSandboxProvider` | `trueforge-core/src/core/sandbox/provider/` | TrueFoundry hosted |
| `LocalSandboxProvider` | `trueforge/src/sandbox/local/provider/` | bubblewrap (Linux) / seatbelt (macOS) |

Only `daytona` is exposed. `catalog/sandbox-catalog.yaml` lists it alone, and
`schemas/sandboxProvider.ts` carries a prose TODO:

```ts
/** Single variant today -- Widen to z.discriminatedUnion('type', [...]) when a second provider ships. */
export const SandboxProviderManifestSchema = DaytonaSandboxProviderSchema.openapi('SandboxProviderManifest');
```

So `LocalSandboxProvider` is implemented and tested (`test:local-sandbox:contract`,
`smoke:local-sandbox`) but unreachable from configuration.

Daytona is not an option for self-hosting: it went closed source in June 2026, the
public repo is frozen at v0.190.0 and unmaintained, and there is no way to deploy a
current instance on your own infrastructure.

## Attempt 1: teach the existing bubblewrap sandbox about the GPU

> **This section originally concluded that bubblewrap cannot reach a GPU. That
> conclusion was wrong, it was published upstream before it was retested, and the
> retest is below.** The wrong reasoning is kept rather than deleted, because the
> way it was wrong is the useful part.

`hostRun.ts` declares the Linux dependency set as `['bwrap', 'socat', 'rg']`, and the
Linux read policy already allows `/dev`, `/usr/lib`, `/usr/local`, `/proc` and `/sys`.
On paper the device nodes and driver libraries are already reachable.

What I measured first:

| bwrap configuration | `cuInit` result |
| --- | --- |
| `--dev /dev` (fresh devtmpfs, no nvidia nodes) | `CUDA_ERROR_NO_DEVICE` |
| `--dev-bind /dev /dev` | `CUDA_ERROR_OPERATING_SYSTEM` |
| `--dev /dev` + explicit `--dev-bind` per `/dev/nvidia*` node | `CUDA_ERROR_OPERATING_SYSTEM` |
| ...plus `/proc/driver/nvidia` rebound read-only | `CUDA_ERROR_OPERATING_SYSTEM` |

From which I concluded: *"The cause is the user namespace. `bwrap` is not setuid, so
it always unshares into a new user namespace, and the NVIDIA kernel driver rejects
initialisation from one. This is not a policy that can be relaxed with more bind
mounts."*

**Every row of that table is missing `/sys`.** The four configurations vary only how
`/dev` is handled, so they were four samples of one experiment. Four failures that
agree tell you nothing if they share a cause you never varied — and a table that
*looks* like a controlled sweep is more persuasive than a single failure, which is
what made it convincing rather than suspicious.

## The retest

Probing `cuInit` through `dlopen("libcuda.so.1")` so nothing depends on the CUDA
toolkit. Every row runs under the same unprivileged user namespace:

| bwrap configuration | `cuInit` |
| --- | --- |
| nvidia nodes `--dev-bind`, **no `/sys`** — the original test | `CUDA_ERROR_OPERATING_SYSTEM` (304) |
| `--ro-bind /dev` + `/sys` | `CUDA_ERROR_NO_DEVICE` (100) |
| `--dev-bind /dev` + `/sys` | **OK, devices=1** |
| `--ro-bind /dev`, nvidia nodes `--dev-bind`, + `/sys` | **OK, devices=1** |

And the same 256 MiB float4 copy kernel that the harness uses elsewhere:

```
host:          KERNEL OK  920.7 GB/s
inside bwrap:  KERNEL OK  920.1 GB/s
```

The namespace is genuinely new and genuinely unprivileged — `/proc/self/ns/user`
differs from the host's (`4026536407` against `4026531837`) and `/proc/self/uid_map`
is `1000 1000 1`. `nvidia-smi` works inside it.

So there are two requirements, and neither is about namespaces:

1. **`/sys` must be mounted.** Without it, `CUDA_ERROR_OPERATING_SYSTEM`.
2. **The nvidia device nodes must be writable.** A read-only `/dev` gives
   `CUDA_ERROR_NO_DEVICE`; binding `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`
   and `/dev/nvidia-uvm-tools` read-write is enough, so `/dev` itself can stay
   read-only.

This is exactly what `nvidia-container-toolkit` automates. It is not a capability the
toolkit uniquely confers.

### What it means for the local provider

`ALLOW_READ_BY_PLATFORM.linux` already lists `/dev`, `/proc` and `/sys`, and
`denySharedDefaultWritePaths()` deliberately keeps `/dev/*` out of the deny list. So
the shipped local provider may already be close to GPU support, plausibly needing only
that the nvidia nodes be writable.

That is **not confirmed end to end.** Creating a local sandbox on this host pip-installs
pydantic, and the corporate proxy blocks it, so the shipped provider could not be
driven far enough to test. The mechanism result above is solid; the claim about the
provider specifically is not, and is written here as a hypothesis rather than a finding.

### Why this is in the repository

The whole project argues that a confident claim from the party who benefits from it is
not evidence. I made one, published it in an upstream issue and pull request, and was
caught by a reader asking whether it was true. Both are corrected in public rather
than edited quietly:
[the correction](https://github.com/truefoundry/trueforge/issues/466#issuecomment-5434642663).

The gates in this repository are built to catch a kernel that cannot be trusted about
its own speed. Nothing in it was watching the author.

## Attempt 2: containers

Verified working on the first try, twice:

```
$ docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 \
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
NVIDIA GeForce RTX 4090, 24564 MiB, 580.159.04
```

Docker 29.7.2 with the nvidia runtime and CDI (`nvidia.com/gpu=all`) resolves the
namespace problem properly, because that is precisely what the container toolkit is
for.

## Decision

Implement a **container-backed sandbox provider**, and keep it — but for a different
reason than the one it was chosen for.

The original reasoning was that a container was the *only* way to get a GPU into a
sandbox. That was wrong, per the retest above, so "more code than a bind-mount patch,
but no unknown-unknowns" no longer justifies it: the bind-mount patch has no
unknown-unknowns either, and is about two extra paths.

What still justifies it has nothing to do with GPUs. Requests for non-Daytona
sandbox backends exist upstream -- E2B ([#387](https://github.com/truefoundry/trueforge/issues/387),
`help wanted`, open) and ComputeSDK
([#414](https://github.com/truefoundry/trueforge/issues/414), since closed) -- and
now that Daytona cannot be self-hosted, a container provider is the only route to a
fully self-hosted TrueForge.

This went upstream as
[#466](https://github.com/truefoundry/trueforge/issues/466) (the analysis) and
[#467](https://github.com/truefoundry/trueforge/pull/467) (the implementation), both
now carrying the correction, and the PR retitled around self-hosting with GPU
passthrough demoted to a bonus. The maintainers have been told that closing it is a
perfectly good outcome.

The better follow-up, if anyone wants it, is the two-line policy change to the local
provider rather than this.

`LocalSandboxProvider` is the reference to port from: same `SandboxProvider`
interface, same path-shaped `sandboxId`, same file-transfer and traversal-validation
concerns, with `docker exec` replacing the bwrap supervisor. The shared contract
suite at `tests/core/sandbox/provider/sandboxProviderContractSuite.ts` already
contains a branch for path-id backends, so correctness is measured against tests
the maintainers wrote rather than tests we invented.

### One bug not to reproduce

Upstream issue #416: the TFY provider's `uploadFile` base64-encodes the whole
payload into a single `argv`, so it fails past roughly 96 KiB on `MAX_ARG_STRLEN`
/ `E2BIG`. Kernel projects move real files. The container provider streams through
`docker cp` / stdin instead.
