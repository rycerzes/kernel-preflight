# Getting a GPU inside a TrueForge sandbox

Kernel Preflight needs the agent to compile and benchmark CUDA *inside the harness's
sandbox primitive*, not beside it. This note records what was measured on an RTX 4090
(sm_89, driver 580.159.04, CUDA 13.2) and why the design landed where it did.

## What TrueForge ships today

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

## Can bubblewrap reach a GPU?

Yes. This is worth establishing precisely, because the assumption that it cannot is
easy to make and would point at the wrong design.

Probing `cuInit` through `dlopen("libcuda.so.1")`, so nothing depends on the CUDA
toolkit being present. Every row runs in an unprivileged user namespace:

| bwrap configuration | `cuInit` |
| --- | --- |
| nvidia nodes `--dev-bind`, no `/sys` | `CUDA_ERROR_OPERATING_SYSTEM` (304) |
| `--ro-bind /dev` + `/sys` | `CUDA_ERROR_NO_DEVICE` (100) |
| `--dev-bind /dev` + `/sys` | **OK, devices=1** |
| `--ro-bind /dev`, nvidia nodes `--dev-bind`, + `/sys` | **OK, devices=1** |

Two requirements, and neither involves the namespace:

1. **`/sys` must be mounted.** Without it the driver fails with
   `CUDA_ERROR_OPERATING_SYSTEM`. This is the one that is easy to miss, because a
   sweep that varies only how `/dev` is handled fails on every row and looks
   controlled while holding the actual cause fixed.
2. **The nvidia device nodes must be writable.** A read-only `/dev` gives
   `CUDA_ERROR_NO_DEVICE`. Binding `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`
   and `/dev/nvidia-uvm-tools` read-write is enough, so `/dev` itself can stay
   read-only.

And the same 256 MiB `float4` copy kernel the harness uses elsewhere:

```
host:          KERNEL OK  920.7 GB/s
inside bwrap:  KERNEL OK  920.1 GB/s
```

The namespace is genuinely new and genuinely unprivileged: `/proc/self/ns/user` is
`4026536407` against the host's `4026531837`, and `/proc/self/uid_map` is
`1000 1000 1`. `nvidia-smi` works inside it. This is exactly what
`nvidia-container-toolkit` automates — the toolkit makes it ergonomic, not possible.

### What that implies for `LocalSandboxProvider`

`ALLOW_READ_BY_PLATFORM.linux` in `hostRun.ts` already lists `/dev`, `/proc` and
`/sys`, and `denySharedDefaultWritePaths()` deliberately keeps `/dev/*` out of the
deny list. So the shipped local provider may need little more than making the nvidia
nodes writable.

That is a hypothesis, not a finding. Creating a local sandbox on this host
pip-installs pydantic and the corporate proxy blocks it, so the shipped provider
could not be driven far enough to confirm. The mechanism result above is solid; the
claim about the provider specifically is not.

## Why a container provider anyway

GPU access is not the deciding factor, because bubblewrap has it. Two things are:

**Self-hosting.** `daytona` is the only exposed provider and can no longer be
deployed on your own infrastructure. A container provider is the only route to a
fully self-hosted TrueForge today. Requests for non-Daytona backends are open
upstream — E2B ([#387](https://github.com/truefoundry/trueforge/issues/387),
`help wanted`) and ComputeSDK
([#414](https://github.com/truefoundry/trueforge/issues/414), since closed).

**A pinned toolchain.** The sandbox is an image. A kernel benchmarked against CUDA
13.2 and the same kernel benchmarked against CUDA 12.4 are not the same measurement,
and a host-process sandbox inherits whatever the host has. For a harness whose entire
output is a performance number, that is the difference between a result and an
anecdote.

`LocalSandboxProvider` is the reference to port from: same `SandboxProvider`
interface, same path-shaped `sandboxId`, same file-transfer and traversal-validation
concerns, with `docker exec` replacing the bwrap supervisor. The shared contract suite
at `tests/core/sandbox/provider/sandboxProviderContractSuite.ts` already contains a
branch for path-id backends, so correctness is measured against tests the maintainers
wrote rather than tests we invented.

Submitted upstream as
[#466](https://github.com/truefoundry/trueforge/issues/466) and
[#467](https://github.com/truefoundry/trueforge/pull/467).

### One bug not to reproduce

Upstream issue [#416](https://github.com/truefoundry/trueforge/issues/416): the TFY
provider's `uploadFile` base64-encodes the whole payload into a single `argv`, so it
fails past roughly 96 KiB on `MAX_ARG_STRLEN` / `E2BIG`. Kernel projects move real
files. The container provider streams through `docker cp` / stdin instead.
