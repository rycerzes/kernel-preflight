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

This was the preferred outcome -- a small diff to an existing provider rather than a
new one. `hostRun.ts` declares the Linux dependency set as `['bwrap', 'socat', 'rg']`,
and the Linux read policy already allows `/dev`, `/usr/lib`, `/usr/local`, `/proc`
and `/sys`. On paper the device nodes and driver libraries are already reachable.

They are reachable. The driver still refuses.

| bwrap configuration | `cuInit` result |
| --- | --- |
| `--dev /dev` (fresh devtmpfs, no nvidia nodes) | `CUDA_ERROR_NO_DEVICE` |
| `--dev-bind /dev /dev` | `CUDA_ERROR_OPERATING_SYSTEM` |
| `--dev /dev` + explicit `--dev-bind` per `/dev/nvidia*` node | `CUDA_ERROR_OPERATING_SYSTEM` |
| ...plus `/proc/driver/nvidia` rebound read-only | `CUDA_ERROR_OPERATING_SYSTEM` |

The first row confirms the harness is otherwise working: hide the nodes and CUDA
correctly reports no device. The rest show the nodes visible and initialisation
blocked anyway.

The cause is the user namespace. `/usr/bin/bwrap` is not setuid
(`-rwxr-xr-x root root`), so it always unshares into a new user namespace, and the
NVIDIA kernel driver rejects initialisation from one. This is not a policy that can
be relaxed with more bind mounts; it is why `nvidia-container-toolkit` exists at all.

`nvidia-container-cli` is installed on this host and can inject a GPU into an
existing namespace, which is how Podman and Singularity solve it. It needs
`CAP_SYS_ADMIN`. A sandbox provider that requires root to start a sandbox is not a
provider anyone should merge, so that route was rejected on design grounds rather
than attempted.

**Conclusion:** GPU support in the bubblewrap provider is a kernel/namespace research
problem, not a configuration fix. Out of scope.

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

Implement a **container-backed sandbox provider**. It is more code than a bind-mount
patch, but it has no unknown-unknowns, and it is the industry-standard answer to
this exact problem.

It also stands on its own merits regardless of GPUs. Requests for non-Daytona
sandbox backends exist upstream -- E2B ([#387](https://github.com/truefoundry/trueforge/issues/387),
`help wanted`, open) and ComputeSDK
([#414](https://github.com/truefoundry/trueforge/issues/414), since closed) -- and
now that Daytona cannot be self-hosted, a container provider is the only route to a
fully self-hosted TrueForge.

This landed upstream as
[#466](https://github.com/truefoundry/trueforge/issues/466) (the analysis) and
[#467](https://github.com/truefoundry/trueforge/pull/467) (the implementation).

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
