"""Device capability probe.

Every gate in this package compares a *claim* against a *physical ceiling*. That
ceiling must come from the hardware, not a hardcoded table -- a hardcoded table is
how a benchmark ends up validating itself against a number someone typed in
optimistically.

Values come from the CUDA driver attribute API via ctypes. Two reasons not to use
the obvious alternatives: CUDA 13 removed ``cudaDeviceProp::memoryClockRate`` so
the runtime struct no longer carries it, and ``nvidia-smi`` has no
``memory.bus_width`` query field at all. The driver attributes are the
authoritative source and are stable across CUDA majors.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import asdict, dataclass
from typing import Any

# CUdevice_attribute values from cuda.h. Stable ABI constants.
_ATTR_SM_CLOCK_KHZ = 13
_ATTR_SM_COUNT = 16
_ATTR_MEMORY_CLOCK_KHZ = 36
_ATTR_MEMORY_BUS_WIDTH_BITS = 37

# FP32 lanes per SM, keyed by full compute capability rather than major version.
# Major alone is wrong: GA100 (8.0, the A100) has 64 FP32 lanes per SM, while the
# GA10x consumer parts (8.6/8.7) and Ada (8.9) have 128. Keying on major would
# double the A100's compute ceiling and weaken every gate that depends on it.
#
# Anything absent here refuses to guess -- see peak_fp32_flops.
_FP32_LANES_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 64,   # Volta  V100
    (7, 2): 64,   # Xavier
    (7, 5): 64,   # Turing T4, RTX 20xx
    (8, 0): 64,   # Ampere GA100 (A100)
    (8, 6): 128,  # Ampere GA10x (RTX 30xx, A40)
    (8, 7): 128,  # Orin
    (8, 9): 128,  # Ada    (RTX 40xx, L40S)
    (9, 0): 128,  # Hopper H100
    (10, 0): 128,  # Blackwell B100/B200
    (12, 0): 128,  # Blackwell consumer (RTX 50xx)
}


class CudaDriverError(RuntimeError):
    """A libcuda call failed, or libcuda is not present."""


def _load_libcuda() -> ctypes.CDLL:
    for candidate in ("libcuda.so.1", "libcuda.so", ctypes.util.find_library("cuda")):
        if candidate is None:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise CudaDriverError("libcuda not found - is an NVIDIA driver installed?")


def _check(lib: ctypes.CDLL, code: int, call: str) -> None:
    if code == 0:
        return
    name = ctypes.c_char_p()
    try:
        lib.cuGetErrorName(ctypes.c_int(code), ctypes.byref(name))
        detail = name.value.decode() if name.value else f"code {code}"
    except Exception:  # pragma: no cover - error path of an error path
        detail = f"code {code}"
    raise CudaDriverError(f"{call} failed: {detail}")


@dataclass(frozen=True)
class DeviceSpec:
    """Physical ceilings for one CUDA device."""

    name: str
    compute_capability: str
    sm_count: int
    total_memory_bytes: int
    memory_bus_width_bits: int
    memory_clock_khz: int
    sm_clock_khz: int

    @property
    def peak_memory_bandwidth_bytes_per_s(self) -> float:
        """GDDR transfers twice per clock: 2 * clock * (bus_width / 8)."""
        return 2.0 * (self.memory_clock_khz * 1e3) * (self.memory_bus_width_bits / 8.0)

    @property
    def peak_fp32_flops(self) -> float | None:
        """None when the lane count for this architecture is unknown.

        Returning None forces callers to skip the compute-bound ceiling rather than
        silently audit against a fabricated number.
        """
        major_text, _, minor_text = self.compute_capability.partition(".")
        try:
            capability = (int(major_text), int(minor_text))
        except ValueError:
            return None
        lanes = _FP32_LANES_PER_SM.get(capability)
        if lanes is None:
            return None
        return self.sm_count * lanes * 2 * (self.sm_clock_khz * 1e3)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["peak_memory_bandwidth_gb_s"] = round(self.peak_memory_bandwidth_bytes_per_s / 1e9, 1)
        flops = self.peak_fp32_flops
        d["peak_fp32_tflops"] = round(flops / 1e12, 1) if flops is not None else None
        return d


def probe(device_index: int = 0) -> DeviceSpec:
    """Read physical ceilings for ``device_index`` straight from the driver."""
    lib = _load_libcuda()
    _check(lib, lib.cuInit(0), "cuInit")

    dev = ctypes.c_int()
    _check(lib, lib.cuDeviceGet(ctypes.byref(dev), device_index), "cuDeviceGet")

    def attr(which: int) -> int:
        out = ctypes.c_int()
        _check(lib, lib.cuDeviceGetAttribute(ctypes.byref(out), which, dev), f"cuDeviceGetAttribute({which})")
        return out.value

    name_buf = ctypes.create_string_buffer(256)
    _check(lib, lib.cuDeviceGetName(name_buf, len(name_buf), dev), "cuDeviceGetName")

    total = ctypes.c_size_t()
    _check(lib, lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev), "cuDeviceTotalMem")

    major = ctypes.c_int()
    minor = ctypes.c_int()
    _check(lib, lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev), "cuDeviceComputeCapability")

    return DeviceSpec(
        name=name_buf.value.decode(),
        compute_capability=f"{major.value}.{minor.value}",
        sm_count=attr(_ATTR_SM_COUNT),
        total_memory_bytes=total.value,
        memory_bus_width_bits=attr(_ATTR_MEMORY_BUS_WIDTH_BITS),
        memory_clock_khz=attr(_ATTR_MEMORY_CLOCK_KHZ),
        sm_clock_khz=attr(_ATTR_SM_CLOCK_KHZ),
    )
