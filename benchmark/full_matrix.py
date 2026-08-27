"""Run every candidate through the gates in one sitting, on one GPU, one session.

The point of doing it in a single sweep is that the numbers become comparable.
Figures collected across sessions are not: SM clock drifts with thermal state, and
a sweep run straight after Helion autotuning measures a throttled GPU.

Writes incrementally so a slow or hanging case cannot lose the earlier results.
"""
import json, pathlib, subprocess, sys, time, traceback
sys.path.insert(0, "src")
import preflight.runner as r
assert "/src/" in r.__file__, r.__file__

EX = pathlib.Path("src/preflight/harness/examples")
OUT = pathlib.Path("full_matrix.json")

# (file, op, backend, precision, what this case is here to show)
CASES = [
    # --- CUDA baselines: one per op the CUDA harness supports.
    ("baseline.cu",                   "rmsnorm",   "cuda",     "fp32", "hand-written CUDA"),
    ("baseline_softmax.cu",           "softmax",   "cuda",     "fp32", "hand-written CUDA"),
    ("baseline_silu.cu",              "silu",      "cuda",     "fp32", "hand-written CUDA"),
    ("baseline_transpose.cu",         "transpose", "cuda",     "fp32", "hand-written CUDA"),
    # --- One op, five toolchains: the comparison a schema-based gate makes possible.
    ("triton_rmsnorm.py",             "rmsnorm",   "triton",   "fp32", "Triton"),
    ("tilelang_rmsnorm.py",           "rmsnorm",   "tilelang", "fp32", "TileLang"),
    ("cute_rmsnorm.py",               "rmsnorm",   "cute",     "fp32", "CuTe DSL, one warp per row"),
    ("torch_rmsnorm.py",              "rmsnorm",   "torch",    "fp32", "eager PyTorch reference point"),
    # --- Elementwise across toolchains.
    ("tilelang_silu.py",              "silu",      "tilelang", "fp32", "TileLang"),
    ("cute_silu.py",                  "silu",      "cute",     "fp32", "CuTe DSL"),
    # --- Compute-bound: audited against the arithmetic ceiling, not the bus.
    ("triton_matmul.py",              "matmul",    "triton",   "fp32", "fp32 pinned via input_precision=ieee"),
    ("triton_matmul_tf32.py",         "matmul",    "triton",   "fp32", "TF32 while declaring fp32"),
    ("triton_matmul_tf32_declared.py","matmul",    "triton",   "tf32", "the same kernel, declared honestly"),
    # --- Attention, and one kernel under three precision contracts.
    ("torch_attention.py",            "attention", "torch",    "fp32", "SDPA"),
    ("torch_attention_bf16.py",       "attention", "torch",    "bf16", "SDPA"),
    ("triton_flash_attention.py",     "attention", "triton",   "fp32", "FlashAttention declared fp32"),
    ("triton_flash_attention.py",     "attention", "triton",   "tf32", "the same kernel, declared tf32"),
    ("triton_flash_attention.py",     "attention", "triton",   "bf16", "the same kernel, declared bf16"),
    # --- The two categories KernelBenchX finds hardest, plus LayerNorm's two-stage
    # --- reduction. Fusion fails 72% of the time across every method they tested and
    # --- Quantization is 0 of 30, so these are where a harness earns its keep.
    ("torch_layernorm.py",            "layernorm", "torch",    "fp32", "torch's fused layer_norm"),
    ("triton_layernorm.py",           "layernorm", "triton",   "fp32", "one pass, mean and variance together"),
    ("torch_swiglu.py",               "swiglu",    "torch",    "fp32", "unfused, materialises the intermediate"),
    ("triton_swiglu.py",              "swiglu",    "triton",   "fp32", "fused, one pass per input"),
    ("torch_quantize.py",             "quantize",  "torch",    "fp32", "unfused int8 round trip"),
    ("triton_quantize.py",            "quantize",  "triton",   "fp32", "exponent from the float bits"),
    # --- Wrong, but not dishonest. Must fail on correctness with a readable reason.
    ("wrong_transpose.cu",            "transpose", "cuda",     "fp32", "off by one row"),
    # --- Adversarial. Every one of these must be rejected.
    ("cheat_noop.cu",                 "rmsnorm",   "cuda",     "fp32", "writes nothing"),
    ("cheat_cached.cu",               "rmsnorm",   "cuda",     "fp32", "hardcodes the answer"),
    ("cheat_forge.cu",                "rmsnorm",   "cuda",     "fp32", "forges a verdict on stdout before main"),
    ("cheat_forge_proc.cu",           "rmsnorm",   "cuda",     "fp32", "forges a verdict from /proc/self/cmdline"),
    ("cheat_python_noop.py",          "rmsnorm",   "torch",    "fp32", "writes nothing"),
    ("cheat_python_forge.py",         "rmsnorm",   "torch",    "fp32", "forges a verdict from sys.argv at import"),
    ("cheat_timed_nan.py",            "rmsnorm",   "torch",    "fp32", "one true value and NaNs, only while timed"),
    ("cheat_silent_bf16.py",          "attention", "torch",    "fp32", "computes in bf16 while declaring fp32"),
    ("cheat_pay_the_clock.py",        "matmul",    "torch",    "fp32", "cached answer, caught on the compute axis"),
    ("cheat_cached_timed.py",         "rmsnorm",   "torch",    "fp32", "cached answer, memory-bound: needs the input drift"),
    ("cheat_truncating_quantize.py",  "quantize",  "torch",    "fp32", "truncates instead of rounding"),
    # --- Reduced storage precision, on the backends that can honour it.
    ("triton_rmsnorm.py",             "rmsnorm",   "triton",   "fp16", "fp16 storage"),
    ("tilelang_rmsnorm.py",           "rmsnorm",   "tilelang", "bf16", "bf16 storage"),
    ("tilelang_silu.py",              "silu",      "tilelang", "fp16", "fp16 storage"),
    ("cute_rmsnorm.py",               "rmsnorm",   "cute",     "bf16", "bf16 storage"),
    ("cute_rmsnorm.py",               "rmsnorm",   "cute",     "fp16", "fp16 storage"),
    ("triton_flash_attention.py",     "attention", "triton",   "fp16", "fp16 tensor cores"),
    # Last on purpose: autotuning runs for minutes and leaves the GPU hot.
    ("helion_rmsnorm.py",             "rmsnorm",   "helion",   "fp32", "Helion (autotuned)"),
]

def wait_until_cool(max_c=55.0, timeout_s=240):
    """Block until the GPU is back near idle temperature, or give up and say so.

    Deliberately *not* a clock check: an idle 4090 sits at 210 MHz of 3120, so
    "clock is near maximum" is false at rest and only becomes true under load.
    Temperature is the signal that means something between runs.

    This matters because the roofline ceiling is computed from peak clock, and
    measuring a thermally throttled GPU against it understates utilisation -- a
    sweep run straight after autotuning once read about a third of the throughput
    of the same kernel idle. The harness also samples the SM clock during each
    run and flags samples below 90% of peak, so this is belt and braces.
    """
    deadline = time.time() + timeout_s
    while True:
        # Every failure mode here -- nvidia-smi missing, a transient driver error,
        # empty output, an unparseable line -- has to be survivable. This runs
        # outside the per-case handler, so raising would abort the whole sweep and
        # discard the cases already recorded, which is the opposite of the
        # incremental-write guarantee the rest of this script makes.
        temp = None
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            if probe.returncode == 0:
                lines = [l.strip() for l in probe.stdout.splitlines() if l.strip()]
                if lines:
                    temp = float(lines[0])
        except (OSError, subprocess.SubprocessError, ValueError):
            temp = None
        if temp is None:
            return None  # cannot tell; let the caller note it and carry on
        if temp <= max_c:
            return temp
        if time.time() >= deadline:
            return None
        time.sleep(10)


results = []
for i, (fname, op, backend, precision, note) in enumerate(CASES, 1):
    if wait_until_cool() is None:
        print(f"  ... GPU still hot before case {i}; it may read low", flush=True)
    started = time.time()
    row = {"file": fname, "op": op, "backend": backend, "precision": precision, "note": note}
    try:
        p = r.preflight_file(EX / fname, op=op, backend=backend,
                             precision=precision, repeats=30).preflight
        m = p.measurement
        row.update(
            admitted=p.admitted,
            roofline=next(g.detail for g in p.gates if g.name == "roofline"),
            failures=[g.name for g in p.gates if g.blocking],
            sm_clock_ghz=round(m.get("sm_clock_hz", 0) / 1e9, 3),
            shapes=[{k: sh[k] for k in ("rows", "cols", "median_ms", "violation", "inner_iters")}
                    for sh in m["shapes"]],
        )
        verdict = "ADMITTED" if p.admitted else "rejected"
        detail = row["roofline"][:78] if p.admitted else ",".join(row["failures"])
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        row.update(admitted=False, error=message[:600], trace=traceback.format_exc()[-600:])
        # A candidate that cannot produce a measurement at all has been rejected,
        # not errored. That is the intended fate of a forgery: the supervisor is the
        # only writer, so a worker that exits early leaves nothing behind.
        if "without reporting a measurement" in message or "wrote no measurement" in message:
            row["rejected_before_gates"] = True
            verdict, detail = "rejected", "no measurement produced"
        else:
            verdict, detail = "ERROR", message[:78]
    row["elapsed_s"] = round(time.time() - started, 1)
    results.append(row)
    print(f"[{i:2d}/{len(CASES)}] {backend:9s} {precision:5s} {op:10s} {fname:32s} "
          f"{verdict:9s} {detail}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))

admitted = sum(1 for x in results if x.get("admitted"))
cheats = [x for x in results if x["file"].startswith("cheat_")]
errors = [x for x in results if x.get("error") and not x.get("rejected_before_gates")]
print(f"\n{len(results)} cases: {admitted} admitted, {len(results) - admitted} not")
print(f"adversarial: {sum(1 for x in cheats if not x.get('admitted'))}/{len(cheats)} rejected")
if errors:
    print(f"unexpected errors: {', '.join(x['file'] for x in errors)}")
