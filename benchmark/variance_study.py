"""How many repeats does the variance gate need before its own statistic is stable?

The gate rejects when p75/p25 exceeds 1.5. An honest TileLang kernel was rejected at
repeats=15 and passed three times at repeats=30 with a ratio of 1.04. So the
threshold is not the problem; the estimator is, at low sample counts. This measures
where it settles instead of guessing.
"""
import json, pathlib, statistics, sys, time
sys.path.insert(0, "src")
import preflight.runner as r
assert "/src/" in r.__file__, r.__file__

CANDIDATE = "src/preflight/harness/examples/tilelang_silu.py"
REPEATS = [5, 8, 10, 15, 20, 30, 50]
TRIALS = 4

rows = []
for n in REPEATS:
    for t in range(TRIALS):
        p = r.preflight_file(CANDIDATE, op="silu", backend="tilelang",
                             precision="bf16", repeats=n).preflight
        ratios = [s["p75_ms"] / s["p25_ms"] for s in p.measurement["shapes"] if s["p25_ms"] > 0]
        rows.append({
            "repeats": n, "trial": t, "worst_ratio": max(ratios),
            "ratios": ratios, "admitted": p.admitted,
            "variance_status": next(g.status.value for g in p.gates if g.name == "variance"),
        })
        print(f"  repeats={n:3d} trial={t} worst p75/p25={max(ratios):.3f} "
              f"admitted={p.admitted}", flush=True)
        pathlib.Path("variance_study.json").write_text(json.dumps(rows, indent=2))

print("\nrepeats  worst-of-trials  median  spread(max-min)  rejections")
for n in REPEATS:
    got = [x["worst_ratio"] for x in rows if x["repeats"] == n]
    rej = sum(1 for x in rows if x["repeats"] == n and not x["admitted"])
    print(f"{n:7d}  {max(got):15.3f}  {statistics.median(got):6.3f}  "
          f"{max(got) - min(got):15.3f}  {rej}/{TRIALS}")
