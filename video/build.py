#!/usr/bin/env python3
"""Generate index.html from the measured narration timings.

Eleven scenes is too much hand-arithmetic to keep honest: every clip window, every audio
cue and every tween offset depends on how long the read actually is. So the composition is
generated from `audio/vo-timing.json`, measured from the rendered speech. Change a line,
regenerate the voice, rebuild -- the picture follows the read rather than the reverse.

Every number on screen is copied from the sweep in ../README.md, from `thread_context_log`
in the running TrueForge instance, or from a live harness run. Nothing is illustrative, and
the demo beats are a recorded session replayed verbatim rather than a mock-up.

Design note: no rounded corners anywhere. This is an instrument readout, and the panels are
square, ruled and aligned to a single grid. Emphasis comes from rules and weight, not from
pills and cards.
"""

from __future__ import annotations

import html
import json
import pathlib

HERE = pathlib.Path(__file__).parent
TIMING = json.loads((HERE / "audio" / "vo-timing.json").read_text())

LEAD = 0.5   # silence before the first line
GAP = 0.40   # breath between beats
CUE = 0.25   # picture lands just before the voice starts
TAIL = 1.80  # hold after the last word

scenes: list[dict] = []
t = LEAD
for entry in TIMING:
    dur = entry["dur"] + CUE + (TAIL if entry["line"] == len(TIMING) else 0.0)
    scenes.append({
        "n": entry["line"],
        "start": round(t, 2),
        "dur": round(dur, 2),
        "cue": round(t + CUE, 2),
        "vo": entry["dur"],
    })
    t += dur + GAP
TOTAL = round(t + 0.3, 1)
S = {s["n"]: s for s in scenes}

# --- palette -----------------------------------------------------------------
BG = "#080b10"
PANEL = "#0e141c"
EDGE = "#1e2833"
EDGE2 = "#2b3746"
TEXT = "#e8eef5"
MUTE = "#7d8b9c"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
VIOLET = "#bc8cff"
DIM = "#727a84"    # meets 4.5:1 on BG; used for the scene counter and the split arrow

# Section labels, so a judge can see which submission point each beat answers.
SECTION = {
    1: "The problem", 2: "The problem",
    3: "The design",
    4: "Live demo", 5: "Live demo", 6: "Live demo", 7: "Live demo",
    8: "Why it holds", 9: "Why it holds",
    10: "Results",
    11: "What it taught",
}

BED = 0.14
VOICE_GROUP = "voiceover"


def scene(n: int, body: str) -> str:
    s = S[n]
    return (
        f'      <section id="sc{n}" class="clip" data-start="{s["start"]}" '
        f'data-duration="{s["dur"]}" data-track-index="1">\n'
        f'        <div class="pad">\n'
        f'          <div class="sect"><span class="srule"></span>{SECTION[n]}'
        f'<span class="sn">{n:02d} / {len(TIMING)}</span></div>\n{body}\n'
        f'        </div>\n      </section>'
    )


def audio_tags() -> str:
    rows = [
        f'      <audio id="vo-{s["n"]}" class="clip" src="audio/vo-{s["n"]}.wav" '
        f'data-start="{s["cue"]}" data-duration="{s["vo"]:.2f}" '
        f'data-audio-group="{VOICE_GROUP}" data-track-index="9"></audio>'
        for s in scenes
    ]
    rows.append(
        f'      <audio id="music-bed" class="clip" src="audio/bgm.wav" data-start="0" '
        f'data-duration="{TOTAL}" data-volume="{BED}" data-track-index="10"></audio>'
    )
    return "\n".join(rows)


# --- building blocks ---------------------------------------------------------
def kicker(text: str) -> str:
    return f'          <div class="kick">{text}</div>'


def headline(text: str) -> str:
    return f'          <h1 class="head">{text}</h1>'


def term(title: str, meta: str, lines: list[tuple[str, str]], cls: str = "") -> str:
    """A square terminal panel. `lines` are (class, raw-text) pairs, escaped here.

    Line classes: cmd, out, ok, bad, dim, warn, note, blank.
    """
    rows = []
    for kind, text in lines:
        if kind == "blank":
            rows.append('            <div class="tl blank"></div>')
        else:
            rows.append(f'            <div class="tl {kind}">{html.escape(text)}</div>')
    body = "\n".join(rows)
    return (
        f'          <div class="term {cls}">\n'
        f'            <div class="tbar"><span class="tdot"></span>'
        f'<span class="ttl">{html.escape(title)}</span>'
        f'<span class="tmeta">{html.escape(meta)}</span></div>\n'
        f'            <div class="tbody">\n{body}\n            </div>\n'
        f'          </div>'
    )


B = {}

# 1 -- the hook.
B[1] = (
    kicker("February 2025 &middot; Sakana AI")
    + headline('An agent reported <em>10&ndash;100&times;</em> speedups.<br/>The kernels were not faster.')
    + '          <div class="row">\n'
      f'            <div class="stat s-red"><div class="v">150&times;</div>'
      '<div class="l">headline speedup,<br/>later withdrawn</div></div>\n'
      f'            <div class="stat s-red"><div class="v">30&times;</div>'
      '<div class="l">more throughput than<br/>the hardware can deliver</div></div>\n'
      f'            <div class="stat"><div class="v">0</div>'
      '<div class="l">kernels that were<br/>actually faster</div></div>\n'
      '          </div>'
    + '          <div class="foot">It had exploited the benchmark harness. The numbers were '
      'real outputs of a real measurement &mdash; of the wrong thing.</div>'
)

# 2 -- the arrangement, not the model.
B[2] = (
    kicker("The failure is structural")
    + headline('One agent writes the kernel<br/>and reports its own speedup.')
    + '          <div class="note">\n'
      '            <div class="nrow"><span class="x">&times;</span>Correctness tests do not catch it</div>\n'
      '            <div class="nrow sub">A kernel can be numerically perfect and still be timed dishonestly</div>\n'
      '            <div class="nrow"><span class="x">&times;</span>Neither does a better model</div>\n'
      '            <div class="nrow sub">The incentive is in the arrangement, not in the weights</div>\n'
      '          </div>'
)

# 3 -- what the project is.
B[3] = (
    kicker("kernel-preflight &middot; an agent inside TrueForge")
    + headline('It submits <em>source</em>.<br/>It never submits a <em>number</em>.')
    + '          <div class="owns">\n'
      '            <div class="ol">A fixed harness the agent cannot see owns</div>\n'
      '            <div class="chips">\n'
      + "\n".join(
          f'              <div class="chip">{c}</div>'
          for c in ("allocation", "input distribution", "the timing loop",
                    "the reference", "the tolerances")
      )
      + '\n            </div>\n          </div>'
)

# 4 -- DEMO: the real ask, and the real device_spec reply.
B[4] = (
    kicker("Recorded session &middot; replayed verbatim")
    + headline('A real run.')
    + term(
        "trueforge — kernel-preflight", "session 01m0zc9y",
        [
            ("cmd", "Write the fastest fp32 matmul you can in Triton"),
            ("cmd", "and get it admitted by preflight_kernel."),
            ("blank", ""),
            ("dim", "· device_spec()"),
            ("out", '  { "name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9",'),
            ("out", '    "sm_count": 128, "peak_memory_bandwidth_gb_s": 1008.1,'),
            ("ok", '    "peak_fp32_tflops": 83.1 }'),
            ("blank", ""),
            ("dim", "· exec  cat skills/triton-kernels/SKILL.md"),
            ("out", "  Hugging Face's own Triton skill, pinned at 3b21db3"),
        ],
    )
)

# 5 -- DEMO: the agent spots the precision trap on its own.
B[5] = (
    kicker("Unprompted, in the agent's own reasoning")
    + headline('It finds the trap.')
    + '          <div class="quote">\n'
      '            <div class="qm">reasoning</div>\n'
      '            <div class="qt">&ldquo;<code>tl.dot</code> uses TF32 by default on sm_89. '
      'To get true fp32, I need <code>input_precision="ieee"</code> in <code>tl.dot</code>. '
      'This will be judged against the 83.1 TFLOP/s FP32 ceiling.&rdquo;</div>\n'
      '          </div>'
    + '          <div class="foot">Nothing in the prompt mentions TF32. Handed fp32 tensors, '
      'Triton routes through tensor cores and computes in something else &mdash; and the agent has '
      'to declare what it actually computes, not what it was given.</div>'
)

# 6 -- DEMO: two failures, the real traceback, then the admitted gate table.
B[6] = (
    kicker("Submit &rarr; fail &rarr; read the traceback &rarr; fix")
    + term(
        "preflight_kernel", "op=matmul  backend=triton  precision=fp32  repeats=30",
        [
            ("bad", "✗ harness exited 1"),
            ("out", "    TypeError: dynamic_func() got multiple values for argument 'M'"),
            ("dim", "  M, N, K passed positionally and again as keywords — the agent finds it and fixes it"),
            ("blank", ""),
            ("ok", "ADMITTED   NVIDIA GeForce RTX 4090"),
            ("pass", "  [pass] wellformed         4 shapes with every expected field"),
            ("pass", "  [pass] provenance         nonce echoed; 25470 ms of work inside a 25761 ms process"),
            ("pass", "  [pass] correctness        worst deviation 0.10x of the fp32 tolerance at 4096x4096"),
            ("pass", "  [pass] timed_work         the measured calls produced correct output"),
            ("pass", "  [pass] liveness           output written at every shape"),
            ("pass", "  [pass] input_sensitivity  output tracks the input at every shape"),
            ("pass", "  [pass] shape_consistency  behaves consistently across 4 shapes"),
            ("pass", "  [pass] variance           worst interquartile spread 1.06x over 30 repeats"),
            ("pass", "  [pass] roofline           peak fp32 pipelines utilisation 60.9% at 4096x4096"),
        ],
        cls="tall",
    )
)

# 7 -- DEMO: the same tool against a candidate that cheats the timing loop.
B[7] = (
    kicker("Same tool &middot; a kernel that serves a cached answer while timed")
    + term(
        "preflight_kernel", "candidate=cheat_cached_timed.py  op=rmsnorm  repeats=15",
        [
            ("bad", "REJECTED   NVIDIA GeForce RTX 4090"),
            ("pass", "  [pass] wellformed         5 shapes with every expected field"),
            ("pass", "  [pass] provenance         nonce echoed; 2405 ms of work inside a 2674 ms process"),
            ("pass", "  [pass] correctness        worst deviation 0.00x of the fp32 tolerance at 512x2048"),
            ("fail", "  [FAIL] timed_work         output after timing is wrong at 512x2048 (5.06e+05x tolerance);"),
            ("fail", "                            the measured calls did not do the same work as the warmup"),
            ("pass", "  [pass] liveness           output written at every shape"),
            ("pass", "  [pass] input_sensitivity  output tracks the input at every shape"),
            ("pass", "  [pass] shape_consistency  behaves consistently across 5 shapes"),
            ("pass", "  [pass] variance           worst interquartile spread 1.02x over 15 repeats"),
            ("pass", "  [pass] roofline           peak memory bus utilisation 90.0% at 8192x4096"),
        ],
        cls="tall",
    )
    + '          <div class="foot">Eight of nine pass. Correctness passes &mdash; the kernel really '
      'does compute rmsnorm. The gate that fails is the one that checks the <em>timed</em> calls did it.</div>'
)

# 8 -- why the verdict cannot be forged.
B[8] = (
    kicker("Inside the sandbox")
    + headline('Two processes.')
    + '          <div class="split">\n'
      f'            <div class="proc p-ok"><div class="pt">supervisor</div>'
      '<div class="pl strong">holds the nonce and the output path</div>'
      '<div class="pl">writes the verdict</div>'
      '<div class="pl">never loads candidate code</div></div>\n'
      '            <div class="arrow">&rarr;</div>\n'
      f'            <div class="proc p-bad"><div class="pt">worker</div>'
      '<div class="pl strong">links and runs the candidate</div>'
      '<div class="pl">reports on a file descriptor</div>'
      '<div class="pl">told neither secret</div></div>\n'
      '          </div>'
    + '          <div class="foot">Before the split, two candidates read the nonce out of '
      '<code>sys.argv</code> and <code>/proc/self/cmdline</code> and forged a verdict at '
      '<em>92% of peak</em> without launching a kernel. Both are regression tests now.</div>'
)

# 9 -- schema, not toolchain.
B[9] = (
    kicker("The gates adjudicate a measurement schema, not a toolchain")
    + '          <div class="tcs">\n'
      + "\n".join(f'            <div class="tc">{c}</div>'
                  for c in ("CUDA", "Triton", "Helion", "CuTe DSL", "TileLang", "torch"))
      + '\n            <div class="tcn">&mdash; five added after the gates were written. Zero gate changes.</div>\n'
      '          </div>'
    + '          <div class="prec">\n'
      '            <div class="pc bad"><div class="pn">declared fp32</div>'
      '<div class="pv">REJECTED</div><div class="pl">900&times; the fp32 tolerance</div></div>\n'
      '            <div class="pc ok"><div class="pn">declared tf32</div>'
      '<div class="pv">ADMITTED</div><div class="pl">64.7% of the tf32 ceiling</div></div>\n'
      '            <div class="pc ok"><div class="pn">declared bf16</div>'
      '<div class="pv">ADMITTED</div><div class="pl">61.2% of the bf16 ceiling</div></div>\n'
      '          </div>'
    + '          <div class="foot">One FlashAttention kernel, three submissions, byte-identical '
      'source. Declaring what you actually compute is the entire difference.</div>'
)

# 10 -- the sweep.
B[10] = (
    kicker("Full matrix &middot; RTX 4090, sm_89")
    + '          <div class="row four">\n'
      '            <div class="stat"><div class="v">18</div><div class="l">operations</div></div>\n'
      '            <div class="stat"><div class="v">6</div><div class="l">toolchains</div></div>\n'
      '            <div class="stat s-green"><div class="v">51</div><div class="l">admitted<br/>of 67 measured</div></div>\n'
      '            <div class="stat s-green"><div class="v">12/12</div><div class="l">adversarial<br/>kernels rejected</div></div>\n'
      '          </div>'
    + '          <div class="ops">\n'
      + "\n".join(f'            <div class="op">{o}</div>' for o in (
          "matmul", "attention", "attention_causal", "attention_decode", "attention_gqa",
          "attention_paged", "attention_backward", "moe_gemm", "rmsnorm", "layernorm",
          "softmax", "silu", "swiglu", "rope", "quantize", "gather", "cross_entropy",
          "transpose"))
      + '\n          </div>'
)

# 11 -- the honest close.
B[11] = (
    kicker("What building it taught")
    + '          <div class="closing">\n'
      '            <div class="cl"><span class="cn">3</span><span>attacks it '
      '<em>failed</em> before it caught them</span></div>\n'
      '            <div class="cl sub">Two forged the verdict at 92% of peak. One served a cache at 89.7%.</div>\n'
      '            <div class="cl"><span class="cn">6</span><span>times a gate rejected '
      '<em>correct</em> work</span></div>\n'
      '            <div class="cl sub">Which is worse than shipping no gate at all. All six are written up.</div>\n'
      '          </div>'
    + '          <div class="foot big">A verifier is only as good as the attacks it has actually survived.</div>'
    + '          <div class="repo">github.com/rycerzes/kernel-preflight</div>'
)

# --- emit --------------------------------------------------------------------
CSS = f"""
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      html, body {{ width:1920px; height:1080px; overflow:hidden; background:{BG}; }}
      body {{ font-family:"Inter","Helvetica Neue",Arial,sans-serif; color:{TEXT};
              -webkit-font-smoothing:antialiased; }}
      code, .mono {{ font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .clip {{ position:absolute; inset:0; width:1920px; height:1080px; }}
      .pad {{ position:absolute; inset:0; padding:104px 120px 96px; display:flex;
              flex-direction:column; justify-content:center; }}

      /* Section marker: a rule and a counter, not a badge. */
      .sect {{ position:absolute; top:58px; left:120px; right:120px; font-size:19px;
               letter-spacing:.24em; text-transform:uppercase; color:{MUTE}; font-weight:600;
               display:flex; align-items:center; gap:18px; }}
      .srule {{ display:inline-block; width:52px; height:2px; background:{BLUE}; }}
      .sn {{ margin-left:auto; color:{DIM}; letter-spacing:.18em;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}

      .kick {{ font-size:25px; color:{BLUE}; letter-spacing:.03em; margin-bottom:24px;
               font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .head {{ font-size:80px; line-height:1.08; font-weight:700; letter-spacing:-.025em; }}
      .head em {{ font-style:normal; color:{AMBER}; }}
      .foot {{ margin-top:40px; font-size:27px; color:{MUTE}; line-height:1.5; max-width:1500px;
               padding-left:20px; border-left:2px solid {EDGE2}; }}
      .foot code {{ color:{TEXT}; font-size:25px; }}
      .foot em {{ font-style:normal; color:{TEXT}; font-weight:600; }}
      .foot.big {{ font-size:36px; color:{TEXT}; border-left-color:{AMBER}; margin-top:52px; }}

      /* Stats: ruled columns, no cards. */
      .row {{ display:flex; gap:0; margin-top:58px; }}
      .row > * {{ flex:1; padding:0 44px; border-left:1px solid {EDGE}; }}
      .row > *:first-child {{ padding-left:0; border-left:0; }}
      .stat .v {{ font-size:94px; font-weight:700; letter-spacing:-.04em; line-height:1;
                  font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .stat .l {{ font-size:23px; color:{MUTE}; margin-top:16px; line-height:1.45; }}
      .s-red .v {{ color:{RED}; }}
      .s-green .v {{ color:{GREEN}; }}

      .note {{ margin-top:54px; }}
      .nrow {{ font-size:33px; display:flex; align-items:center; gap:18px; margin-top:26px; }}
      .nrow .x {{ color:{RED}; font-size:34px; width:24px; }}
      .nrow.sub {{ font-size:25px; color:{MUTE}; margin-top:10px; padding-left:42px; }}

      .owns {{ margin-top:52px; }}
      .ol {{ font-size:21px; color:{MUTE}; letter-spacing:.18em; text-transform:uppercase;
             margin-bottom:22px; }}
      .chips {{ display:flex; flex-wrap:wrap; gap:0; }}
      .chip {{ font-size:25px; padding:16px 30px; border:1px solid {EDGE}; margin:-1px 0 0 -1px;
               background:{PANEL}; color:{TEXT}; }}

      /* Terminal: square, ruled, monospace. The demo beats live here. */
      .term {{ margin-top:34px; border:1px solid {EDGE2}; background:{PANEL}; }}
      .term.tall {{ margin-top:26px; }}
      .tbar {{ display:flex; align-items:center; gap:14px; padding:14px 24px;
               border-bottom:1px solid {EDGE2}; background:#0a1017; }}
      .tdot {{ width:9px; height:9px; background:{GREEN}; display:inline-block; }}
      .ttl {{ font-size:20px; color:{TEXT}; letter-spacing:.04em;
              font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .tmeta {{ margin-left:auto; font-size:18px; color:{MUTE};
                font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .tbody {{ padding:26px 30px; }}
      .tl {{ font-size:23px; line-height:1.62; white-space:pre; letter-spacing:-.005em;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; color:{TEXT}; }}
      .term.tall .tl {{ font-size:21px; line-height:1.58; }}
      .tl.blank {{ height:16px; }}
      .tl.cmd {{ color:{TEXT}; font-weight:600; }}
      .tl.cmd::before {{ content:"> "; color:{BLUE}; }}
      .tl.out {{ color:#9fb0c3; }}
      .tl.dim {{ color:{MUTE}; }}
      .tl.ok {{ color:{GREEN}; font-weight:600; }}
      .tl.bad {{ color:{RED}; font-weight:600; }}
      .tl.pass {{ color:#9fb0c3; }}
      .tl.fail {{ color:{RED}; }}

      /* Reasoning pull-quote. */
      .quote {{ margin-top:40px; border-left:3px solid {VIOLET}; padding:6px 0 6px 34px; }}
      .qm {{ font-size:19px; letter-spacing:.22em; text-transform:uppercase; color:{VIOLET};
             margin-bottom:18px; font-weight:600; }}
      .qt {{ font-size:38px; line-height:1.48; color:{TEXT}; max-width:1560px; }}
      .qt code {{ font-size:34px; color:{AMBER}; }}

      /* Process split. */
      .split {{ display:flex; align-items:stretch; gap:34px; margin-top:50px; }}
      .proc {{ flex:1; background:{PANEL}; border:1px solid {EDGE}; padding:30px 32px; }}
      .p-ok {{ border-left:3px solid {GREEN}; }}
      .p-bad {{ border-left:3px solid {RED}; }}
      .pt {{ font-size:31px; font-weight:700; margin-bottom:18px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .pl {{ font-size:22px; color:{MUTE}; line-height:1.75; }}
      .pl.strong {{ color:{TEXT}; font-weight:600; }}
      .arrow {{ font-size:42px; color:{DIM}; align-self:center; }}

      /* Toolchains + precision contracts. */
      .tcs {{ display:flex; flex-wrap:wrap; align-items:center; gap:0; margin-top:34px; }}
      .tc {{ font-size:24px; padding:14px 28px; border:1px solid {BLUE}; color:{BLUE};
             margin:-1px 0 0 -1px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .tcn {{ font-size:23px; color:{MUTE}; margin-left:26px; }}
      .prec {{ display:flex; gap:0; margin-top:48px; }}
      .pc {{ flex:1; padding:34px 32px; background:{PANEL}; border:1px solid {EDGE};
             margin-left:-1px; }}
      .pc:first-child {{ margin-left:0; }}
      .pc.bad {{ border-top:3px solid {RED}; }}
      .pc.ok {{ border-top:3px solid {GREEN}; }}
      .pn {{ font-size:24px; color:{MUTE}; margin-bottom:16px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .pv {{ font-size:52px; font-weight:700; letter-spacing:-.02em;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .pc.bad .pv {{ color:{RED}; }}
      .pc.ok .pv {{ color:{GREEN}; }}
      .pc .pl {{ margin-top:14px; }}

      /* Operator grid. */
      .ops {{ display:flex; flex-wrap:wrap; gap:0; margin-top:52px; max-width:1680px; }}
      .op {{ font-size:21px; padding:12px 20px; background:{PANEL}; border:1px solid {EDGE};
             color:{MUTE}; margin:-1px 0 0 -1px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}

      /* Close. */
      .closing {{ margin-top:20px; }}
      .cl {{ font-size:37px; display:flex; align-items:baseline; gap:26px; margin-top:34px; }}
      .cl em {{ font-style:normal; color:{AMBER}; }}
      .cn {{ font-size:62px; font-weight:700; color:{RED}; min-width:84px; line-height:1;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .cl.sub {{ font-size:25px; color:{MUTE}; margin-top:12px; padding-left:110px; }}
      .repo {{ margin-top:52px; font-size:28px; color:{BLUE}; letter-spacing:.02em;
               font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
"""

# One entrance per scene, offset to its own start so the timeline stays seekable.
#
# Selectors are emitted only when the scene actually contains them. GSAP warns on a
# target that matches nothing, and a composition that prints warnings on every check
# trains you to stop reading them.
BODY_SELECTORS = [
    (".row > *", 'class="row'), (".note > *", 'class="note'),
    (".chips .chip", 'class="chip'), (".split > *", 'class="split'),
    (".tcs > *", 'class="tcs'), (".prec .pc", 'class="prec'),
    (".ops .op", 'class="ops'), (".closing > *", 'class="closing'),
    (".repo", 'class="repo'), (".quote", 'class="quote'),
]

tweens = []
for s in scenes:
    n, st = s["n"], s["start"]
    body = B[n]
    if 'class="kick"' in body:
        tweens.append(f'tl.from("#sc{n} .kick", {{opacity:0, y:-14, duration:.5, ease:"power2.out"}}, {st});')
    if 'class="head"' in body:
        tweens.append(f'tl.from("#sc{n} .head", {{opacity:0, y:24, duration:.7, ease:"power3.out"}}, {st + 0.12});')
    present = [sel for sel, marker in BODY_SELECTORS if marker in body]
    if present:
        tweens.append(
            f'tl.from("#sc{n} :is({", ".join(present)})", '
            f'{{opacity:0, y:16, duration:.5, stagger:.045, ease:"power2.out"}}, {st + 0.4});'
        )
    # The terminal panel settles, then prints its lines one at a time -- the whole
    # print is packed into the front of the beat so the table is readable and still
    # for most of the narration rather than arriving under the last word.
    if 'class="term' in body:
        n_lines = body.count('class="tl ')
        span = min(2.6, max(1.2, n_lines * 0.16))
        tweens.append(f'tl.from("#sc{n} .term", {{opacity:0, y:20, duration:.55, ease:"power2.out"}}, {st + 0.3});')
        tweens.append(
            f'tl.from("#sc{n} .tl", {{opacity:0, duration:.22, '
            f'stagger:{span / max(n_lines, 1):.3f}, ease:"none"}}, {st + 0.75});'
        )
    if 'class="foot' in body:
        tweens.append(f'tl.from("#sc{n} .foot", {{opacity:0, duration:.6}}, {st + 0.95});')

HTML = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>{CSS}    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{TOTAL}"
         data-width="1920" data-height="1080">
{chr(10).join(scene(n, B[n]) for n in sorted(B))}

{audio_tags()}
    </div>

    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      {chr(10).join("      " + t for t in tweens).strip()}
      window.__timelines["main"] = tl;
    </script>
  </body>
</html>
"""

(HERE / "index.html").write_text(HTML)
print(f"index.html: {len(scenes)} scenes, {TOTAL}s")
for s in scenes:
    print(f"  scene {s['n']:2d}  start {s['start']:6.2f}  dur {s['dur']:5.2f}  "
          f"vo {s['vo']:5.2f}  [{SECTION[s['n']]}]")
