#!/usr/bin/env python3
"""Generate index.html from the measured narration timings.

Ten scenes is too much hand-arithmetic to keep honest: every clip window, every audio cue
and every tween offset depends on how long the read actually is. So the composition is
generated from `audio/vo-timing.json`, measured from the rendered speech. Change a line,
regenerate the voice, rebuild -- the picture follows the read rather than the reverse.

Every number on screen is copied from the sweep in ../README.md. Nothing is illustrative.
"""

from __future__ import annotations

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
BG = "#0a0e14"
PANEL = "#141b24"
EDGE = "#232c38"
TEXT = "#e6edf3"
MUTE = "#8b949e"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"

# Section labels, so a judge can see which submission point each beat answers.
SECTION = {
    1: "The problem", 2: "The problem",
    3: "Architecture", 4: "Architecture", 5: "Architecture",
    6: "Demo", 7: "Demo", 8: "Demo",
    9: "What it taught", 10: "What it taught",
}


def esc_attr(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("&", "&amp;").replace('"', "&quot;")


# The bed is carved against the narration rather than EQ'd by hand. A voiceover carve
# takes only the bands the voice actually occupies, so the music keeps its low end and its
# top instead of going limp for the whole read -- and it follows the speech, so the bed
# comes back up between phrases.
#
# Written by `scripts/carve.mjs` from the hyperframes-audio skill, which measures both
# tracks and emits `data-fx-carve`, a chain of peaking filters and a gain stage, and the
# envelopes. Re-runnable: a re-carve replaces the previous one and leaves anything
# hand-authored in place.
#
# The first attempt here was a hand-written chain and it failed the render outright --
# `limiter` takes `limit` in dB and `release` in milliseconds, not `threshold` and
# seconds. A chain the engine cannot parse fails the whole mix by design rather than
# quietly writing the dry signal, which is the right call.
#
# Fades live in the WAV itself, built in the bgm step, so no volume lane is needed.
BED = 0.14
VOICE_GROUP = "voiceover"


def scene(n: int, body: str) -> str:
    s = S[n]
    return (
        f'      <section id="sc{n}" class="clip" data-start="{s["start"]}" '
        f'data-duration="{s["dur"]}" data-track-index="1">\n'
        f'        <div class="pad">\n'
        f'          <div class="sect">{SECTION[n]}</div>\n{body}\n'
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


# --- scene bodies ------------------------------------------------------------
def kicker(text: str) -> str:
    return f'          <div class="kick">{text}</div>'


def headline(text: str) -> str:
    return f'          <h1 class="head">{text}</h1>'


B = {}

B[1] = (
    kicker("February 2025 &middot; &ldquo;AI CUDA Engineer&rdquo;")
    + headline("It reported <em>10&ndash;100&times;</em> speedups.")
    + '''
          <div class="row">
            <div class="stat s-red"><div class="v" id="s1a">30&times;</div><div class="l">above what the hardware<br/>can physically deliver</div></div>
            <div class="stat"><div class="v">0</div><div class="l">kernels that were<br/>actually faster</div></div>
          </div>
          <div class="foot">The benchmark harness was exploited, not the GPU.</div>'''
)

B[2] = (
    kicker("Not a bug &mdash; an arrangement")
    + headline("One agent <em>writes</em> the kernel<br/>and <em>reports</em> its own speedup.")
    + '''
          <div class="note">
            <div class="nrow"><span class="x">&#10007;</span> correctness tests do not catch it</div>
            <div class="nrow sub">a kernel can be numerically perfect and still be timed dishonestly</div>
          </div>'''
)

B[3] = (
    kicker("kernel-preflight &middot; an agent inside TrueForge")
    + headline("It submits <em>source</em>.<br/>It never submits a <em>number</em>.")
    + '''
          <div class="owns">
            <div class="ol">the harness owns</div>
            <div class="chips">
              <span class="chip">allocation</span><span class="chip">input distribution</span>
              <span class="chip">the timing loop</span><span class="chip">the reference</span>
              <span class="chip">the tolerances</span>
            </div>
          </div>'''
)

B[4] = (
    kicker("Inside the sandbox")
    + headline("Two processes, and the split<br/>is the whole guarantee.")
    + '''
          <div class="split">
            <div class="proc p-ok">
              <div class="pt">supervisor</div>
              <div class="pl">holds the nonce + output path</div>
              <div class="pl">writes the verdict</div>
              <div class="pl strong">never loads candidate code</div>
            </div>
            <div class="arrow">&#8594;</div>
            <div class="proc p-bad">
              <div class="pt">worker</div>
              <div class="pl">links / imports the candidate</div>
              <div class="pl">reports on a descriptor</div>
              <div class="pl strong">told neither secret</div>
            </div>
          </div>'''
)

B[5] = (
    kicker("Nine gates")
    + headline("They adjudicate a <em>schema</em>,<br/>not a toolchain.")
    + '''
          <div class="gates">
            <span class="g">wellformed</span><span class="g">provenance</span><span class="g">correctness</span>
            <span class="g">timed_work</span><span class="g">liveness</span><span class="g">input_sensitivity</span>
            <span class="g">shape_consistency</span><span class="g">variance</span><span class="g">roofline</span>
          </div>
          <div class="tcs">
            <span class="tc">CUDA</span><span class="tc">Triton</span><span class="tc">Helion</span>
            <span class="tc">CuTe DSL</span><span class="tc">TileLang</span>
            <span class="tcn">five toolchains &middot; zero gate changes</span>
          </div>'''
)

B[6] = (
    kicker("cheat_silent_bf16 &middot; declares fp32, computes bf16")
    + headline("Caught three separate times.")
    + '''
          <div class="verdicts">
            <div class="vd" id="v6a"><span class="vg">correctness</span><span class="vv">594&times; over tolerance</span></div>
            <div class="vd" id="v6b"><span class="vg">timed_work</span><span class="vv">wrong after timing</span></div>
            <div class="vd hero" id="v6c"><span class="vg">roofline</span><span class="vv">100.9 TFLOP/s vs an 83.1 ceiling</span></div>
          </div>
          <div class="foot">The last one needs no reference output. The arithmetic is impossible at the precision claimed.</div>'''
)

B[7] = (
    kicker("One Triton FlashAttention kernel, unchanged")
    + headline("Honest at one precision,<br/>dishonest at another.")
    + '''
          <div class="prec">
            <div class="pc bad"><div class="pn">declared fp32</div><div class="pv">REJECTED</div></div>
            <div class="pc ok"><div class="pn">declared tf32</div><div class="pv">83.4%</div></div>
            <div class="pc ok"><div class="pn">declared bf16</div><div class="pv">88.7%</div></div>
          </div>
          <div class="foot"><code>tl.dot</code> silently uses tensor cores. Declaring what you actually compute is the difference.</div>'''
)

B[8] = (
    kicker("One sweep, one GPU, thermal cooldown between cases")
    + headline("Eighteen operations.")
    + '''
          <div class="row four">
            <div class="stat"><div class="v" id="s8a">67</div><div class="l">measured cases</div></div>
            <div class="stat s-green"><div class="v" id="s8b">51</div><div class="l">admitted</div></div>
            <div class="stat s-red"><div class="v" id="s8c">12/12</div><div class="l">adversarial rejected</div></div>
            <div class="stat"><div class="v" id="s8d">6</div><div class="l">toolchains</div></div>
          </div>
          <div class="ops">
            <span class="op">attention</span><span class="op">causal</span><span class="op">decode</span><span class="op">paged</span>
            <span class="op">gqa</span><span class="op">backward</span><span class="op">moe_gemm</span><span class="op">quantize</span>
            <span class="op">rope</span><span class="op">swiglu</span><span class="op">layernorm</span><span class="op">cross_entropy</span>
            <span class="op">gather</span><span class="op">matmul</span><span class="op">rmsnorm</span><span class="op">softmax</span>
            <span class="op">silu</span><span class="op">transpose</span>
          </div>'''
)

B[9] = (
    kicker("Then it caught me")
    + headline("Three attacks were <em>admitted</em><br/>before they were fixed.")
    + '''
          <div class="atk">
            <div class="ar"><span class="an">forged verdict, Python</span><span class="av">92% of peak</span></div>
            <div class="ar"><span class="an">forged verdict, CUDA constructor</span><span class="av">92% of peak</span></div>
            <div class="ar"><span class="an">cached answer served while timed</span><span class="av">89.7% of the bus</span></div>
          </div>
          <div class="foot">All three are regression tests now. A single process cannot both run a candidate and be trusted to report on it.</div>'''
)

B[10] = (
    kicker("The lesson worth keeping")
    + headline("A verifier is only as good as<br/>the attacks it has <em>survived</em>.")
    + '''
          <div class="closing">
            <div class="cl"><span class="cn">6&times;</span> I shipped a gate that rejected <em>correct</em> work</div>
            <div class="cl sub">which is worse than shipping no gate at all</div>
          </div>
          <div class="repo">github.com/rycerzes/kernel-preflight</div>'''
)

# --- emit --------------------------------------------------------------------
CSS = f"""
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      html, body {{ width:1920px; height:1080px; overflow:hidden; background:{BG}; }}
      body {{ font-family:"Inter","Helvetica Neue",Arial,sans-serif; color:{TEXT};
              -webkit-font-smoothing:antialiased; }}
      code, .mono {{ font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .clip {{ position:absolute; inset:0; width:1920px; height:1080px; }}
      .pad {{ position:absolute; inset:0; padding:110px 130px; display:flex;
              flex-direction:column; justify-content:center; }}
      .sect {{ position:absolute; top:64px; left:130px; font-size:20px; letter-spacing:.22em;
               text-transform:uppercase; color:{MUTE}; font-weight:600; }}
      .kick {{ font-size:26px; color:{BLUE}; letter-spacing:.04em; margin-bottom:26px;
               font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .head {{ font-size:82px; line-height:1.1; font-weight:700; letter-spacing:-.02em; }}
      .head em {{ font-style:normal; color:{AMBER}; }}
      .foot {{ margin-top:44px; font-size:28px; color:{MUTE}; line-height:1.45; max-width:1450px; }}
      .foot code {{ color:{TEXT}; font-size:26px; }}

      .row {{ display:flex; gap:80px; margin-top:60px; }}
      .row.four {{ gap:76px; }}
      .stat .v {{ font-size:96px; font-weight:700; letter-spacing:-.03em;
                  font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .stat .l {{ font-size:24px; color:{MUTE}; margin-top:10px; line-height:1.4; }}
      .s-red .v {{ color:{RED}; }}
      .s-green .v {{ color:{GREEN}; }}

      .note {{ margin-top:58px; }}
      .nrow {{ font-size:34px; display:flex; align-items:center; gap:18px; }}
      .nrow .x {{ color:{RED}; font-size:38px; }}
      .nrow.sub {{ font-size:26px; color:{MUTE}; margin-top:14px; padding-left:56px; }}

      .owns {{ margin-top:56px; }}
      .ol {{ font-size:22px; color:{MUTE}; letter-spacing:.16em; text-transform:uppercase;
             margin-bottom:20px; }}
      .chips {{ display:flex; flex-wrap:wrap; gap:16px; }}
      .chip {{ font-size:26px; padding:12px 24px; border:1px solid {EDGE}; border-radius:999px;
               background:{PANEL}; color:{TEXT}; }}

      .split {{ display:flex; align-items:center; gap:44px; margin-top:56px; }}
      .proc {{ flex:1; background:{PANEL}; border:1px solid {EDGE}; border-radius:16px; padding:32px 34px; }}
      .p-ok {{ border-left:5px solid {GREEN}; }}
      .p-bad {{ border-left:5px solid {RED}; }}
      .pt {{ font-size:32px; font-weight:700; margin-bottom:16px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .pl {{ font-size:23px; color:{MUTE}; line-height:1.7; }}
      .pl.strong {{ color:{TEXT}; font-weight:600; }}
      .arrow {{ font-size:46px; color:{MUTE}; }}

      .gates {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:52px; max-width:1500px; }}
      .g {{ font-size:24px; padding:11px 20px; border-radius:8px; background:{PANEL};
            border:1px solid {EDGE}; color:{TEXT};
            font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .tcs {{ display:flex; flex-wrap:wrap; align-items:center; gap:14px; margin-top:38px; }}
      .tc {{ font-size:24px; padding:10px 20px; border-radius:8px; border:1px solid {BLUE};
             color:{BLUE}; }}
      .tcn {{ font-size:24px; color:{MUTE}; margin-left:14px; }}

      .verdicts {{ margin-top:50px; display:flex; flex-direction:column; gap:18px; max-width:1500px; }}
      .vd {{ display:flex; align-items:baseline; gap:28px; background:{PANEL};
             border:1px solid {EDGE}; border-left:5px solid {RED}; border-radius:12px;
             padding:24px 30px; }}
      .vg {{ font-size:26px; color:{MUTE}; width:280px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .vv {{ font-size:34px; font-weight:600; }}
      .vd.hero .vv {{ color:{RED}; font-size:40px; }}

      .prec {{ display:flex; gap:34px; margin-top:56px; }}
      .pc {{ flex:1; border-radius:16px; padding:38px 34px; background:{PANEL}; border:1px solid {EDGE}; }}
      .pc.bad {{ border-left:5px solid {RED}; }}
      .pc.ok {{ border-left:5px solid {GREEN}; }}
      .pn {{ font-size:26px; color:{MUTE}; margin-bottom:16px;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .pv {{ font-size:62px; font-weight:700;
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .pc.bad .pv {{ color:{RED}; font-size:46px; }}
      .pc.ok .pv {{ color:{GREEN}; }}

      .ops {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:52px; max-width:1600px; }}
      .op {{ font-size:22px; padding:9px 17px; border-radius:7px; background:{PANEL};
             border:1px solid {EDGE}; color:{MUTE};
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}

      .atk {{ margin-top:52px; display:flex; flex-direction:column; gap:16px; max-width:1500px; }}
      .ar {{ display:flex; justify-content:space-between; align-items:baseline;
             background:{PANEL}; border:1px solid {EDGE}; border-left:5px solid {AMBER};
             border-radius:12px; padding:24px 32px; }}
      .an {{ font-size:29px; }}
      .av {{ font-size:34px; font-weight:700; color:{AMBER};
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}

      .closing {{ margin-top:56px; }}
      .cl {{ font-size:38px; display:flex; align-items:baseline; gap:22px; }}
      .cl em {{ font-style:normal; color:{AMBER}; }}
      .cn {{ font-size:60px; font-weight:700; color:{RED};
             font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
      .cl.sub {{ font-size:27px; color:{MUTE}; margin-top:14px; padding-left:104px; }}
      .repo {{ margin-top:64px; font-size:30px; color:{BLUE};
               font-family:"JetBrains Mono","SF Mono",Menlo,monospace; }}
"""

# One entrance per scene, offset to its own start so the timeline stays seekable.
#
# Selectors are emitted only when the scene actually contains them. GSAP warns on a
# target that matches nothing, and a composition that prints twelve warnings on every
# check trains you to stop reading them.
BODY_SELECTORS = [
    (".row > *", "class=\"row"), (".note > *", "class=\"note"),
    (".chips .chip", "class=\"chip"), (".split > *", "class=\"split"),
    (".gates .g", "class=\"gates"), (".tcs > *", "class=\"tcs"),
    (".verdicts .vd", "class=\"verdicts"), (".prec .pc", "class=\"prec"),
    (".ops .op", "class=\"ops"), (".atk .ar", "class=\"atk"),
    (".closing > *", "class=\"closing"), (".repo", "class=\"repo"),
]

tweens = []
for s in scenes:
    n, st = s["n"], s["start"]
    body = B[n]
    tweens.append(f'tl.from("#sc{n} .kick", {{opacity:0, y:-16, duration:.5, ease:"power2.out"}}, {st});')
    tweens.append(f'tl.from("#sc{n} .head", {{opacity:0, y:26, duration:.7, ease:"power3.out"}}, {st + 0.12});')
    present = [sel for sel, marker in BODY_SELECTORS if marker in body]
    if present:
        tweens.append(
            f'tl.from("#sc{n} :is({", ".join(present)})", '
            f'{{opacity:0, y:18, duration:.5, stagger:.045, ease:"power2.out"}}, {st + 0.4});'
        )
    if 'class="foot"' in body:
        tweens.append(f'tl.from("#sc{n} .foot", {{opacity:0, duration:.6}}, {st + 0.85});')

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
    print(f"  scene {s['n']:2d}  start {s['start']:6.2f}  dur {s['dur']:5.2f}  vo {s['vo']:5.2f}  [{SECTION[s['n']]}]")
