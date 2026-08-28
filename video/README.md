# Demo video

The submission video is generated, not screen-recorded: `index.html` is a
[HyperFrames](https://hyperframes.heygen.com) composition — HTML with `data-*` timing
attributes and one paused GSAP timeline — rendered to MP4 with narration and a music bed.

Every number on screen is copied from the 67-case sweep in [../README.md](../README.md).
Nothing is illustrative.

**Output:** `out/kernel-preflight-demo.mp4` — 1920×1080, 30fps, 165.2s, h264 + AAC.

Four of the eleven beats are a **recorded TrueForge session replayed verbatim**: the prompt,
the `device_spec` reply, the agent's own reasoning about TF32, the traceback it debugged and
the gate table it earned are copied out of `thread_context_log` in the running instance
(session `01m0zc9y`). The rejection beat is a live harness run against
`cheat_cached_timed.py`. Nothing in the demo is staged.

## Build

Order matters. `build.py` regenerates `index.html` from scratch, so the carve — which is
written *onto* that file — has to run after it:

```bash
python build.py                                   # windows recomputed from the measured read
node ~/.claude/skills/hyperframes-audio/scripts/carve.mjs --comp index.html
npx hyperframes check                             # lint, runtime, layout, motion, contrast
npx hyperframes snapshot --at 8,22,38,52,66,80,96,112,128,143,158
npx hyperframes render --quality high --output out/kernel-preflight-demo.mp4
```

Requires Node 22+ and FFmpeg. The delivered MP4 is committed so the submission is
self-contained; snapshots, narration WAVs and the retrieved music are not, since they
regenerate from `SCRIPT.md` and `audio_request.json`.

## Narration

Eleven lines in [SCRIPT.md](SCRIPT.md), spoken by Kokoro `bm_george` at **speed 1.25** — a
local model, so no API key and no per-render cost:

```bash
npx hyperframes tts --text-file line.txt --voice bm_george --speed 1.25 --output audio/vo-1.wav
```

Kokoro needs `kokoro-onnx` and `soundfile` in a Python environment, plus espeak-ng for
phonemisation. Keep them out of the project venv:

```bash
brew install espeak-ng
uv venv /tmp/ttsenv && uv pip install --python /tmp/ttsenv/bin/python kokoro-onnx==0.5.0 soundfile
export HYPERFRAMES_PYTHON=/tmp/ttsenv/bin/python
export ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
```

**Scene timing follows the read, not the reverse.** Each line is measured with `ffprobe`
into `audio/vo-timing.json`, and `build.py` derives every clip window, audio cue and tween
offset from those durations, so no beat outruns its sentence. Eleven scenes is too much
hand-arithmetic to keep honest.

The read is generated at **1.25× rather than retimed afterwards**. A constant
`data-playback-rate` would have worked for picture and pitch, but regenerating means the
measured durations are the real ones and every window still derives from them — no second
source of truth for timing. The first cut ran 167.7s of speech in 176.8s; this one carries
an extra beat and four demo panels in **155.4s of speech inside a 165.2s cut**.

Numbers are written as words in the script where a digit string would be mis-read: "float
thirty-two", not "fp32".

## Background music

Chosen by measurement, not by taking the top search result. Three catalog candidates long
enough to be worth looping were downloaded and scored on **loudness variance**, because a
bed that swells is a bed that distracts:

| Track | Length | Variance | Description |
| --- | --- | --- | --- |
| `f30e1301` | 60s | **2.24 dB** | minimalist ambient tech, inspiring and clean |
| `50209dd7` | 120s | 6.22 dB | subtle corporate technology background music |
| `90c8e017` | 158s | 6.72 dB | modern tech ambient, subtle and professional |

```bash
heygen auth login --oauth
heygen audio sounds list --query "subtle minimal technology background music" --limit 8
```

The 60s pick needed looping to cover the cut, and catalog tracks fade out at the end — the
last three seconds of `f30e1301` sit 4.5 dB below its interior, so a naive loop inherits
that dip at every seam. The head 54s is taken instead, four copies joined with a 0.6s
crossfade, then the ends faded. The result is flat to **1.43 dB** across 171s, tighter than
the 2.24 dB source. It is trimmed to the cut with the fade-out starting at 162.7s — after the
last word ends, not under it.

## The mix

The bed is **carved** against the narration rather than EQ'd by hand. A carve takes only
the bands the voice actually occupies, so the music keeps its low end and its top instead
of going limp for the whole read, and it follows the speech, so the bed comes back up
between phrases:

```bash
node ~/.claude/skills/hyperframes-audio/scripts/carve.mjs --comp index.html
```

It measured both tracks and wrote three peaking filters — 160 Hz −6 dB, 1.6 kHz −3 dB,
2.5 kHz −3 dB — a gain stage, and a 188-point level envelope with a −6 dB floor.

The carve names the **group**, not the eleven clips: `sources: ["voiceover"]` against
`data-audio-group="voiceover"` on each narration clip. Naming clips one by one has to be
exhaustively right and stays right only until the next edit — a twelfth line added later
would play outside the carve's awareness and the bed would fail to duck under it silently.

**A hand-written chain was the first attempt and it failed the render outright**, with
`audio_processing_failed`. `limiter` takes `limit` in dB and `release` in **milliseconds**;
the guess used `threshold` and seconds. A chain the engine cannot parse fails the whole mix
rather than quietly writing the dry signal — the right call, and the reason to use the
carve script instead of hand-building the graph.

Measured on the delivered file:

| | |
| --- | --- |
| Bed alone, between beats | −35.7 dB |
| With narration | −21.1 to −21.6 dB |
| Separation | **14.6 dB** |
| Peak | −3.3 dBFS |
| Final line, over the fade | −21.3 dB (bed fades after the word, not under it) |

Guidance for narration is a bed 12–18 dB under the voice; 14.6 dB sits inside
that. The bed should be felt, not heard.

## Structure

Under the three-minute cap at **165.2s**, organised to the submission's four points.
Section labels and an `nn / 11` counter sit on screen so a judge can see which point each
beat answers and how far in they are.

| At | Section | Beat |
| --- | --- | --- |
| 0.5s | The problem | Sakana's 10–100×, and 30× above the hardware maximum |
| 15.7s | The problem | One agent writes the kernel and reports its own speedup |
| 30.4s | The design | Submits source, never a number; what the harness owns |
| 45.4s | Live demo | The real prompt, the real `device_spec`, the pinned HF skill |
| 57.6s | Live demo | The agent finds the TF32 trap unprompted, in its own words |
| 72.6s | Live demo | Two failures, the real traceback, then nine gates green |
| 87.3s | Live demo | A cached-answer kernel: eight gates pass, `timed_work` fails |
| 103.0s | Why it holds | Supervisor and worker — the split is the guarantee |
| 120.3s | Why it holds | A schema, not a toolchain: six toolchains, one gate set |
| 136.1s | Results | Eighteen operations, 67 cases, 12 of 12 adversarial rejected |
| 150.9s | What it taught | Three attacks admitted before they were caught; six false accusations |

## Design

No rounded corners anywhere — `grep -c border-radius index.html` returns 0. The first cut
leaned on pill chips and 16px cards, which reads as generic. This one is an instrument
readout: square panels, 1px rules, a single grid, and emphasis carried by weight and a
left- or top-edge accent rather than by a card.

Two things are taken from the registry's `code-terminal-run` component rather than
invented: **authored token colouring** (the terminal's `cmd` / `out` / `ok` / `bad` /
`pass` / `fail` classes route to palette tokens, so nothing is parsed at runtime) and
**deterministic per-line printing** (lines reveal on a fixed stagger packed into the front
of the beat, so the table is still and readable for most of the narration). The panel
itself is inlined rather than mounted, because the component's default chrome is rounded
and the whole point of this pass was to remove that.

## Verification

`npx hyperframes check` passes with **0 errors** and **66/66 text checks at WCAG AA**
across lint, runtime, layout, motion and contrast.

One lint warning is left standing on purpose: `timeline_track_too_dense`, because all
eleven scenes are timed elements on one track. The suggested fix is to split them into
mounted sub-compositions. That is the right shape for a composition several people edit;
here the scenes are generated from one `build.py` and one timing file, so splitting them
would add eleven files and a mount contract without making anything easier to diff. It is
recorded rather than silenced.

Getting to zero *contrast* warnings took two passes: the scene counter and the process
arrow were first drawn in the panel-edge colour, which is 1.63:1 on this background. Both
moved to a token that clears 4.5:1. Getting to zero *motion* warnings meant emitting tweens
only for elements a scene actually contains — GSAP warns on a target that matches nothing,
and a composition that prints warnings on every check trains you to stop reading them.
