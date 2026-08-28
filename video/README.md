# Demo video

The submission video is generated, not screen-recorded: `index.html` is a
[HyperFrames](https://hyperframes.heygen.com) composition — HTML with `data-*` timing
attributes and one paused GSAP timeline — rendered to MP4 with narration and a music bed.

Every number on screen is copied from the 67-case sweep in [../README.md](../README.md).
Nothing is illustrative.

**Output:** `out/kernel-preflight-demo.mp4` — 1920×1080, 30fps, 176.8s, h264 + AAC.

## Build

Order matters. `build.py` regenerates `index.html` from scratch, so the carve — which is
written *onto* that file — has to run after it:

```bash
python build.py                                   # windows recomputed from the measured read
node ~/.claude/skills/hyperframes-audio/scripts/carve.mjs --comp index.html
npx hyperframes check                             # lint, runtime, layout, motion, contrast
npx hyperframes snapshot --at 3,22,40,58,76,92,112,131,148,166
npx hyperframes render --quality high --output out/kernel-preflight-demo.mp4
```

Requires Node 22+ and FFmpeg. The delivered MP4 is committed so the submission is
self-contained; snapshots, narration WAVs and the retrieved music are not, since they
regenerate from `SCRIPT.md` and `audio_request.json`.

## Narration

Ten lines in [SCRIPT.md](SCRIPT.md), spoken by Kokoro `bm_george` — a local model, so no
API key and no per-render cost:

```bash
npx hyperframes tts "<line>" --voice bm_george --output audio/vo-1.wav
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
offset from those durations, so no beat outruns its sentence. Ten scenes is too much
hand-arithmetic to keep honest.

The first pass came in at 176.4s of speech, which overran the three-minute cap once gaps
were added. Four beats were trimmed rather than shrinking the gaps — a rushed read is more
obvious than a shorter one. Now 167.7s of speech inside a 176.8s cut.

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

The 60s pick needed looping to cover 176.8s, and catalog tracks fade out at the end — the
last three seconds of `f30e1301` sit 4.5 dB below its interior, so a naive loop inherits
that dip at every seam. The head 54s is taken instead, four copies joined with a 0.6s
crossfade, then the ends faded. The result is flat to **1.43 dB** across 171s, tighter than
the 2.24 dB source.

## The mix

The bed is **carved** against the narration rather than EQ'd by hand. A carve takes only
the bands the voice actually occupies, so the music keeps its low end and its top instead
of going limp for the whole read, and it follows the speech, so the bed comes back up
between phrases:

```bash
node ~/.claude/skills/hyperframes-audio/scripts/carve.mjs --comp index.html
```

It measured both tracks and wrote three peaking filters — 160 Hz −6 dB, 1.6 kHz −3 dB,
2.5 kHz −3 dB — a gain stage, and a 211-point level envelope with a −6 dB floor.

The carve names the **group**, not the ten clips: `sources: ["voiceover"]` against
`data-audio-group="voiceover"` on each narration clip. Naming clips one by one has to be
exhaustively right and stays right only until the next edit — an eleventh line added later
would play outside the carve's awareness and the bed would fail to duck under it silently.

**A hand-written chain was the first attempt and it failed the render outright**, with
`audio_processing_failed`. `limiter` takes `limit` in dB and `release` in **milliseconds**;
the guess used `threshold` and seconds. A chain the engine cannot parse fails the whole mix
rather than quietly writing the dry signal — the right call, and the reason to use the
carve script instead of hand-building the graph.

Measured on the delivered file:

| | |
| --- | --- |
| Bed alone, between lines | −37.0 dB (range −40.1 to −34.9) |
| With narration | −21.2 dB |
| Separation | **15.8 dB** |
| Peak | −3.4 dBFS |
| Music still present at 165s | −21.2 dB (no silent tail) |

Guidance for narration is a bed 12–18 dB under the voice; 15.8 dB sits in the middle of
that. The bed should be felt, not heard.

## Structure

Under the three-minute cap at **176.8s**, organised to the submission's four points.
Section labels sit on screen so a judge can see which point each beat answers.

| At | Section | Beat |
| --- | --- | --- |
| 0.5s | The problem | Sakana's 10–100×, and 30× above the hardware maximum |
| 18.4s | The problem | One agent writes the kernel and reports its own speedup |
| 36.0s | Architecture | Submits source, never a number; what the harness owns |
| 54.4s | Architecture | Supervisor and worker — the split is the guarantee |
| 72.7s | Architecture | Nine gates adjudicate a schema, so five toolchains cost nothing |
| 85.7s | Demo | The undeclared bf16 kernel, caught three separate ways |
| 108.3s | Demo | One kernel: rejected as fp32, admitted as tf32 and bf16 |
| 127.0s | Demo | Eighteen operations, 67 cases, 12 of 12 adversarial rejected |
| 144.5s | What it taught | Three attacks that were admitted before they were fixed |
| 160.2s | What it taught | Six false accusations; a verifier is only as good as its attacks |

## Verification

`npx hyperframes check` passes with 0 errors and 0 warnings, including **52/52 text checks
at WCAG AA**. Getting to zero warnings meant emitting tweens only for elements a scene
actually contains — GSAP warns on a target that matches nothing, and a composition that
prints twelve warnings on every check trains you to stop reading them.
