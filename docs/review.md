# Code review evidence


Every substantive change goes through a pull request reviewed by Qodo before merge.

| Round | Findings | Resolution |
| --- | --- | --- |
| [#1, first pass](https://github.com/rycerzes/kernel-preflight/pull/1) | 5 High, 3 Medium | 7 fixed, 1 partially fixed with a documented scope note. [Full response](https://github.com/rycerzes/kernel-preflight/pull/1#issuecomment-5414904562) |
| [#1, review of the fixes](https://github.com/rycerzes/kernel-preflight/pull/1) | 1 High, 2 Medium | all 3 fixed |
| [#2, preflight agent](https://github.com/rycerzes/kernel-preflight/pull/2) | 6 findings | all 6 fixed. [Full response](https://github.com/rycerzes/kernel-preflight/pull/2#issuecomment-5420739404) |
| [#3, benchmark and ops guide](https://github.com/rycerzes/kernel-preflight/pull/3) | 7 findings across two rounds | all 7 fixed |
| [#4, six ops and six backends](https://github.com/rycerzes/kernel-preflight/pull/4) | 9 findings | all 9 fixed; one of them falsified the premise a second time |

The second round reviewed the fixes themselves and found three second-order bugs,
which is the review loop doing its job:

- **Test deletes active sandboxes** (High) — the reaper test marked every labelled
  container stale, so it would delete live sandboxes belonging to another test
  worker or a running dev server on the same daemon. Fixed by making ownership
  scoped: containers carry a scope label, `reapStale` requires one, and a negative
  cutoff is now rejected outright. The regression test proves a bystander scope
  survives a reap.
- **Exit 124 conflated with timeout** (Medium) — GNU `timeout` exits 124 when it
  fires *and* passes a command's own 124 straight through, confirmed by
  experiment. Exit codes now ride in the successful envelope per the provider
  contract, and `timeout --verbose` supplies the stderr diagnostic that
  distinguishes a real timeout.
- **Oversized download reported missing** (Medium) — collapsing the size check and
  the read into one invocation removed a check/use window, but a file can still
  grow mid-`cat`, making the streaming cap the last line of defence. It killed the
  child without recording why, so an oversized file surfaced as missing.

Checking that round also surfaced a packaging problem Qodo had not raised: the GPU
test did not match upstream's `contract.test.ts$` ignore pattern, so it sat in the
default unit run and would have failed on any machine without Docker. Renamed to
follow the convention — infra-dependent tests are opt-in.

Findings were verified independently before being acted on, rather than accepted
on the reviewer's word. Two of the eight needed the claim narrowed:

- **Timeout leaves workload running** — real, and reproduced by hand
  (`timeout 2s sh -c "sleep 120"` → exit 124 in 2.0s, workload gone from `ps`).
  Fixed for the foreground workload; a deliberately detached child still outlives
  the bound, because `timeout` signals its child and not the process group. That
  residual is bounded by container removal and `reapStale`, and is documented at
  the call site rather than papered over.
- **Code mode admitted then crashes** — narrower than described, because
  `Sandbox.ts` gates Code Mode lazily via `codeModeEnv`. The failure is now a
  typed `CodeModeUnsupportedError`; full capability negotiation is out of scope
  for this PR.

The highest-impact finding was **cross-turn sandbox reuse**: the server builds a
fresh provider per turn and hands it a sandbox id carried over from an earlier
turn, so the original instance-local container map was empty exactly when it was
needed. Container names are now derived from the sandbox id.

Verifying the fixes also caught two bugs in the fixes themselves — a lookup left
keyed on the wrong tuple field, and a regression test whose `grep` matched its own
argv. Both are recorded in the PR response, because "the test passed" and "the fix
works" are different claims.

## The finding that invalidated the premise, twice

Review of the agent PR found that a candidate could **forge its own verdict**. The
harness printed its measurement to stdout, the candidate was linked into the same
binary, and C++ static constructors run before `main`. A kernel with an empty body,
whose constructor printed a fabricated measurement and exited, was **admitted at a
plausible 91% of peak** — six gates, all green, for code that never launched
anything.

That is not a bug in a gate. It falsified the sentence at the top of this file.

The first fix was to stop taking stdout at face value: the harness had to echo a
nonce chosen after the candidate's source was fixed, and account for wall-clock time
the caller independently observed. Fabricated work cannot pay for time that never
elapsed. Later the measurement moved out of stdout entirely, into a file, which
removed the channel `cheat_forge.cu` had used.

**That fix was not enough, and review of PR #4 found why.** The nonce and the output
path were still passed in `argv`, and the candidate was still running inside the
process that wrote the verdict. Two new candidates read them straight back out:

- `cheat_python_forge.py` — the Python harness imports candidates with
  `exec_module`, so module-level code runs before anything is measured, with
  `sys.argv` intact.
- `cheat_forge_proc.cu` — a C++ static constructor is not passed `argv`, but
  `/proc/self/cmdline` contains it.

Both wrote a measurement claiming 92% of the memory bus and exited. Both were
admitted, all nine gates green. The first fix had made forgery harder to reach
without making it impossible, which is the difference between a mitigation and a
boundary.

Hiding the values better does not work, and this was measured rather than assumed:
inside the container `/proc/<ppid>/cmdline` and `/proc/<ppid>/environ` are readable,
while `/proc/<ppid>/mem` and `ptrace(PTRACE_ATTACH)` are refused with `EPERM`. So no
flag and no environment variable can carry a secret past code running there — but a
*sibling process* can hold one safely.

That is the fix. `supervisor.py` takes the nonce and the output path on stdin,
spawns the worker with neither, and is the only writer. It also stamps the wall time
it observed itself, so the provenance check that the timed loops fit inside the run
is finally made against a duration the measured code does not control. A worker that
exits early now leaves nothing behind — which the runner already treats as a
rejection, and which is exactly what both forges now do.

Both attacks are committed as regression tests, and `cheat_forge_proc.cu` now fails
for the reason that matters: `argv` carries neither the nonce nor the output path any
more, so there is nothing in `/proc/self/cmdline` worth reading. The defence is which
process knows what, not how well a value is hidden inside one.

The residual limit is stated in [what it does not claim](results.md#what-it-does-not-claim).

The same review round also found that the tf32 correctness bar was 128x looser than
the contract, that five of six input builders silently ignored a declared reduced
precision, and that a merely *wrong* transpose came back as a JSON parse error
instead of a correctness verdict. Nine findings, all fixed, each verified against
the code or the hardware before being acted on — one of them,
[a claim about GA10x tensor ceilings](https://forums.developer.nvidia.com/t/tf32-tflops-of-geforce-rtx-3090-vs-a40/265828),
turned out to be a real risk of *falsely accusing* correct kernels.

The earlier review also found that candidate code was being compiled and executed
**on the host**, in the process environment holding the publishing token — while
this repository was contributing a GPU container sandbox to TrueForge. Execution now
happens in a container with no network and no inherited environment.
