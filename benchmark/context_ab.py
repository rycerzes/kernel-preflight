"""A/B the harness's context management on the kernel-preflight task.

Hugging Face documents a specific failure for agentic kernel work: *"One common
issue is that the agent will not integrate the kernel at all. Typically because
the project's context is so long."* That is a context-engineering failure, and
context engineering is what TrueForge claims to provide. This measures whether
the claim holds on that exact workload.

Two configurations, identical but for the harness features under test:

  managed   compaction on, large tool responses offloaded to the sandbox
  raw       both off — every skill file and every gate report stays in context

The task is deliberately context-hungry: the `cuda-kernels` skill pack is large,
gate reports are verbose, and a kernel iteration re-reads both.

Reports completion rate first and tokens second. A cheaper run that fails is not
a better run, and a token count alone would hide that.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE = "http://127.0.0.1:8790/api/v1"

TASK = (
    "Write a vectorised RMSNorm kernel for sm_89, consult the cuda-kernels skill "
    "for guidance, submit it to preflight_kernel, and report the gate table."
)


def call(method: str, path: str, body: dict[str, Any] | None = None, timeout: int = 1800) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def agent_spec(*, managed: bool) -> dict[str, Any]:
    return {
        "model": {"name": "hf-router/glm-5.2", "params": {"temperature": 0.2, "max_tokens": 16384}},
        "instructions": (
            "You optimise GPU kernels. Consult the cuda-kernels skill, write a vectorised "
            "RMSNorm kernel for sm_89, and submit it to preflight_kernel. Report the gate "
            "table you get back. Never report a number the tool did not return."
        ),
        "mcp_servers": [
            {
                "name": "kernel-preflight",
                "enable_tools": ["@all"],
                "require_approval_for_tools": ["publish_kernel"],
                "preload": True,
            }
        ],
        "skills": [{"name": "cuda-kernels"}],
        "config": {
            "sandbox": {"enabled": True},
            "generative_ui": {"enabled": False},
            "ask_user_questions": {"enabled": False},
            "dynamic_sub_agents": {"enabled": False},
            "context_management": {
                "compaction": {"enabled": managed, "compaction_threshold_tokens": 50000},
                "large_tool_response": {"enabled": managed},
            },
            "iteration_limit": 60,
        },
    }


@dataclass
class Trial:
    arm: str
    completed: bool
    admitted: bool
    total_tokens: int
    input_tokens: int
    output_tokens: int
    wall_s: float
    failure: str | None = None


@dataclass
class Arm:
    name: str
    trials: list[Trial] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return sum(t.completed for t in self.trials) / len(self.trials) if self.trials else 0.0

    @property
    def admitted_rate(self) -> float:
        return sum(t.admitted for t in self.trials) / len(self.trials) if self.trials else 0.0

    def median(self, attr: str) -> float:
        values = [getattr(t, attr) for t in self.trials]
        return statistics.median(values) if values else 0.0


def run_trial(arm: str, managed: bool) -> Trial:
    session = call("POST", "/sessions", {"agent": {"spec": agent_spec(managed=managed)}})
    sid = session["data"]["id"]
    started = time.monotonic()
    turn = call(
        "POST",
        f"/sessions/{sid}/turns",
        {"input": [{"type": "user.message", "content": TASK}], "stream": False},
    )
    tid = turn["data"]["id"]

    events: list[dict[str, Any]] = []
    for _ in range(360):
        events = call("GET", f"/sessions/{sid}/turns/{tid}/events").get("data", [])
        if any(e.get("type") == "turn.done" for e in events):
            break
        time.sleep(5)
    wall = time.monotonic() - started

    done = next((e for e in events if e.get("type") == "turn.done"), None)
    state = (done or {}).get("state", {})
    metrics = state.get("metrics", {})
    completed = state.get("status") == "done"

    admitted = False
    for event in events:
        content = event.get("content")
        if isinstance(content, str) and '"admitted": true' in content.replace("'", '"'):
            admitted = True
        elif isinstance(content, str) and '"admitted":true' in content:
            admitted = True

    return Trial(
        arm=arm,
        completed=completed,
        admitted=admitted,
        total_tokens=metrics.get("total_tokens", 0),
        input_tokens=metrics.get("total_input_tokens", 0),
        output_tokens=metrics.get("total_output_tokens", 0),
        wall_s=wall,
        failure=None if completed else str(state.get("message"))[:120],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3, help="trials per arm")
    parser.add_argument("--out", default="benchmark/context_ab_results.json")
    args = parser.parse_args()

    arms = {"managed": Arm("managed"), "raw": Arm("raw")}
    # Interleaved rather than blocked, so drift in GPU state or provider latency
    # hits both arms rather than only the one that ran second.
    for index in range(args.trials):
        for name, managed in (("managed", True), ("raw", False)):
            print(f"[{index + 1}/{args.trials}] {name} ...", flush=True)
            trial = run_trial(name, managed)
            arms[name].trials.append(trial)
            print(
                f"    completed={trial.completed} admitted={trial.admitted} "
                f"tokens={trial.total_tokens} wall={trial.wall_s:.0f}s"
                + (f" failure={trial.failure}" if trial.failure else ""),
                flush=True,
            )

    print("\n" + "=" * 72)
    print(f"{'arm':<10}{'completed':>11}{'admitted':>10}{'med tokens':>12}{'med input':>11}{'med wall':>10}")
    print("=" * 72)
    for arm in arms.values():
        print(
            f"{arm.name:<10}{arm.completion_rate:>10.0%}{arm.admitted_rate:>10.0%}"
            f"{arm.median('total_tokens'):>12,.0f}{arm.median('input_tokens'):>11,.0f}"
            f"{arm.median('wall_s'):>9.0f}s"
        )

    payload = {
        "task": TASK,
        "trials_per_arm": args.trials,
        "arms": {
            name: {
                "completion_rate": arm.completion_rate,
                "admitted_rate": arm.admitted_rate,
                "median_total_tokens": arm.median("total_tokens"),
                "median_input_tokens": arm.median("input_tokens"),
                "median_wall_s": arm.median("wall_s"),
                "trials": [t.__dict__ for t in arm.trials],
            }
            for name, arm in arms.items()
        },
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
