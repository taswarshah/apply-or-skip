"""Tool-selection eval over 40 LABELED SYNTHETIC postings (examples/postings + evals/labels.jsonl).

    python -m evals.run_eval --router both            # real Decider (DECIDER_URL) + real Bedrock
    python -m evals.run_eval --router both --mock     # offline plumbing check (keyword mock, NOT Decider 2B)

Routers compared on the same postings:
  decider : DeciderRouter (Decider 2B picks each next tool; Nova Micro only on low confidence)
  llm     : baseline where Nova Micro picks each next tool (one Bedrock call per step)
Fit checks (score_fit) use the Decider in both, so the comparison isolates next-tool routing.
Writes evals/results.md and runs/eval-<router>.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Any

from apply_or_skip.bedrock import TokenLedger
from apply_or_skip.config import Settings
from apply_or_skip.mocks import HeuristicDecider, MockGenerator
from apply_or_skip.triage import HARD_RULES, NEXT_TOOL, Deps, build_decider, build_generator, triage
from patterns.tool_selection import Decision, LLMRouter
from patterns.tool_selection.router import DeciderRouter

ROOT = Path(__file__).resolve().parents[1]
POSTINGS = ROOT / "examples" / "postings"
LABELS = ROOT / "evals" / "labels.jsonl"
PROFILE = ROOT / "examples" / "profile.example.yaml"


class MockRouteGenerator(MockGenerator):
    """Offline stand-in for Nova Micro as a router: applies the same keyword rules to the prompt's state."""

    def generate(self, purpose, system, prompt):
        if purpose == "route":
            state = prompt.split("STATE:\n", 1)[-1].rsplit("\n\nTool name:", 1)[0]
            names = [ln[2:].split(":", 1)[0] for ln in prompt.splitlines() if ln.startswith("- ")]
            self.route_reply = HeuristicDecider()._next_tool(state, names or NEXT_TOOL.names)
        return super().generate(purpose, system, prompt)


class LLMOnlyRouter(DeciderRouter):
    """Baseline: the LLM router picks every step (same option set, masking and hard rules)."""

    def next_tool(self, state, *, context=None, allowed=None):
        started = time.perf_counter()
        picked = self.fallback.choose(state, self.option_set.subset(allowed)) if self.fallback else None
        d = Decision(tool=picked or "ask_user", source="llm_router", p_top=1.0 if picked else 0.0,
                     latency_ms=round((time.perf_counter() - started) * 1000, 2))
        return self._apply_rules(d, context)


def load_labels() -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in LABELS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def make_deps(router_kind: str, settings: Settings, mock: bool, mock_bedrock: bool = False) -> Deps:
    ledger = TokenLedger.from_settings(settings)
    decider = HeuristicDecider() if mock else build_decider(settings)
    gen = MockRouteGenerator(ledger) if (mock or mock_bedrock) else build_generator(settings, ledger)
    fallback = LLMRouter(lambda p: gen.generate("route", "Reply with one tool name.", p).text) \
        if settings.llm_fallback or router_kind == "llm" else None
    cls = LLMOnlyRouter if router_kind == "llm" else DeciderRouter
    router = cls(decider, NEXT_TOOL, t_act=settings.t_act, t_floor=settings.t_floor, fallback=fallback,
                 hard_rules=HARD_RULES)
    return Deps(settings, decider, gen, router)


def run(router_kind: str, mock: bool, profile: dict[str, Any], limit: int | None = None,
        mock_bedrock: bool = False) -> dict[str, Any]:
    settings = Settings()
    settings = dataclasses.replace(settings, runs_dir=str(ROOT / "runs"),
                                   max_tailored_per_run=max(settings.max_tailored_per_run, 50))
    deps = make_deps(router_kind, settings, mock, mock_bedrock)
    rows, latencies = [], []
    for item in load_labels()[:limit]:
        st = triage(str(POSTINGS / item["posting"]), profile, deps)
        final = st.steps[-1]["tool"] if st.steps else None
        lat = [s["latency_ms"] for s in st.steps if s["source"] in ("decider", "llm_fallback", "low_confidence", "rule", "llm_router")]
        latencies += lat
        rows.append({"posting": item["posting"], "expected": item["expected_final_tool"], "reason": item["reason"],
                     "got": final, "correct": final == item["expected_final_tool"], "outcome": st.outcome,
                     "path": [s["tool"] for s in st.steps], "sources": [s["source"] for s in st.steps],
                     "fallbacks": sum(s["source"] == "llm_fallback" for s in st.steps)})
    led = deps.gen.ledger.summary()
    route = led["by_purpose"].get("route", {"calls": 0, "input": 0, "output": 0})
    n = len(rows)
    summary = {
        "router": router_kind, "mock": mock, "mock_bedrock": mock_bedrock or mock, "postings": n,
        "final_tool_accuracy": round(sum(r["correct"] for r in rows) / max(1, n), 3),
        "avg_steps": round(sum(len(r["path"]) for r in rows) / max(1, n), 2),
        "routing_llm_calls": route["calls"], "routing_llm_tokens": route["input"] + route["output"],
        "llm_fallback_steps": sum(r["fallbacks"] for r in rows),
        "decision_latency_ms_p50": round(statistics.median(latencies), 2) if latencies else None,
        "decision_latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
        "bedrock_calls_total": led["calls"], "bedrock_tokens_total": led["input_tokens"] + led["output_tokens"],
        "est_bedrock_cost_usd": led["est_cost_usd"],
    }
    out = ROOT / "runs" / f"eval-{router_kind}{'-mock' if mock else ''}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n" + json.dumps({"summary": summary}) + "\n")
    return {"summary": summary, "rows": rows}


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    mock = any(r["summary"]["mock"] for r in results)
    lines = ["# Eval results", "",
             "Dataset: 40 **labeled synthetic** postings (`examples/postings`, `evals/labels.jsonl`), "
             "fictional candidate profile.", ""]
    if not mock and any(r["summary"]["mock_bedrock"] for r in results):
        lines += ["> **Real Decider 2B, template Bedrock.** Routing and fit decisions come from the real checkpoint; "
                  "parse/tailor text is templated, so Bedrock token and cost numbers are not real.", ""]
    if mock:
        lines += ["> **OFFLINE MOCK RUN.** Decider = keyword heuristic, Bedrock = templates. These numbers only "
                  "prove the plumbing works; they say nothing about Strands Decider 2B or Nova quality. "
                  "Re-run on EC2 without `--mock`.", ""]
    keys = list(results[0]["summary"])
    lines += ["| metric | " + " | ".join(r["summary"]["router"] for r in results) + " |",
              "|---|" + "---|" * len(results)]
    for k in keys[4:]:
        lines.append(f"| {k} | " + " | ".join(str(r["summary"][k]) for r in results) + " |")
    lines += ["", "## Misses", ""]
    for r in results:
        for row in r["rows"]:
            if not row["correct"]:
                lines.append(f"- [{r['summary']['router']}] {row['posting']}: expected {row['expected']}, "
                             f"got {row['got']} via {' > '.join(row['path'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--router", choices=["decider", "llm", "both"], default="decider")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mock-bedrock", action="store_true",
                    help="real Decider, template Bedrock (set AOS_LLM_FALLBACK=0 for a Decider-only measurement)")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results.md"))
    args = ap.parse_args(argv)
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    kinds = ["decider", "llm"] if args.router == "both" else [args.router]
    results = [run(k, args.mock, profile, args.limit, args.mock_bedrock) for k in kinds]
    out = Path(args.out)
    if args.mock and out.name == "results.md":
        out = out.with_name("results-mock.md")
    write_markdown(results, out)
    for r in results:
        print(json.dumps(r["summary"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
