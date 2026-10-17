"""apply-or-skip CLI.

    apply-or-skip health
    apply-or-skip triage <url-or-file> --profile examples/profile.example.yaml
    apply-or-skip batch  <dir-or-list.txt> --profile ...
    apply-or-skip serve-ui --port 8080
    apply-or-skip eval   [--router decider|llm|both]

Add --mock-decider / --mock-bedrock (or --mock for both) to run fully offline with
keyword stand-ins (NOT the real models; for demos of the plumbing only).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import Settings
from .triage import Deps, triage, write_trace

try:
    from rich.console import Console
    from rich.table import Table

    console: Any = Console()
except ImportError:  # pragma: no cover
    console = None

ICON = {"decider": "D", "llm_fallback": "L", "low_confidence": "?", "rule": "R", "decider_error": "!"}


def load_profile(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def print_step(rec: dict[str, Any], _st=None) -> None:
    probs = sorted(rec.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:3]
    top = " ".join(f"{k}={v:.2f}" for k, v in probs)
    line = (f"[{ICON.get(rec['source'], ' ')}] step {rec['step']}: {rec['tool']:<14} p={rec['p_top']:.2f} "
            f"{rec['latency_ms']:.0f}ms  ({rec['source']})  {top}")
    print(line)
    print(f"      -> {rec['result'][:160].splitlines()[0] if rec['result'] else ''}")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    if getattr(args, "mock", False) or getattr(args, "mock_decider", False):
        s = dataclasses.replace(s, decider_backend="mock")
    return s


def _mock_bedrock(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "mock", False) or getattr(args, "mock_bedrock", False))


def _banner(s: Settings, mock_bedrock: bool) -> None:
    if s.decider_backend == "mock" or mock_bedrock:
        print("!! OFFLINE MOCK MODE: "
              + ("decider=keyword-heuristic (NOT Decider 2B) " if s.decider_backend == "mock" else "")
              + ("bedrock=templates (no AWS calls)" if mock_bedrock else ""), file=sys.stderr)


def cmd_triage(args: argparse.Namespace) -> int:
    s = _settings(args)
    _banner(s, _mock_bedrock(args))
    deps = Deps.create(s, mock_bedrock=_mock_bedrock(args))
    profile = load_profile(args.profile)
    if args.mode == "gate":
        from .gate_agent import build_gate_agent

        agent, gate, st = build_gate_agent(args.source, profile, deps)
        result = agent("Triage this job posting now.")
        u = result.metrics.accumulated_usage
        deps.gen.ledger.record(s.agent_model_id, "agent", int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0)))
        print(json.dumps({"gate_trace": gate.trace, "outcome": st.outcome}, indent=2))
    else:
        st = triage(args.source, profile, deps, on_step=None if args.json else print_step)
    path = write_trace(st, deps, s.runs_dir)
    summary = {"source": args.source, "outcome": st.outcome, "output": st.output[:4000],
               "bedrock": deps.gen.ledger.summary(), "trace": str(path)}
    if args.json:
        summary["steps"] = st.steps
    print(json.dumps(summary, indent=2) if args.json else
          f"\nOUTCOME: {st.outcome}\n{st.output}\n\nBedrock: {deps.gen.ledger.summary()}\nTrace: {path}")
    return 0


def _batch_sources(target: str) -> list[str]:
    p = Path(target)
    if p.is_dir():
        return sorted(str(x) for x in p.iterdir() if x.suffix in {".md", ".txt", ".html"})
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def cmd_batch(args: argparse.Namespace) -> int:
    s = _settings(args)
    _banner(s, _mock_bedrock(args))
    deps = Deps.create(s, mock_bedrock=_mock_bedrock(args))
    profile = load_profile(args.profile)
    rows = []
    for src in _batch_sources(args.target):
        st = triage(src, profile, deps)
        write_trace(st, deps, s.runs_dir)
        rows.append((src, st.outcome, len(st.steps), " > ".join(r["tool"] for r in st.steps), st.output[:70]))
    if console:
        t = Table(title="Apply-or-Skip batch")
        for col in ("posting", "outcome", "steps", "path", "note"):
            t.add_column(col)
        for r in rows:
            t.add_row(Path(r[0]).name, r[1] or "", str(r[2]), r[3], r[4].replace("\n", " "))
        console.print(t)
    else:  # pragma: no cover
        for r in rows:
            print(" | ".join(map(str, r)))
    print(json.dumps({"jobs": len(rows), "tailored": sum(r[1] == "tailored" for r in rows),
                      "skipped": sum(r[1] == "skipped" for r in rows), "bedrock": deps.gen.ledger.summary()}))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    s = _settings(args)
    from patterns.tool_selection import HttpDeciderClient

    try:
        print(json.dumps(HttpDeciderClient(s.decider_url, timeout_s=5).health(), indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"decider not reachable at {s.decider_url}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import serve

    s = _settings(args)
    _banner(s, _mock_bedrock(args))
    serve(Deps.create(s, mock_bedrock=_mock_bedrock(args)), load_profile(args.profile), host=args.host, port=args.port)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evals.run_eval import main as eval_main

    argv = ["--router", args.router] + (["--mock"] if args.mock else [])
    return eval_main(argv)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="apply-or-skip", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def mocks(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mock", action="store_true", help="offline: mock Decider AND mock Bedrock")
        p.add_argument("--mock-decider", action="store_true")
        p.add_argument("--mock-bedrock", action="store_true")

    p = sub.add_parser("triage", help="triage one posting")
    p.add_argument("source")
    p.add_argument("--profile", default=os.environ.get("AOS_PROFILE", "examples/profile.example.yaml"))
    p.add_argument("--mode", choices=["loop", "gate"], default="loop",
                   help="loop: Decider drives (default). gate: Nova Lite drives, ToolSelectionGate checks each call")
    p.add_argument("--json", action="store_true")
    mocks(p)
    p.set_defaults(fn=cmd_triage)

    p = sub.add_parser("batch", help="triage a directory of saved postings or a file of URLs")
    p.add_argument("target")
    p.add_argument("--profile", default=os.environ.get("AOS_PROFILE", "examples/profile.example.yaml"))
    mocks(p)
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("health", help="check the local Decider server")
    p.set_defaults(fn=cmd_health)

    p = sub.add_parser("serve-ui", help="tiny local web UI with the live decision trace")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--profile", default=os.environ.get("AOS_PROFILE", "examples/profile.example.yaml"))
    mocks(p)
    p.set_defaults(fn=cmd_ui)

    p = sub.add_parser("eval", help="run the tool-selection eval")
    p.add_argument("--router", choices=["decider", "llm", "both"], default="decider")
    p.add_argument("--mock", action="store_true")
    p.set_defaults(fn=cmd_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
