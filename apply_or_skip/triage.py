"""The deterministic agent loop: Decider 2B picks each next tool (no LLM routing calls)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from patterns.tool_selection import DeciderRouter, LLMRouter, OptionSet, build_state
from patterns.tool_selection.decider_client import DeciderClient, DeciderError, HttpDeciderClient

from . import tools as T
from .bedrock import BudgetExceeded, StrandsBedrockGenerator, TextGenerator, TokenLedger
from .config import Settings
from .fit_questions import evaluate_fit

GOAL = ("An agent is triaging one job posting for a candidate. It must fetch the posting, summarize it, "
        "check the candidate's fit, then either tailor a resume or skip the job.")

NEXT_TOOL = OptionSet(
    instructions="Which tool should the job triage agent run next?",
    options={
        "fetch_posting": "the posting text has not been retrieved yet",
        "parse_posting": "raw posting text exists but has not been summarized into requirements",
        "score_fit": "requirements are known but the candidate's fit has not been checked",
        "tailor_resume": "fit was checked, no dealbreaker was found, and the job is worth applying to",
        "skip_job": "a dealbreaker was found or the fit is clearly poor",
        "ask_user": "required input is missing or unreadable and only the user can supply it",
    },
    escape_option="ask_user",
)

TERMINAL = {"tailor_resume", "skip_job", "ask_user"}


# ------------------------------------------------------------------ hard rules
def rule_no_tailor_with_dealbreaker(st: T.TriageState, tool: str) -> str | None:
    if tool == "tailor_resume" and st.fit is not None and st.fit.dealbreakers:
        return "skip_job"
    return None


def rule_tailor_needs_fit_check(st: T.TriageState, tool: str) -> str | None:
    if tool == "tailor_resume" and st.fit is None:
        if st.parsed:
            return "score_fit"
        if st.fetch_failed:
            return "ask_user"
        return "parse_posting" if st.fetched else "fetch_posting"
    return None


def rule_needs_profile(st: T.TriageState, tool: str) -> str | None:
    if tool in {"score_fit", "tailor_resume"} and not st.profile:
        return "ask_user"
    return None


HARD_RULES = [rule_needs_profile, rule_tailor_needs_fit_check, rule_no_tailor_with_dealbreaker]


def allowed_tools(st: T.TriageState) -> list[str]:
    """Mask steps that already ran (action masking); the escape option is always kept."""
    done = {"fetch_posting": st.fetched and not st.fetch_failed, "parse_posting": st.parsed is not None,
            "score_fit": st.fit is not None}
    return [n for n in NEXT_TOOL.names if not done.get(n, False)]


def state_text(st: T.TriageState) -> str:
    return build_state(GOAL, {}, st.last_result, extra={"So far": st.narrative()}, last_result_max_tokens=600)


# ------------------------------------------------------------------ wiring
def build_decider(settings: Settings) -> DeciderClient:
    if settings.decider_backend == "mock":
        from .mocks import HeuristicDecider

        return HeuristicDecider()
    if settings.decider_backend == "inprocess":
        from patterns.tool_selection.decider_client import InProcessDeciderClient

        return InProcessDeciderClient(settings.decider_checkpoint, device=settings.decider_device)
    return HttpDeciderClient(settings.decider_url, timeout_s=settings.decider_timeout_s)


def build_generator(settings: Settings, ledger: TokenLedger, mock: bool = False) -> TextGenerator:
    if mock:
        from .mocks import MockGenerator

        return MockGenerator(ledger)
    return StrandsBedrockGenerator(settings, ledger)


def build_router(client: DeciderClient, gen: TextGenerator | None, settings: Settings) -> DeciderRouter:
    fallback = None
    if gen is not None and settings.llm_fallback:
        fallback = LLMRouter(lambda prompt: gen.generate("route", "Reply with one tool name.", prompt).text)
    return DeciderRouter(client, NEXT_TOOL, t_act=settings.t_act, t_floor=settings.t_floor,
                         fallback=fallback, hard_rules=HARD_RULES)


@dataclass
class Deps:
    settings: Settings
    decider: DeciderClient
    gen: TextGenerator
    router: DeciderRouter
    http_get: T.HttpGet = T.default_http_get

    @classmethod
    def create(cls, settings: Settings, *, mock_bedrock: bool = False, ledger: TokenLedger | None = None,
               decider: DeciderClient | None = None) -> Deps:
        ledger = ledger or TokenLedger.from_settings(settings)
        dec = decider or build_decider(settings)
        gen = build_generator(settings, ledger, mock=mock_bedrock)
        return cls(settings, dec, gen, build_router(dec, gen, settings))


# ------------------------------------------------------------------ loop
def run_tool(tool: str, st: T.TriageState, deps: Deps) -> str:
    s = deps.settings
    if tool == "fetch_posting":
        return T.fetch_posting(st, deps.http_get, s.max_posting_chars)
    if tool == "parse_posting":
        return T.parse_posting(st, deps.gen)
    if tool == "score_fit":
        if not st.parsed:
            return "SCORE SKIPPED: requirements are not parsed yet."
        st.fit = evaluate_fit(deps.decider, st.parsed, st.profile or {},
                              dealbreaker_threshold=s.dealbreaker_threshold, fit_threshold=s.fit_threshold)
        probs = ", ".join(f"{k}={v:.2f}" for k, v in st.fit.probabilities.items())
        return f"FIT CHECKED: {st.fit.summary()} ({probs})"
    if tool == "tailor_resume":
        return T.tailor_resume(st, deps.gen)
    if tool == "skip_job":
        return T.skip_job(st)
    return T.ask_user(st)


def triage(source: str, profile: dict[str, Any] | None, deps: Deps, on_step=None) -> T.TriageState:
    st = T.TriageState(source=source, profile=profile)
    for step in range(1, deps.settings.max_steps + 1):
        decision = deps.router.next_tool(state_text(st), context=st, allowed=allowed_tools(st))
        started = time.perf_counter()
        try:
            result = run_tool(decision.tool, st, deps)
        except BudgetExceeded as exc:
            st.outcome, st.output, result = "budget_stop", str(exc), f"BUDGET STOP: {exc}"
        except DeciderError as exc:
            st.outcome, st.output, result = "needs_user", f"Decider unavailable: {exc}", f"DECIDER ERROR: {exc}"
        record = {"step": step, **decision.to_dict(), "tool_ms": round((time.perf_counter() - started) * 1000, 1),
                  "result": result[:300]}
        st.steps.append(record)
        st.last_result = result
        if on_step:
            on_step(record, st)
        if st.done:
            return st
    st.outcome, st.output = "max_steps", f"stopped after {deps.settings.max_steps} steps"
    return st


def write_trace(st: T.TriageState, deps: Deps, runs_dir: str | Path) -> Path:
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{time.strftime('%Y%m%d-%H%M%S')}-{abs(hash(st.source)) % 10_000:04d}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in st.steps:
            fh.write(json.dumps({"type": "step", "source": st.source, **rec}) + "\n")
        fh.write(json.dumps({"type": "outcome", "source": st.source, "outcome": st.outcome, "output": st.output,
                             "fit": st.fit.to_dict() if st.fit else None,
                             "bedrock": deps.gen.ledger.summary()}) + "\n")
    return path
