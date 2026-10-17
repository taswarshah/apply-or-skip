"""Variant 2: an LLM-driven ``strands.Agent`` with the ``ToolSelectionGate`` plug-in.

Here Nova Lite proposes tool calls the usual way; before each call the gate asks
Decider 2B whether that tool is the right next step and steers with ``Guide`` when it
confidently disagrees. Same tools and state as the deterministic loop in triage.py.
"""

from __future__ import annotations

from typing import Any

from strands import Agent, tool

from patterns.tool_selection import ToolSelectionGate

from . import tools as T
from .triage import Deps, run_tool, state_text

SYSTEM = ("You triage one job posting for the candidate. Use the tools one at a time: fetch, parse, score fit, "
          "then either tailor_resume or skip_job. Use ask_user if input is missing. Stop after a final tool.")


def build_gate_agent(source: str, profile: dict[str, Any] | None, deps: Deps, model: Any = None):
    st = T.TriageState(source=source, profile=profile)

    def _run(name: str) -> str:
        if st.done:
            return f"Triage already finished ({st.outcome}). Stop calling tools."
        out = run_tool(name, st, deps)
        st.last_result = out
        st.steps.append({"tool": name, "source": "llm_agent", "result": out[:300]})
        return out[:2000]

    @tool
    def fetch_posting() -> str:
        """Retrieve the job posting text from the job URL or file."""
        return _run("fetch_posting")

    @tool
    def parse_posting() -> str:
        """Summarize the fetched posting into structured requirements."""
        return _run("parse_posting")

    @tool
    def score_fit() -> str:
        """Check the candidate's fit and dealbreakers against the parsed requirements."""
        return _run("score_fit")

    @tool
    def tailor_resume() -> str:
        """Write a tailored resume and cover note. Only for jobs worth applying to."""
        return _run("tailor_resume")

    @tool
    def skip_job() -> str:
        """Record that this job should be skipped, with the reason."""
        return _run("skip_job")

    @tool
    def ask_user() -> str:
        """Ask the user for missing or unreadable input."""
        return _run("ask_user")

    gate = ToolSelectionGate(deps.router, state_fn=lambda event: state_text(st))
    if model is None:
        from strands.models import BedrockModel

        s = deps.settings
        model = BedrockModel(model_id=s.agent_model_id, region_name=s.aws_region, max_tokens=s.agent_max_tokens)
    agent = Agent(model=model, system_prompt=SYSTEM, callback_handler=None, interventions=[gate],
                  tools=[fetch_posting, parse_posting, score_fit, tailor_resume, skip_job, ask_user])
    return agent, gate, st
