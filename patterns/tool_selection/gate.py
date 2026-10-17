"""ToolSelectionGate: a drop-in Strands ``InterventionHandler`` for tool selection.

Use it when an LLM already drives your ``strands.Agent`` and you want Decider 2B to
second-guess each tool call:

    gate = ToolSelectionGate(router)                       # router: DeciderRouter
    agent = Agent(model=..., tools=[...], interventions=[gate])

In ``before_tool_call`` the gate asks the Decider which option fits the current state:
  * Decider agrees with the LLM's tool             -> ``Proceed()``
  * Decider confidently (p >= guide_threshold)
    prefers another tool                           -> ``Guide(feedback="Call X instead: ...")``
    (Strands cancels the call and shows the feedback to the model, which retries)
  * Decider unsure, tool not in the option set,
    or the guide limit was reached                 -> ``Proceed()`` and log it

Verified against strands-agents 1.61.1: ``strands.interventions.InterventionHandler``,
``Proceed``, ``Guide``; ``BeforeToolCallEvent.tool_use`` / ``.agent``; ``Agent(interventions=[...])``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from strands.hooks.events import BeforeToolCallEvent
from strands.interventions import Guide, InterventionHandler, Proceed

from .decider_client import DeciderError
from .question import truncate
from .router import DeciderRouter

logger = logging.getLogger(__name__)

StateFn = Callable[[BeforeToolCallEvent], str]


def default_state_from_messages(event: BeforeToolCallEvent, max_messages: int = 6, max_tokens: int = 1200) -> str:
    """Render the tail of the agent conversation as compact text."""
    lines: list[str] = []
    for msg in list(getattr(event.agent, "messages", []))[-max_messages:]:
        for block in msg.get("content", []):
            if "text" in block:
                lines.append(f"{msg['role']}: {block['text']}")
            elif "toolUse" in block:
                tu = block["toolUse"]
                lines.append(f"{msg['role']} called {tu.get('name')}({json.dumps(tu.get('input', {}))[:200]})")
            elif "toolResult" in block:
                tr = block["toolResult"]
                texts = " ".join(c.get("text", "") for c in tr.get("content", []) if isinstance(c, dict))
                lines.append(f"tool result [{tr.get('status')}]: {texts}")
    return truncate("\n".join(lines), max_tokens)


class ToolSelectionGate(InterventionHandler):
    name = "tool-selection-gate"

    def __init__(
        self,
        router: DeciderRouter,
        *,
        state_fn: StateFn | None = None,
        guide_threshold: float | None = None,
        max_guides_per_turn: int = 2,
    ) -> None:
        self.router = router
        self.state_fn = state_fn or default_state_from_messages
        self.guide_threshold = router.t_act if guide_threshold is None else guide_threshold
        self.max_guides_per_turn = max_guides_per_turn
        self._guides_in_a_row = 0
        self.trace: list[dict[str, Any]] = []

    @property
    def on_error(self) -> str:
        # A broken Decider must not take the agent down; log and let the LLM's call through.
        return "proceed"

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Guide:
        proposed = event.tool_use["name"]
        record: dict[str, Any] = {"proposed": proposed}
        if proposed not in self.router.option_set.options:
            record["action"] = "proceed_unmanaged"
            self.trace.append(record)
            return Proceed(reason="tool not managed by the gate")
        try:
            decision = self.router.ask_decider(self.state_fn(event))
        except DeciderError as exc:
            record.update(action="proceed_decider_error", error=str(exc)[:200])
            self.trace.append(record)
            return Proceed(reason="decider unavailable")
        record.update(decider=decision.decider_choice, p_top=round(decision.p_top, 4),
                      latency_ms=decision.latency_ms)

        if decision.tool == proposed:
            self._guides_in_a_row = 0
            record["action"] = "proceed_agree"
            self.trace.append(record)
            return Proceed(reason=f"decider agrees (p={decision.p_top:.2f})")

        if decision.p_top >= self.guide_threshold and self._guides_in_a_row < self.max_guides_per_turn:
            self._guides_in_a_row += 1
            criterion = self.router.option_set.options.get(decision.tool, "")
            record["action"] = "guide"
            self.trace.append(record)
            return Guide(
                feedback=f"Call {decision.tool} instead of {proposed}: {criterion}.",
                reason=f"decider prefers {decision.tool} (p={decision.p_top:.2f})",
            )

        self._guides_in_a_row = 0
        record["action"] = "proceed_disagree_low_confidence" if decision.p_top < self.guide_threshold else "proceed_guide_limit"
        logger.info("tool-selection-gate: decider=%s p=%.2f, LLM=%s; letting it through",
                    decision.tool, decision.p_top, proposed)
        self.trace.append(record)
        return Proceed(reason="disagreement below threshold or guide limit reached")
