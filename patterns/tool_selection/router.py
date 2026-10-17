"""DeciderRouter: Decider 2B picks the next tool; plain Python applies the policy.

    p_top >= t_act    -> act on the Decider's choice          (source="decider")
    p_top >= t_floor  -> ask the LLM fallback router, if set  (source="llm_fallback")
    otherwise         -> escape option, e.g. ask_user          (source="low_confidence")

Hard rules run after the threshold policy and can veto a choice (source="rule").
"The model classifies, the code decides."
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .decider_client import DeciderClient, DeciderError
from .question import OptionSet, build_choice_question

QUESTION_KEY = "next_tool"

# A hard rule sees (context, proposed_tool) and returns an override tool name, or None.
HardRule = Callable[[Any, str], "str | None"]


@dataclass
class Decision:
    tool: str
    source: str  # decider | llm_fallback | low_confidence | rule | decider_error
    p_top: float = 0.0
    confidence: float = 0.0
    decider_choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    fallback_tool: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMRouter:
    """Cheap LLM router used as the fallback (and as the eval baseline).

    ``generate(prompt) -> str`` is any text model call, e.g. Nova Micro via Bedrock with
    ``max_tokens`` around 20-50. The reply is matched against the option names.
    """

    PROMPT = (
        "You route a tool-using agent. Read the state and reply with exactly one tool name "
        "from the list, nothing else.\n\nTOOLS:\n{options}\n\nSTATE:\n{state}\n\nTool name:"
    )

    def __init__(self, generate: Callable[[str], str]) -> None:
        self.generate = generate

    def choose(self, state: str, option_set: OptionSet) -> str | None:
        options = "\n".join(f"- {n}: {d}" for n, d in option_set.options.items())
        reply = self.generate(self.PROMPT.format(options=options, state=state)) or ""
        return parse_tool_name(reply, option_set.names)


def parse_tool_name(reply: str, names: list[str]) -> str | None:
    """First option name that appears in the reply (exact token match wins)."""
    text = reply.strip().lower()
    for name in names:
        if text == name.lower():
            return name
    hits = [(m.start(), n) for n in names for m in [re.search(rf"\b{re.escape(n.lower())}\b", text)] if m]
    return min(hits)[1] if hits else None


class DeciderRouter:
    def __init__(
        self,
        client: DeciderClient,
        option_set: OptionSet,
        *,
        t_act: float = 0.70,
        t_floor: float = 0.40,
        fallback: LLMRouter | None = None,
        hard_rules: list[HardRule] | None = None,
    ) -> None:
        if not 0.0 <= t_floor <= t_act <= 1.0:
            raise ValueError("need 0 <= t_floor <= t_act <= 1")
        self.client = client
        self.option_set = option_set
        self.t_act = t_act
        self.t_floor = t_floor
        self.fallback = fallback
        self.hard_rules = list(hard_rules or [])

    def ask_decider(self, state: str, allowed: list[str] | None = None) -> Decision:
        """One Decider round trip, no policy applied."""
        options = self.option_set.subset(allowed)
        started = time.perf_counter()
        resp = self.client.ask(state, {QUESTION_KEY: build_choice_question(options)})
        answer = resp["answers"][QUESTION_KEY]
        probs = {k: float(v) for k, v in answer["probabilities"].items()}
        choice = answer.get("choice") or max(probs, key=probs.__getitem__)
        latency = float(resp.get("latency_ms", (time.perf_counter() - started) * 1000))
        return Decision(
            tool=choice, source="decider", p_top=probs.get(choice, 0.0),
            confidence=float(answer.get("confidence", 0.0)), decider_choice=choice,
            probabilities=probs, latency_ms=latency,
        )

    def next_tool(self, state: str, *, context: Any = None, allowed: list[str] | None = None) -> Decision:
        escape = self.option_set.escape_option
        try:
            decision = self.ask_decider(state, allowed)
        except (DeciderError, KeyError, TypeError, ValueError) as exc:
            decision = Decision(tool=escape or self.option_set.names[0], source="decider_error", note=str(exc)[:300])
            if self.fallback is not None:
                picked = self.fallback.choose(state, self.option_set.subset(allowed))
                if picked:
                    decision.tool, decision.source, decision.fallback_tool = picked, "llm_fallback", picked
            return self._apply_rules(decision, context)

        if decision.p_top >= self.t_act:
            pass
        elif decision.p_top >= self.t_floor and self.fallback is not None:
            picked = self.fallback.choose(state, self.option_set.subset(allowed))
            decision.fallback_tool = picked
            if picked:
                decision.tool, decision.source = picked, "llm_fallback"
            else:
                decision.tool, decision.source = escape or decision.tool, "low_confidence"
                decision.note = "fallback reply did not name a tool"
        elif escape is not None:
            decision.tool, decision.source = escape, "low_confidence"
        return self._apply_rules(decision, context)

    def _apply_rules(self, decision: Decision, context: Any) -> Decision:
        for rule in self.hard_rules:
            override = rule(context, decision.tool)
            if override and override != decision.tool:
                decision.note = (decision.note + f" rule {getattr(rule, '__name__', 'rule')}: "
                                 f"{decision.tool} -> {override}").strip()
                decision.tool, decision.source = override, "rule"
        return decision
