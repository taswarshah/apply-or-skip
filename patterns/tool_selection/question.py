"""The typed question and the compact state the Decider reads.

Lessons baked in (from the strands-decider docs and testing):
  * Every option gets a one-line description. Bare option names are ambiguous.
  * Include an explicit escape option ("ask_user") so the model is never forced to
    pick a wrong tool when none fits.
  * Phrase the question positively. Decision models tend to ignore negation
    ("which tool should NOT run" behaves like "which tool should run").
  * Keep the state small: the Decider's context window is 4,096 tokens and the state
    is shared by every question in one request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Rough chars-per-token for English text; used only for truncation budgets.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class OptionSet:
    """A named set of tools, each with the criterion that makes it the right next step."""

    instructions: str
    options: Mapping[str, str]
    escape_option: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("an OptionSet needs at least 2 options")
        for name, desc in self.options.items():
            if not desc or not desc.strip():
                raise ValueError(f"option {name!r} needs a description")
        if self.escape_option is not None and self.escape_option not in self.options:
            raise ValueError(f"escape option {self.escape_option!r} is not one of the options")

    @property
    def names(self) -> list[str]:
        return list(self.options)

    def subset(self, allowed: list[str] | None) -> OptionSet:
        """Restrict to ``allowed`` (keeps the escape option so there is always a way out)."""
        if not allowed:
            return self
        keep = [n for n in self.options if n in allowed or n == self.escape_option]
        return OptionSet(
            instructions=self.instructions,
            options={n: self.options[n] for n in keep},
            escape_option=self.escape_option,
            meta=self.meta,
        )


def build_choice_question(option_set: OptionSet) -> dict[str, Any]:
    """Render an OptionSet as a ``choice`` question in the /v1/systemone wire format."""
    return {
        "type": "choice",
        "instructions": option_set.instructions,
        "criteria": dict(option_set.options),
    }


def truncate(text: str, max_tokens: int) -> str:
    max_chars = max(0, max_tokens * CHARS_PER_TOKEN)
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 15)].rstrip() + " ...[truncated]"


def build_state(
    goal: str,
    progress: Mapping[str, Any],
    last_result: str = "",
    *,
    last_result_max_tokens: int = 800,
    extra: Mapping[str, str] | None = None,
) -> str:
    """Compact, deterministic state text.

    The same inputs always render the same string, so the Decider server's shared-prefix
    cache stays valid across the questions of one request.
    """
    lines = ["--- GOAL ---", goal.strip()]
    if progress:
        lines.append("--- PROGRESS ---")
    for key, value in progress.items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        lines.append(f"{key}: {value}")
    for title, body in (extra or {}).items():
        lines += [f"--- {title.upper()} ---", body.strip()]
    lines += ["--- LAST TOOL RESULT (truncated) ---", truncate(last_result, last_result_max_tokens) or "(none)"]
    return "\n".join(lines)
