"""Reusable Decider 2B pattern: Tool Selection (#tool-selection).

Strands Decider 2B answers a typed ``choice`` question ("which tool next?") with
calibrated per-option probabilities. Plain Python applies the thresholds:

    p_top >= t_act   -> run the Decider's tool
    p_top >= t_floor -> ask a cheap LLM router (optional fallback)
    otherwise        -> escape option (e.g. ask_user)

Two wirings:
  * ``DeciderRouter.next_tool(state)`` for a deterministic agent loop.
  * ``ToolSelectionGate`` - a Strands ``InterventionHandler`` that checks the tool an
    LLM-driven ``strands.Agent`` proposes and steers it with ``Guide`` when the
    Decider confidently prefers another tool.

The folder is self-contained: copy it into your project. Only ``gate.py`` imports
``strands``; everything else is standard library.
"""

from .decider_client import DeciderClient, DeciderError, HttpDeciderClient, InProcessDeciderClient
from .question import OptionSet, build_choice_question, build_state
from .router import Decision, DeciderRouter, LLMRouter

__all__ = [
    "DeciderClient",
    "DeciderError",
    "HttpDeciderClient",
    "InProcessDeciderClient",
    "OptionSet",
    "build_choice_question",
    "build_state",
    "Decision",
    "DeciderRouter",
    "LLMRouter",
]

try:  # gate.py needs strands-agents; keep the rest importable without it.
    from .gate import ToolSelectionGate  # noqa: F401

    __all__.append("ToolSelectionGate")
except ImportError:  # pragma: no cover
    pass
