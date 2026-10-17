"""Bedrock text generation (Amazon Nova via Strands ``BedrockModel``) with a hard token cap.

The Decider never generates text; Bedrock never routes (except as the low-confidence
fallback). Every call goes through ``TokenLedger.reserve`` first, so a run can never
spend more than ``MAX_BEDROCK_TOKENS_PER_RUN`` tokens, and ``max_tokens`` is set on
every model.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


class BudgetExceeded(RuntimeError):
    pass


def _price_key(model_id: str) -> str:
    m = model_id.lower()
    return "nova-lite" if "lite" in m else "nova-micro" if "micro" in m else m


@dataclass
class TokenLedger:
    max_tokens: int
    max_tailored: int
    price_in: dict[str, float] = field(default_factory=dict)
    price_out: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    tailored: int = 0
    by_purpose: dict[str, dict[str, int]] = field(default_factory=dict)
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def reserve(self, est_input: int, max_output: int, purpose: str) -> None:
        with self._lock:
            if purpose == "tailor" and self.tailored >= self.max_tailored:
                raise BudgetExceeded(f"MAX_TAILORED_PER_RUN={self.max_tailored} reached")
            if self.total + est_input + max_output > self.max_tokens:
                raise BudgetExceeded(
                    f"MAX_BEDROCK_TOKENS_PER_RUN={self.max_tokens} would be exceeded "
                    f"(used {self.total}, next call up to {est_input + max_output})")

    def record(self, model_id: str, purpose: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if purpose == "tailor":
                self.tailored += 1
            p = self.by_purpose.setdefault(purpose, {"calls": 0, "input": 0, "output": 0})
            p["calls"] += 1
            p["input"] += input_tokens
            p["output"] += output_tokens
            key = _price_key(model_id)
            self.cost_usd += (input_tokens * self.price_in.get(key, 0.0)
                              + output_tokens * self.price_out.get(key, 0.0)) / 1_000_000

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "tailored": self.tailored, "est_cost_usd": round(self.cost_usd, 6), "by_purpose": self.by_purpose}

    @classmethod
    def from_settings(cls, s: Settings) -> TokenLedger:
        return cls(max_tokens=s.max_bedrock_tokens_per_run, max_tailored=s.max_tailored_per_run,
                   price_in=dict(s.price_in), price_out=dict(s.price_out))


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str


class TextGenerator(Protocol):
    ledger: TokenLedger

    def generate(self, purpose: str, system: str, prompt: str) -> GenResult: ...


def estimate_tokens(*texts: str) -> int:
    return sum(len(t or "") for t in texts) // 4 + 16


class StrandsBedrockGenerator:
    """Text generation through a tool-less ``strands.Agent`` backed by ``BedrockModel``.

    ``purpose`` selects the model and output cap: parse -> Nova Micro (400),
    tailor -> Nova Lite (900), route -> Nova Micro (20). ``model_factory`` lets tests
    inject a fake model; by default it builds ``strands.models.BedrockModel``.
    """

    def __init__(self, settings: Settings, ledger: TokenLedger,
                 model_factory: Callable[[str, int], Any] | None = None) -> None:
        self.settings = settings
        self.ledger = ledger
        self._model_factory = model_factory or self._bedrock_model
        self._models: dict[str, Any] = {}

    def _bedrock_model(self, model_id: str, max_tokens: int) -> Any:
        from strands.models import BedrockModel

        return BedrockModel(model_id=model_id, region_name=self.settings.aws_region,
                            max_tokens=max_tokens, temperature=0.2)

    def _route(self, purpose: str) -> tuple[str, int]:
        s = self.settings
        return {
            "parse": (s.parse_model_id, s.parse_max_tokens),
            "tailor": (s.tailor_model_id, s.tailor_max_tokens),
            "route": (s.router_model_id, s.router_max_tokens),
        }[purpose]

    def generate(self, purpose: str, system: str, prompt: str) -> GenResult:
        from strands import Agent

        model_id, max_out = self._route(purpose)
        self.ledger.reserve(estimate_tokens(system, prompt), max_out, purpose)
        key = f"{model_id}:{max_out}"
        if key not in self._models:
            self._models[key] = self._model_factory(model_id, max_out)
        agent = Agent(model=self._models[key], system_prompt=system, tools=[], callback_handler=None)
        result = agent(prompt)
        usage = result.metrics.accumulated_usage
        tin, tout = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
        self.ledger.record(model_id, purpose, tin, tout)
        return GenResult(text=str(result).strip(), input_tokens=tin, output_tokens=tout, model_id=model_id)
