"""All settings come from environment variables (12-factor). Defaults are cheap and safe."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # AWS / Bedrock
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    parse_model_id: str = field(default_factory=lambda: os.environ.get("AOS_PARSE_MODEL_ID", "amazon.nova-micro-v1:0"))
    tailor_model_id: str = field(default_factory=lambda: os.environ.get("AOS_TAILOR_MODEL_ID", "amazon.nova-lite-v1:0"))
    router_model_id: str = field(default_factory=lambda: os.environ.get("AOS_ROUTER_MODEL_ID", "amazon.nova-micro-v1:0"))
    agent_model_id: str = field(default_factory=lambda: os.environ.get("AOS_AGENT_MODEL_ID", "amazon.nova-lite-v1:0"))
    agent_max_tokens: int = field(default_factory=lambda: _i("AOS_AGENT_MAX_TOKENS", 300))
    parse_max_tokens: int = field(default_factory=lambda: _i("AOS_PARSE_MAX_TOKENS", 400))
    tailor_max_tokens: int = field(default_factory=lambda: _i("AOS_TAILOR_MAX_TOKENS", 900))
    router_max_tokens: int = field(default_factory=lambda: _i("AOS_ROUTER_MAX_TOKENS", 20))
    # Hard spend caps (per process run)
    max_bedrock_tokens_per_run: int = field(default_factory=lambda: _i("MAX_BEDROCK_TOKENS_PER_RUN", 200_000))
    max_tailored_per_run: int = field(default_factory=lambda: _i("MAX_TAILORED_PER_RUN", 10))
    max_posting_chars: int = field(default_factory=lambda: _i("AOS_MAX_POSTING_CHARS", 12_000))
    # Prices per 1M tokens (USD) for the cost report only. Check aws.amazon.com/bedrock/pricing.
    price_in: dict = field(default_factory=lambda: {
        "nova-micro": _f("AOS_PRICE_MICRO_IN", 0.035), "nova-lite": _f("AOS_PRICE_LITE_IN", 0.06)})
    price_out: dict = field(default_factory=lambda: {
        "nova-micro": _f("AOS_PRICE_MICRO_OUT", 0.14), "nova-lite": _f("AOS_PRICE_LITE_OUT", 0.24)})
    # Decider
    decider_backend: str = field(default_factory=lambda: os.environ.get("DECIDER_BACKEND", "http"))  # http|inprocess|mock
    decider_url: str = field(default_factory=lambda: os.environ.get("DECIDER_URL", "http://127.0.0.1:8099"))
    decider_checkpoint: str = field(default_factory=lambda: os.environ.get(
        "DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v21"))
    decider_device: str = field(default_factory=lambda: os.environ.get("DECIDER_DEVICE", "cpu"))
    decider_timeout_s: float = field(default_factory=lambda: _f("DECIDER_TIMEOUT_S", 60.0))
    # Policy
    t_act: float = field(default_factory=lambda: _f("T_ACT", 0.70))
    t_floor: float = field(default_factory=lambda: _f("T_FLOOR", 0.40))
    dealbreaker_threshold: float = field(default_factory=lambda: _f("DEALBREAKER_THRESHOLD", 0.60))
    fit_threshold: float = field(default_factory=lambda: _f("FIT_THRESHOLD", 0.50))
    max_steps: int = field(default_factory=lambda: _i("MAX_STEPS", 8))
    llm_fallback: bool = field(default_factory=lambda: os.environ.get("AOS_LLM_FALLBACK", "1") not in ("0", "false", "no"))
    runs_dir: str = field(default_factory=lambda: os.environ.get("AOS_RUNS_DIR", "runs"))
