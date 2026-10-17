"""Minimal clients for the Strands Decider System One API.

Wire format (strands-decider 0.1.0, ``strands_decider/schema.py`` and ``server.py``)::

    POST /v1/systemone
    {"state": str | dict | list,
     "questions": {"<key>": {"type": "choice", "instructions": str, "criteria": {opt: desc}}
                         | {"type": "noul",   "instructions": str, "criteria": {"true": .., "false": ..}}},
     "model": "strands-decider-latest"}
    -> {"model": str,
        "answers": {"<key>": {"type": "choice", "choice": str, "probabilities": {opt: p}, "confidence": float}
                           | {"type": "noul", "noul": P(true)}},
        "usage": {"input_tokens": int, "output_tokens": int},
        "latency_ms": float}

    GET /health -> {"status": "ok", "model": ..., "device": ..., ...}

The server has no authentication. Run it on 127.0.0.1 and never expose the port.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol


class DeciderError(RuntimeError):
    """The Decider could not answer (server down, bad request, timeout)."""


class DeciderClient(Protocol):
    """Anything that answers System One requests. Returns the raw response dict."""

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]: ...


class HttpDeciderClient:
    """Standard-library HTTP client for ``strands-decider serve``."""

    def __init__(self, base_url: str = "http://127.0.0.1:8099", timeout_s: float = 30.0,
                 model: str = "strands-decider-latest") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.model = model

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - local URL
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise DeciderError(f"decider HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderError(f"decider unreachable at {self.base_url}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        out = self._request("POST", "/v1/systemone", {"state": state, "questions": questions, "model": self.model})
        out.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 2))
        return out


class InProcessDeciderClient:
    """Loads the checkpoint in this process via ``strands_decider.infer.load_engine``.

    Handy for evals on a GPU box. Needs ``pip install strands-decider`` and enough RAM
    (the 2B torso is upcast to fp32 on CPU, roughly 9 GB).
    """

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        from strands_decider.infer import load_engine  # lazy: heavy import (torch)
        from strands_decider.schema import SystemOneRequest

        self._request_type = SystemOneRequest
        self._engine = load_engine(checkpoint, device=device)

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        request = self._request_type.model_validate({"state": state, "questions": questions})
        try:
            response = self._engine.evaluate(request)
        except ValueError as exc:
            raise DeciderError(str(exc)) from exc
        out = response.model_dump()
        out["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return out
