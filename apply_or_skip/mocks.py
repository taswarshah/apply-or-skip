"""OFFLINE STAND-INS. NOT A MODEL. Used by unit tests and the no-AWS demo mode.

``HeuristicDecider`` imitates the *wire format* of ``strands-decider serve`` with
keyword rules so the pipeline, UI and eval harness can run without the 2B checkpoint.
Its numbers say nothing about Strands Decider 2B quality. ``MockGenerator`` imitates
Bedrock with templates. Real results must come from DECIDER_BACKEND=http|inprocess.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .bedrock import GenResult, TokenLedger, estimate_tokens


def _soft(choice: str, names: list[str], p: float) -> dict[str, float]:
    rest = (1 - p) / max(1, len(names) - 1)
    return {n: round(p if n == choice else rest, 4) for n in names}


class HeuristicDecider:
    """Keyword rules shaped like Decider answers. ``fixed`` forces answers (tests)."""

    def __init__(self, p_top: float = 0.9, fixed: dict[str, Any] | None = None) -> None:
        self.p_top = p_top
        self.fixed = fixed or {}
        self.requests: list[dict[str, Any]] = []

    def _next_tool(self, state: str, names: list[str]) -> str:
        so_far = state.split("--- SO FAR ---", 1)[-1].split("--- LAST TOOL RESULT", 1)[0].lower()
        if "no candidate profile" in so_far or "without the user" in so_far or "no job url" in so_far:
            pick = "ask_user"
        elif "dealbreaker -" in so_far or "not worth applying" in so_far:
            pick = "skip_job"
        elif "worth applying" in so_far:
            pick = "tailor_resume"
        elif "fit has not been checked" in so_far:
            pick = "score_fit"
        elif "not been summarized" in so_far:
            pick = "parse_posting"
        else:
            pick = "fetch_posting"
        return pick if pick in names else names[-1]

    def _noul(self, key: str, state: str) -> float:
        s = state.split("--- CANDIDATE PROFILE ---", 1)[0].lower()  # job section only
        if key == "remote_ok_for_candidate":
            limited = re.search(r"(us|u\.s\.|united states|eu|uk|canada|emea|est|pst|cet)[- ]only|must (be )?(reside|live|be located)|"
                                r"on-?site|hybrid|relocat|within the (us|united states)|located in the us", s)
            worldwide = re.search(r"remote[^.\n]{0,40}(worldwide|anywhere|any country|global)|pakistan|apac|asia", s)
            return 0.9 if worldwide and not limited else 0.1 if limited else 0.6
        if key == "requires_us_work_authorization":
            return 0.92 if re.search(r"us citizen|u\.s\. citizen|green card|authori[sz]ed to work in the (us|united states)|"
                                     r"no (visa )?sponsorship|us work authorization required", s) else 0.05
        if key == "requires_security_clearance":
            return 0.9 if re.search(r'"security_clearance": "required"|clearance (is )?required|ts/sci|secret clearance', s) else 0.04
        if key == "must_haves_met":
            m = re.search(r'"must_haves": \[(.*?)\]', state, flags=re.S)
            prof = state.split("--- CANDIDATE PROFILE ---", 1)[-1].lower()
            if not m:
                return 0.5
            items = [i.strip(' "').lower() for i in m.group(1).split(",") if i.strip()]
            hit = sum(1 for i in items if any(w in prof for w in re.findall(r"[a-z0-9+#]{3,}", i)))
            return 0.85 if items and hit / len(items) >= 0.6 else 0.2
        if key == "seniority_matches":
            req = re.search(r'"years_required": (\d+)', state)
            have = re.search(r'"years_experience": (\d+)', state)
            if req and have:
                return 0.85 if abs(int(req.group(1)) - int(have.group(1))) <= 2 else 0.15
            return 0.7
        return 0.5

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.requests.append({"state": state, "questions": questions})
        answers: dict[str, Any] = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                names = list(q["criteria"])
                pick = self.fixed.get(key, {}).get("choice") or self._next_tool(str(state), names)
                p = self.fixed.get(key, {}).get("p", self.p_top)
                probs = _soft(pick, names, p)
                n = len(names)
                answers[key] = {"type": "choice", "choice": pick, "probabilities": probs,
                                "confidence": round(max(0.0, (n * p - 1) / (n - 1)), 4)}
            else:
                answers[key] = {"type": "noul", "noul": float(self.fixed.get(key, self._noul(key, str(state))))}
        return {"model": "MOCK-heuristic-not-a-model", "answers": answers,
                "usage": {"input_tokens": len(str(state)) // 4, "output_tokens": 0}, "latency_ms": 0.1}


class MockGenerator:
    """Template 'LLM' that records token usage in the real ledger."""

    def __init__(self, ledger: TokenLedger, route_reply: str | None = None) -> None:
        self.ledger = ledger
        self.route_reply = route_reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, purpose: str, system: str, prompt: str) -> GenResult:
        out_cap = {"parse": 400, "tailor": 900, "route": 20}[purpose]
        tin = estimate_tokens(system, prompt)
        self.ledger.reserve(tin, out_cap, purpose)
        self.calls.append((purpose, prompt[:200]))
        if purpose == "parse":
            text = json.dumps(_mock_parse(prompt))
        elif purpose == "tailor":
            text = "## Summary\nMOCK tailored resume (offline mode, no Bedrock call).\n\n- bullet\n\nCover note: MOCK."
        else:
            text = self.route_reply or "ask_user"
        tout = min(out_cap, estimate_tokens(text))
        self.ledger.record("mock-nova-micro" if purpose != "tailor" else "mock-nova-lite", purpose, tin, tout)
        return GenResult(text=text, input_tokens=tin, output_tokens=tout, model_id="mock")


def _mock_parse(text: str) -> dict[str, Any]:
    def section(name: str) -> list[str]:
        items: list[str] = []
        inside = False
        for ln in text.splitlines():
            if re.match(r"^#+\s", ln):
                inside = bool(re.search(name, ln, flags=re.I))
                continue
            item = ln.strip().lstrip("-*").strip()
            if inside and item:
                items.append(item)
        return items[:8]

    title = re.search(r"(?m)^#\s+(.+)$", text)
    years = re.search(r"(\d+)\+?\s+years", text)
    loc = re.search(r"(?im)^\**location\**:?\s*(.+)$", text)
    return {
        "title": title.group(1).strip() if title else None,
        "location_policy": loc.group(1).strip() if loc else None,
        "must_haves": section("(must[- ]haves?|requirements|what you bring)"),
        "nice_to_haves": section("nice"),
        "years_required": int(years.group(1)) if years else None,
        "work_authorization": "US work authorization required" if re.search(r"(?i)us citizen|authori[sz]ed to work in the u", text) else None,
        "security_clearance": "required" if re.search(r"(?i)clearance", text) else None,
    }
