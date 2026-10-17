"""score_fit: the Decider answers 5 yes/no ("noul") questions in ONE request.

The state (parsed posting + candidate profile) is encoded once and shared by every
question. ``score`` questions are avoided on purpose: the strands-decider eval notes
say the score primitive is the weakest; nouls are calibrated P(true).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from patterns.tool_selection.decider_client import DeciderClient
from patterns.tool_selection.question import truncate


def fit_questions(location: str) -> dict[str, dict[str, Any]]:
    return {
        "remote_ok_for_candidate": {
            "type": "noul",
            "instructions": f"The job can be done by someone living in {location}, either fully remote "
                            f"worldwide or remote in a region that includes {location}.",
            "criteria": {
                "true": f"the posting allows remote work from anywhere, or explicitly from {location} or its region",
                "false": "the posting requires on-site/hybrid work or limits remote work to countries or time zones "
                         f"that exclude {location}",
            },
        },
        "requires_us_work_authorization": {
            "type": "noul",
            "instructions": "The posting requires US citizenship, a green card, or existing authorization to work in the US.",
            "criteria": {"true": "US citizenship / US work authorization is required",
                         "false": "no US work authorization requirement is stated"},
        },
        "requires_security_clearance": {
            "type": "noul",
            "instructions": "The posting requires a government security clearance.",
            "criteria": {"true": "a security clearance is required or must be obtainable",
                         "false": "no security clearance is mentioned"},
        },
        "must_haves_met": {
            "type": "noul",
            "instructions": "The candidate's skills cover most of the posting's must-have requirements.",
            "criteria": {"true": "the profile covers most must-have skills",
                         "false": "several must-have skills are missing from the profile"},
        },
        "seniority_matches": {
            "type": "noul",
            "instructions": "The role's seniority level fits the candidate's years of experience.",
            "criteria": {"true": "the required experience is within about two years of the candidate's",
                         "false": "the role is far more senior or far more junior than the candidate"},
        },
    }


# For each question: which answer is a dealbreaker (True = P(true) high is bad).
DEALBREAKERS = {
    "remote_ok_for_candidate": False,
    "requires_us_work_authorization": True,
    "requires_security_clearance": True,
}
FIT_SIGNALS = ("must_haves_met", "seniority_matches")


@dataclass
class FitResult:
    probabilities: dict[str, float]
    dealbreakers: list[str] = field(default_factory=list)
    fit_signals_met: int = 0
    worth_applying: bool = False
    latency_ms: float = 0.0

    def summary(self) -> str:
        if self.dealbreakers:
            return "dealbreaker: " + ", ".join(self.dealbreakers)
        return (f"{self.fit_signals_met}/{len(FIT_SIGNALS)} fit signals met; no dealbreaker; "
                f"{'worth applying' if self.worth_applying else 'weak fit'}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_state(parsed: dict[str, Any], profile: dict[str, Any]) -> str:
    prof = {k: profile.get(k) for k in ("headline", "location", "years_experience", "skills", "work_authorization")}
    return ("--- JOB POSTING (summary) ---\n" + truncate(json.dumps(parsed, ensure_ascii=False), 900)
            + "\n--- CANDIDATE PROFILE ---\n" + truncate(json.dumps(prof, ensure_ascii=False), 400))


def evaluate_fit(client: DeciderClient, parsed: dict[str, Any], profile: dict[str, Any], *,
                 dealbreaker_threshold: float = 0.6, fit_threshold: float = 0.5) -> FitResult:
    resp = client.ask(fit_state(parsed, profile), fit_questions(profile.get("location", "Pakistan")))
    probs = {k: float(v["noul"]) for k, v in resp["answers"].items()}
    breakers = []
    for key, bad_when_true in DEALBREAKERS.items():
        p = probs.get(key, 0.5)
        if (bad_when_true and p >= dealbreaker_threshold) or (not bad_when_true and 1 - p >= dealbreaker_threshold):
            breakers.append(key)
    met = sum(1 for k in FIT_SIGNALS if probs.get(k, 0.0) >= fit_threshold)
    return FitResult(probabilities=probs, dealbreakers=breakers, fit_signals_met=met,
                     worth_applying=not breakers and met == len(FIT_SIGNALS),
                     latency_ms=float(resp.get("latency_ms", 0.0)))
