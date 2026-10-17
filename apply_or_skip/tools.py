"""The six tools. Only parse_posting and tailor_resume call Bedrock."""

from __future__ import annotations

import html
import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bedrock import TextGenerator
from .fit_questions import FitResult

TOOL_NAMES = ("fetch_posting", "parse_posting", "score_fit", "tailor_resume", "skip_job", "ask_user")
MIN_READABLE_CHARS = 200
USER_AGENT = "apply-or-skip/0.1 (+https://github.com/taswarshah/apply-or-skip; personal job triage)"

HttpGet = Callable[[str], tuple[int, str, str]]  # url -> (status, content_type, body)


@dataclass
class TriageState:
    source: str
    profile: dict[str, Any] | None
    raw_text: str = ""
    title_hint: str = ""
    fetched: bool = False
    fetch_failed: bool = False
    parsed: dict[str, Any] | None = None
    fit: FitResult | None = None
    outcome: str | None = None  # tailored | skipped | needs_user | max_steps | budget_stop
    output: str = ""
    last_result: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.outcome is not None

    def narrative(self) -> str:
        """Progress as plain sentences. In CPU smoke tests Decider 2B picked the right tool from
        sentences like these, but not from terse `key: yes/no` flags (see docs/decider-smoke-test.md)."""
        if not self.profile:
            return "No candidate profile was given, so the candidate's fit cannot be checked without the user."
        if not self.source:
            return "No job URL or posting was given."
        if self.fetch_failed:
            return ("Fetching was attempted but the posting is unreadable or the fetch failed; the agent "
                    "cannot get it without the user's help.")
        if not self.fetched:
            return "Nothing has been done yet. The job posting text has NOT been retrieved yet."
        if self.parsed is None:
            return "The raw posting text was retrieved but has NOT been summarized into requirements yet."
        if self.fit is None:
            return ("The posting was summarized into requirements, but the candidate's fit has NOT been "
                    "checked yet.")
        if self.fit.dealbreakers:
            return "Fetched, summarized and fit checked. Fit check result: DEALBREAKER - " + ", ".join(
                d.replace("_", " ") for d in self.fit.dealbreakers) + ". The job is not suitable."
        if self.fit.worth_applying:
            return ("Fetched, summarized and fit checked. Fit check result: no dealbreaker, must-have skills "
                    "met, seniority matches. Worth applying.")
        return ("Fetched, summarized and fit checked. Fit check result: no dealbreaker, but the fit is poor "
                f"({self.fit.summary()}). Not worth applying.")

    def progress(self) -> dict[str, Any]:
        return {
            "job_source_given": bool(self.source),
            "candidate_profile_given": bool(self.profile),
            "fetched": self.fetched,
            "fetch_failed_or_unreadable": self.fetch_failed,
            "parsed": self.parsed is not None,
            "fit_checked": self.fit is not None,
            "dealbreaker_found": bool(self.fit and self.fit.dealbreakers),
            "fit_summary": self.fit.summary() if self.fit else "(not checked)",
        }


# ----------------------------------------------------------------- fetch_posting

def default_http_get(url: str, timeout_s: float = 20.0, max_bytes: int = 2_000_000) -> tuple[int, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - public job URLs only
        body = resp.read(max_bytes).decode("utf-8", "replace")
        return resp.status, resp.headers.get("Content-Type", ""), body


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|li|h[1-6]|div|tr)>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


_GREENHOUSE = re.compile(r"https?://(?:job-)?boards\.greenhouse\.io/(?:embed/job_app\?for=)?([\w-]+)/jobs/(\d+)")
_LEVER = re.compile(r"https?://jobs\.lever\.co/([\w.-]+)/([0-9a-f-]{36})")
_ASHBY = re.compile(r"https?://jobs\.ashbyhq\.com/([\w.%-]+)/([0-9a-f-]{36})")


def resolve_api_url(url: str) -> tuple[str, str]:
    """Map a public ATS posting URL to its public JSON API (kind, api_url)."""
    if m := _GREENHOUSE.match(url):
        return "greenhouse", f"https://boards-api.greenhouse.io/v1/boards/{m.group(1)}/jobs/{m.group(2)}"
    if m := _LEVER.match(url):
        return "lever", f"https://api.lever.co/v0/postings/{m.group(1)}/{m.group(2)}"
    if m := _ASHBY.match(url):
        return "ashby", f"https://api.ashbyhq.com/posting-api/job-board/{m.group(1)}?includeCompensation=true#{m.group(2)}"
    return "html", url


def _from_api(kind: str, body: str, api_url: str) -> tuple[str, str]:
    data = json.loads(body)
    if kind == "greenhouse":
        loc = (data.get("location") or {}).get("name", "")
        return data.get("title", ""), f"{data.get('title', '')}\nLocation: {loc}\n\n" + html_to_text(
            html.unescape(data.get("content", "")))
    if kind == "lever":
        cats = data.get("categories") or {}
        lists = "\n".join(f"{x.get('text', '')}:\n{html_to_text(x.get('content', ''))}" for x in data.get("lists", []))
        return data.get("text", ""), (
            f"{data.get('text', '')}\nLocation: {cats.get('location', '')} | Workplace: {data.get('workplaceType', '')}"
            f"\n\n{data.get('descriptionPlain', '')}\n\n{lists}\n\n{data.get('additionalPlain', '')}")
    if kind == "ashby":
        job_id = api_url.split("#", 1)[1]
        for job in data.get("jobs", []):
            if job.get("id") == job_id or job.get("jobUrl", "").endswith(job_id):
                return job.get("title", ""), (f"{job.get('title', '')}\nLocation: {job.get('location', '')} | Remote: "
                                              f"{job.get('isRemote')}\n\n{job.get('descriptionPlain', '')}")
        raise ValueError("job id not found on the Ashby board")
    raise ValueError(kind)


def fetch_posting(state: TriageState, http_get: HttpGet = default_http_get, max_chars: int = 12_000) -> str:
    src = state.source.strip()
    try:
        path = Path(src)
        if not src.startswith(("http://", "https://")) and path.exists():
            text, title = path.read_text(encoding="utf-8"), path.stem
            if path.suffix.lower() in {".html", ".htm"}:
                text = html_to_text(text)
        else:
            kind, api_url = resolve_api_url(src)
            status, ctype, body = http_get(api_url.split("#", 1)[0])
            if status >= 400:
                raise ValueError(f"HTTP {status}")
            if kind == "html":
                title, text = "", html_to_text(body) if "html" in ctype or body.lstrip().startswith("<") else body
            else:
                title, text = _from_api(kind, body, api_url)
    except Exception as exc:  # noqa: BLE001 - surfaced to the Decider as a failed fetch
        state.fetched, state.fetch_failed = True, True
        return f"FETCH FAILED for {src}: {type(exc).__name__}: {str(exc)[:200]}"
    text = text.strip()[:max_chars]
    state.raw_text, state.title_hint, state.fetched = text, title, True
    if len(text) < MIN_READABLE_CHARS:
        state.fetch_failed = True
        return f"FETCHED {len(text)} chars only; the posting looks empty or unreadable:\n{text}"
    state.fetch_failed = False
    return f"FETCHED {len(text)} chars of posting text (not yet summarized):\n{text}"


# ----------------------------------------------------------------- parse_posting

PARSE_SYSTEM = (
    "You extract job requirements. Reply with ONE compact JSON object and nothing else, keys: "
    "title, company, location_policy (one sentence, quote any country/time-zone limits), "
    "remote_regions (list), must_haves (list, <=8 short items), nice_to_haves (list, <=5), "
    "years_required (number or null), work_authorization (string or null), "
    "security_clearance (string or null), salary (string or null). Use null when not stated. Never guess."
)


def extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def parse_posting(state: TriageState, gen: TextGenerator) -> str:
    if not state.raw_text:
        return "PARSE SKIPPED: there is no posting text yet."
    res = gen.generate("parse", PARSE_SYSTEM, state.raw_text[:8000])
    parsed = extract_json(res.text) or {"title": state.title_hint, "unstructured_summary": res.text[:800]}
    state.parsed = parsed
    return "PARSED requirements (fit not checked yet): " + json.dumps(parsed, ensure_ascii=False)


# ----------------------------------------------------------------- tailor_resume

TAILOR_SYSTEM = (
    "You tailor a resume for one job. Use ONLY facts in the candidate profile; never invent employers, "
    "titles, dates, degrees or skills. Output Markdown: a 2-line summary, 5-7 achievement bullets "
    "reordered and reworded for the job's must-haves, a skills line, then a cover note under 90 words."
)


def tailor_resume(state: TriageState, gen: TextGenerator) -> str:
    prompt = ("JOB (parsed):\n" + json.dumps(state.parsed or {}, ensure_ascii=False)
              + "\n\nCANDIDATE PROFILE:\n" + json.dumps(state.profile or {}, ensure_ascii=False))
    res = gen.generate("tailor", TAILOR_SYSTEM, prompt)
    state.outcome, state.output = "tailored", res.text
    return f"TAILORED resume ({res.output_tokens} tokens)."


def skip_job(state: TriageState, reason: str = "") -> str:
    reason = reason or (state.fit.summary() if state.fit else "the Decider judged the job not worth pursuing")
    state.outcome, state.output = "skipped", reason
    return f"SKIPPED: {reason}"


def ask_user(state: TriageState) -> str:
    if not state.profile:
        q = "Please provide a candidate profile (see examples/profile.example.yaml)."
    elif not state.source:
        q = "Please provide a job URL or a saved posting file."
    elif state.fetch_failed:
        q = f"I could not read the posting at {state.source}. Paste the posting text into a file and pass its path."
    else:
        q = "I am not confident about the next step for this posting; please review it manually."
    state.outcome, state.output = "needs_user", q
    return f"ASKED USER: {q}"
