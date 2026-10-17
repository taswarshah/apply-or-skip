import dataclasses
from types import SimpleNamespace

import pytest

from apply_or_skip.bedrock import BudgetExceeded, StrandsBedrockGenerator, TokenLedger
from apply_or_skip.config import Settings
from apply_or_skip.mocks import HeuristicDecider, MockGenerator
from apply_or_skip.tools import extract_json, fetch_posting, resolve_api_url, TriageState
from apply_or_skip.triage import Deps, NEXT_TOOL, build_router, triage, write_trace


def deps(decider=None, **over):
    s = dataclasses.replace(Settings(), **over)
    ledger = TokenLedger.from_settings(s)
    gen = MockGenerator(ledger)
    dec = decider or HeuristicDecider()
    return Deps(s, dec, gen, build_router(dec, gen, s))


def test_good_fit_is_tailored(profile, postings):
    d = deps()
    st = triage(str(postings / "01-northwind-devops-remote.md"), profile, d)
    assert [s["tool"] for s in st.steps] == ["fetch_posting", "parse_posting", "score_fit", "tailor_resume"]
    assert st.outcome == "tailored"
    assert d.gen.ledger.by_purpose.keys() == {"parse", "tailor"}  # no routing LLM calls


def test_us_only_is_skipped_before_any_tailoring(profile, postings):
    d = deps()
    st = triage(str(postings / "02-redfield-sre-us-only.md"), profile, d)
    assert st.outcome == "skipped" and "requires_us_work_authorization" in st.fit.dealbreakers
    assert "tailor" not in d.gen.ledger.by_purpose


def test_unreadable_asks_user(profile, postings):
    st = triage(str(postings / "12-unreadable-stub.html"), profile, deps())
    assert st.outcome == "needs_user"


def test_missing_profile_asks_user(postings):
    st = triage(str(postings / "01-northwind-devops-remote.md"), None, deps())
    assert st.outcome == "needs_user" and len(st.steps) == 1


def test_rule_blocks_tailor_with_dealbreaker(profile, postings):
    # Decider (forced) always says tailor_resume; hard rules must still prevent it.
    dec = HeuristicDecider(fixed={"next_tool": {"choice": "tailor_resume", "p": 0.95}})
    st = triage(str(postings / "03-ironclad-cloud-clearance.md"), profile, deps(dec))
    assert st.outcome == "skipped" and any(s["source"] == "rule" for s in st.steps)


def test_low_confidence_goes_to_llm_fallback(profile, postings):
    d = deps(HeuristicDecider(p_top=0.5))
    d.gen.route_reply = "fetch_posting"
    st = triage(str(postings / "01-northwind-devops-remote.md"), profile, d)
    assert st.steps[0]["source"] == "llm_fallback" and "route" in d.gen.ledger.by_purpose


def test_token_budget_stops_run(profile, postings):
    st = triage(str(postings / "01-northwind-devops-remote.md"), profile, deps(max_bedrock_tokens_per_run=50))
    assert st.outcome == "budget_stop"


def test_tailor_cap():
    led = TokenLedger(max_tokens=10_000, max_tailored=0)
    with pytest.raises(BudgetExceeded):
        led.reserve(10, 10, "tailor")


def test_trace_written(profile, postings, tmp_path):
    d = deps()
    st = triage(str(postings / "11-saffron-cloud-remote-anywhere.md"), profile, d)
    p = write_trace(st, d, tmp_path)
    assert p.read_text().count("\n") == len(st.steps) + 1


def test_fetch_greenhouse_api_mapping():
    kind, url = resolve_api_url("https://boards.greenhouse.io/acme/jobs/123")
    assert kind == "greenhouse" and url.endswith("/boards/acme/jobs/123")
    st = TriageState(source="https://boards.greenhouse.io/acme/jobs/123", profile={})
    body = '{"title":"SRE","location":{"name":"Remote"},"content":"' + "&lt;p&gt;Kubernetes&lt;/p&gt;" * 40 + '"}'
    out = fetch_posting(st, lambda u: (200, "application/json", body))
    assert out.startswith("FETCHED") and "Kubernetes" in st.raw_text and st.title_hint == "SRE"


def test_fetch_http_error_marks_failed():
    st = TriageState(source="https://example.com/job", profile={})
    fetch_posting(st, lambda u: (404, "text/html", ""))
    assert st.fetch_failed


def test_extract_json():
    assert extract_json('noise {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    assert extract_json("no json") is None


class FakeResult:
    def __init__(self, text):
        self.text = text
        self.metrics = SimpleNamespace(accumulated_usage={"inputTokens": 11, "outputTokens": 5})

    def __str__(self):
        return self.text


def test_bedrock_generator_routes_models_and_caps(monkeypatch):
    import strands

    made = []

    class FakeAgent:
        def __init__(self, model, system_prompt, tools, callback_handler):
            self.model = model

        def __call__(self, prompt):
            return FakeResult('{"title":"x"}')

    monkeypatch.setattr(strands, "Agent", FakeAgent)
    s = Settings()
    led = TokenLedger.from_settings(s)
    g = StrandsBedrockGenerator(s, led, model_factory=lambda mid, mx: made.append((mid, mx)) or object())
    g.generate("parse", "sys", "p")
    g.generate("tailor", "sys", "p")
    g.generate("route", "sys", "p")
    assert made == [("amazon.nova-micro-v1:0", 400), ("amazon.nova-lite-v1:0", 900), ("amazon.nova-micro-v1:0", 20)]
    assert led.calls == 3 and led.cost_usd > 0


def test_real_bedrock_model_constructs_with_caps_without_calling_aws():
    from strands.models import BedrockModel

    m = BedrockModel(model_id="amazon.nova-micro-v1:0", region_name="us-east-1", max_tokens=400)
    cfg = m.get_config()
    assert cfg["model_id"] == "amazon.nova-micro-v1:0" and cfg["max_tokens"] == 400


def test_option_set_matches_tools():
    from apply_or_skip.tools import TOOL_NAMES
    assert tuple(NEXT_TOOL.names) == TOOL_NAMES
