import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from patterns.tool_selection import (DeciderError, DeciderRouter, HttpDeciderClient, LLMRouter, OptionSet,
                                     build_choice_question, build_state)
from patterns.tool_selection.question import truncate
from patterns.tool_selection.router import parse_tool_name

OPTS = OptionSet("Which tool next?", {"a": "do a", "b": "do b", "ask": "nothing fits"}, escape_option="ask")


class FakeDecider:
    def __init__(self, choice="a", p=0.9, fail=False):
        self.choice, self.p, self.fail, self.calls = choice, p, fail, []

    def ask(self, state, questions):
        self.calls.append(questions)
        if self.fail:
            raise DeciderError("down")
        names = list(questions["next_tool"]["criteria"])
        rest = (1 - self.p) / (len(names) - 1)
        probs = {n: (self.p if n == self.choice else rest) for n in names}
        return {"answers": {"next_tool": {"type": "choice", "choice": self.choice, "probabilities": probs,
                                          "confidence": 0.5}}, "latency_ms": 3.0}


def test_optionset_validation():
    with pytest.raises(ValueError):
        OptionSet("q", {"only": "one"})
    with pytest.raises(ValueError):
        OptionSet("q", {"a": "x", "b": " "})
    with pytest.raises(ValueError):
        OptionSet("q", {"a": "x", "b": "y"}, escape_option="z")
    assert OPTS.subset(["b"]).names == ["b", "ask"]


def test_choice_question_wire_format():
    q = build_choice_question(OPTS)
    assert q == {"type": "choice", "instructions": "Which tool next?", "criteria": {"a": "do a", "b": "do b", "ask": "nothing fits"}}


def test_state_is_deterministic_and_truncated():
    s1 = build_state("goal", {"x": True, "y": 2}, "z" * 10000, last_result_max_tokens=10)
    assert s1 == build_state("goal", {"x": True, "y": 2}, "z" * 10000, last_result_max_tokens=10)
    assert "x: yes" in s1 and s1.endswith("...[truncated]") and len(truncate("q" * 999, 10)) <= 40


def test_router_acts_on_confident_decider():
    d = DeciderRouter(FakeDecider("b", 0.9), OPTS).next_tool("s")
    assert (d.tool, d.source) == ("b", "decider")


def test_router_uses_llm_fallback_in_middle_band():
    llm = LLMRouter(lambda prompt: "a")
    d = DeciderRouter(FakeDecider("b", 0.55), OPTS, fallback=llm).next_tool("s")
    assert (d.tool, d.source, d.decider_choice) == ("a", "llm_fallback", "b")


def test_router_escapes_when_very_unsure_or_no_fallback():
    assert DeciderRouter(FakeDecider("b", 0.35), OPTS).next_tool("s").tool == "ask"
    assert DeciderRouter(FakeDecider("b", 0.55), OPTS).next_tool("s").source == "low_confidence"


def test_router_fallback_reply_garbage_escapes():
    d = DeciderRouter(FakeDecider("b", 0.55), OPTS, fallback=LLMRouter(lambda p: "hmm")).next_tool("s")
    assert d.tool == "ask" and "did not name" in d.note


def test_router_decider_error_uses_fallback_or_escape():
    assert DeciderRouter(FakeDecider(fail=True), OPTS).next_tool("s").source == "decider_error"
    d = DeciderRouter(FakeDecider(fail=True), OPTS, fallback=LLMRouter(lambda p: "b")).next_tool("s")
    assert (d.tool, d.source) == ("b", "llm_fallback")


def test_hard_rule_overrides():
    def no_b(ctx, tool):
        return "a" if tool == "b" else None
    d = DeciderRouter(FakeDecider("b", 0.95), OPTS, hard_rules=[no_b]).next_tool("s")
    assert (d.tool, d.source) == ("a", "rule")


def test_allowed_mask_sent_to_decider():
    fd = FakeDecider("b", 0.9)
    DeciderRouter(fd, OPTS).next_tool("s", allowed=["b"])
    assert list(fd.calls[0]["next_tool"]["criteria"]) == ["b", "ask"]


def test_threshold_validation():
    with pytest.raises(ValueError):
        DeciderRouter(FakeDecider(), OPTS, t_act=0.3, t_floor=0.5)


@pytest.mark.parametrize("reply,expected", [("b", "b"), ("I pick `a`.", "a"), ("ASK", "ask"), ("none", None)])
def test_parse_tool_name(reply, expected):
    assert parse_tool_name(reply, ["a", "b", "ask"]) == expected


def test_http_client_against_stub_server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"answers": {"next_tool": {"type": "choice", "choice": "a",
                              "probabilities": {"a": 0.8, "b": 0.1, "ask": 0.1}}}, "latency_ms": 7}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = HttpDeciderClient(f"http://127.0.0.1:{srv.server_port}")
        assert c.health()["status"] == "ok"
        d = DeciderRouter(c, OPTS).next_tool("hello")
        assert (d.tool, d.latency_ms) == ("a", 7.0)
        assert seen["path"] == "/v1/systemone" and seen["body"]["state"] == "hello"
    finally:
        srv.shutdown()
    with pytest.raises(DeciderError):
        HttpDeciderClient("http://127.0.0.1:9", timeout_s=1).ask("s", {})
