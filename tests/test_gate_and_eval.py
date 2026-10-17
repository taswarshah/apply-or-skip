from types import SimpleNamespace

from strands.interventions import Guide, Proceed

from patterns.tool_selection import DeciderRouter, ToolSelectionGate
from apply_or_skip.mocks import HeuristicDecider
from apply_or_skip.triage import NEXT_TOOL


def event(name):
    return SimpleNamespace(tool_use={"name": name, "toolUseId": "t1", "input": {}}, agent=SimpleNamespace(messages=[]))


STATE = "--- SO FAR ---\nThe job posting text has NOT been retrieved yet.\n--- LAST TOOL RESULT (truncated) ---\n(none)"


def test_gate_proceeds_when_decider_agrees():
    gate = ToolSelectionGate(DeciderRouter(HeuristicDecider(), NEXT_TOOL), state_fn=lambda e: STATE)
    assert isinstance(gate.before_tool_call(event("fetch_posting")), Proceed)


def test_gate_guides_when_decider_confidently_disagrees():
    gate = ToolSelectionGate(DeciderRouter(HeuristicDecider(), NEXT_TOOL), state_fn=lambda e: STATE)
    out = gate.before_tool_call(event("tailor_resume"))
    assert isinstance(out, Guide) and "fetch_posting" in out.feedback


def test_gate_guide_limit_and_unmanaged_tools():
    gate = ToolSelectionGate(DeciderRouter(HeuristicDecider(), NEXT_TOOL), state_fn=lambda e: STATE,
                             max_guides_per_turn=1)
    assert isinstance(gate.before_tool_call(event("tailor_resume")), Guide)
    assert isinstance(gate.before_tool_call(event("tailor_resume")), Proceed)
    assert isinstance(gate.before_tool_call(event("calculator")), Proceed)


def test_gate_low_confidence_proceeds():
    gate = ToolSelectionGate(DeciderRouter(HeuristicDecider(p_top=0.5), NEXT_TOOL), state_fn=lambda e: STATE)
    assert isinstance(gate.before_tool_call(event("tailor_resume")), Proceed)


def test_gate_registers_on_real_strands_agent():
    from strands import Agent
    from strands.models import BedrockModel

    gate = ToolSelectionGate(DeciderRouter(HeuristicDecider(), NEXT_TOOL))
    Agent(model=BedrockModel(model_id="amazon.nova-lite-v1:0", region_name="us-east-1", max_tokens=50),
          interventions=[gate], callback_handler=None)


def test_eval_harness_mock_end_to_end(tmp_path):
    from evals.run_eval import load_labels, run, write_markdown, PROFILE
    import yaml

    labels = load_labels()
    assert len(labels) == 40 and all(l["synthetic"] for l in labels)
    prof = yaml.safe_load(PROFILE.read_text())
    res = [run("decider", True, prof), run("llm", True, prof)]
    assert res[0]["summary"]["routing_llm_calls"] == 0
    assert res[1]["summary"]["routing_llm_calls"] > 100
    assert res[0]["summary"]["final_tool_accuracy"] >= 0.9
    write_markdown(res, tmp_path / "r.md")
    assert "OFFLINE MOCK RUN" in (tmp_path / "r.md").read_text()
