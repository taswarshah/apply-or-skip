# Reusable Decider 2B pattern: Tool Selection (`#tool-selection`)

A drop-in pattern for letting **Strands Decider 2B** choose which tool an agent runs next,
with calibrated probabilities, a confidence policy in plain Python, and an optional cheap-LLM fallback.

| File | What it is | Depends on |
|---|---|---|
| `question.py` | `OptionSet` (tool name -> one-line "when to use" criterion), `build_choice_question`, compact `build_state` | stdlib |
| `decider_client.py` | `HttpDeciderClient` for `strands-decider serve` (`POST /v1/systemone`), `InProcessDeciderClient` | stdlib (+ `strands-decider` for in-process) |
| `router.py` | `DeciderRouter` (threshold policy + hard rules), `LLMRouter` fallback, `Decision` | stdlib |
| `gate.py` | `ToolSelectionGate`: a Strands `InterventionHandler` that checks an LLM's proposed tool call and returns `Proceed` or `Guide` | `strands-agents` |

## The question

```python
from patterns.tool_selection import OptionSet
NEXT_TOOL = OptionSet(
    instructions="Which tool should the job triage agent run next?",
    options={
        "fetch_posting": "the posting text has not been retrieved yet",
        "parse_posting": "raw posting text exists but has not been summarized into requirements",
        "score_fit":     "requirements are known but the candidate's fit has not been checked",
        "tailor_resume": "fit was checked, no dealbreaker was found, and the job is worth applying to",
        "skip_job":      "a dealbreaker was found or the fit is clearly poor",
        "ask_user":      "required input is missing or unreadable and only the user can supply it",
    },
    escape_option="ask_user",
)
```

## The policy ("the model classifies, the code decides")

```
p_top >= T_ACT   (0.70) -> run the Decider's tool                  source=decider
p_top >= T_FLOOR (0.40) -> ask the LLM fallback router (if given)  source=llm_fallback
otherwise               -> escape option                           source=low_confidence
then hard rules may veto (e.g. never tailor with a dealbreaker)    source=rule
```

Steps that already ran can be masked out with `allowed=[...]` (the escape option is always kept).

## Wiring A: deterministic loop

```python
router = DeciderRouter(HttpDeciderClient("http://127.0.0.1:8099"), NEXT_TOOL,
                       t_act=0.7, t_floor=0.4, fallback=LLMRouter(my_cheap_llm), hard_rules=[...])
decision = router.next_tool(state_text, context=my_state, allowed=["parse_posting", "skip_job"])
run(decision.tool)
```

## Wiring B: Strands plug-in

```python
from strands import Agent
gate = ToolSelectionGate(router, state_fn=lambda event: my_state_text())
agent = Agent(model=bedrock_model, tools=[...], interventions=[gate])
```

The gate returns `Guide(feedback="Call X instead ...")` only when the Decider confidently disagrees
(and at most `max_guides_per_turn` times in a row), so it cannot trap the agent in a loop.
A Decider outage never breaks the agent: the gate logs it and proceeds.

## Adapting it in 3 steps
1. Write your `OptionSet`: one positive, one-line criterion per tool, plus an escape option.
2. Render a compact, deterministic state (goal, progress flags, truncated last result; keep it well under the 4,096-token window).
3. Pick wiring A or B, tune `T_ACT` on a small labeled set, add hard rules for anything that must never happen.

## Lessons
- Describe every option; bare names are ambiguous.
- Always offer a "none of these" escape option.
- Phrase questions positively; negations are easy for a classifier to ignore.
- Keep state short and stable; one request can carry several questions over the same state.
- Measure on your own labeled set before trusting any threshold (see `evals/`).
