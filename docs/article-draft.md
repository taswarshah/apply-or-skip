# Apply or Skip: Letting a 2B Decision Model Pick My Job Agent's Next Tool

*DRAFT for AWS Builder Center. Not published. Tags: `#strands-decider-2b` `#tool-selection`.*
*Fill every [bracket] from the real EC2 eval before publishing.*

## The problem
I'm a cloud engineer in Pakistan. On a typical day I see dozens of "remote" postings, and most of them
quietly exclude me: US-only, EU right-to-work, security clearance, or on-site three days a week. Keyword
filters miss these, and asking an LLM to read every posting *and* decide what to do next means paying for
routing tokens on jobs I'll never apply to.

## What I built
**Apply-or-Skip** is a small open-source Strands agent. You give it a job URL (or a saved posting) and a
short profile. It fetches the posting, summarizes it, checks fit, and then either tailors a resume or skips
the job with a reason. The twist: **an LLM never decides what happens next**. Strands Decider 2B does.

[demo video embed] · Repo: https://github.com/taswarshah/apply-or-skip

## Architecture
[mermaid diagram from README]

- **Decider 2B** runs with `strands-decider serve` on a CPU EC2 instance, bound to localhost.
- **Amazon Bedrock Nova Micro** turns raw posting text into compact JSON (max 400 output tokens).
- **Amazon Bedrock Nova Lite** writes the tailored resume (max 900 output tokens), only for jobs that pass.

## The decision I handed to Decider 2B: which tool next?
Every step, the agent sends one `choice` question with six options, each with a one-line criterion:

```python
"fetch_posting": "the posting text has not been retrieved yet",
"parse_posting": "raw posting text exists but has not been summarized into requirements",
"score_fit":     "requirements are known but the candidate's fit has not been checked",
"tailor_resume": "fit was checked, no dealbreaker was found, and the job is worth applying to",
"skip_job":      "a dealbreaker was found or the fit is clearly poor",
"ask_user":      "required input is missing or unreadable and only the user can supply it",
```

The Decider returns calibrated probabilities. Plain Python decides what to do with them:

```
p_top >= 0.70 -> run it          0.40-0.70 -> Nova Micro gets a 20-token vote          < 0.40 -> ask_user
```

Hard rules sit on top (never tailor with a dealbreaker; never tailor before a fit check), and steps that
already ran are masked out of the option set.

The fit check is the Decider again: five `noul` (yes/no probability) questions answered in **one request**
over the same state: remote OK from the candidate's country, US work authorization required, clearance
required, must-haves met, seniority match.

## The reusable pattern
`patterns/tool_selection/` is labeled as a reusable Decider 2B pattern. It has two wirings:
1. `DeciderRouter.next_tool(state)` for a deterministic loop (what Apply-or-Skip uses by default).
2. `ToolSelectionGate`, a Strands `InterventionHandler`. Drop it into any `Agent(interventions=[gate])`;
   before each tool call it asks the Decider, and returns `Guide("call X instead")` only when the Decider
   confidently disagrees, at most twice in a row. If the Decider is down, it proceeds.

Adapt it in three steps: write your option set, render a short state, tune the threshold on a labeled set.

## Results
40 labeled synthetic postings (fictional companies), one fictional profile.

| | Decider router | LLM router (Nova Micro) |
|---|---|---|
| Final-tool accuracy | [ ] | [ ] |
| Routing LLM calls | [0 + fallbacks] | [ ] |
| Decision latency p50 (CPU r7i.large) | [ ] ms | [ ] ms |
| Bedrock cost for 40 jobs | $[ ] | $[ ] |

Honest note: on CPU each decision takes seconds, not the ~100 ms you get on a GPU.

## What I learned
- **How you phrase the state matters more than anything.** My first state used terse flags
  (`fetched: no`, `parsed: no`). In a CPU smoke test the Decider picked the wrong tool for every case, with
  low confidence. Rewriting progress as plain sentences ("The job posting text has NOT been retrieved yet")
  made it pick the right tool in each case I tried. The confidence threshold caught the bad version: none of
  those wrong answers would have cleared 0.70.
- Describe every option, and always include an escape option (`ask_user`).
- Phrase questions positively.
- Keep the state short; one request can carry several questions over the same state.
- Use `noul` for yes/no fit checks instead of asking for a score.

## Cost and guardrails
AWS Budgets alert, a per-run Bedrock token ledger with hard caps, `max_tokens` on every model, an instance
role that can only invoke the two Nova models, no inbound ports (SSM only), and an idle auto-stop.
Total spend for the build: $[ ].

## Try it
`pip install -e '.[dev]' && apply-or-skip triage examples/postings/02-redfield-sre-us-only.md --mock` runs
offline. Next steps: host the Decider on AgentCore, add more ATS sources, and contribute the pattern.
