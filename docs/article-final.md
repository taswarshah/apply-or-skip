# Apply or Skip: Letting Strands Decider 2B Pick My Job Agent's Next Tool

*Tags: #strands-decider-2b #tool-selection · Repo: https://github.com/taswarshah/apply-or-skip (Apache-2.0)*

## Why I built this
I'm a cloud engineer in Pakistan. Every day I see dozens of "remote" postings, and most of them quietly
exclude me: US-only, EU right-to-work, security clearance, or on-site three days a week. Keyword filters
miss these. Asking an LLM to read every posting *and* decide what to do next means paying for routing
tokens on jobs I will never apply to.

**Apply-or-Skip** is a small open-source Strands agent. You give it a job URL (Greenhouse, Lever, Ashby, any
web page) or a saved posting, plus a short profile. It fetches the posting, summarizes it, checks fit, and
then either tailors a resume or skips the job with a reason. The twist: **an LLM never decides what happens
next. Strands Decider 2B does.**

## Architecture

```mermaid
flowchart LR
  U[CLI / local web UI] -->|posting + profile.yaml| A[Apply-or-Skip agent loop]
  A -->|state + choice question| D[(strands-decider serve<br/>Decider 2B, localhost)]
  D -->|tool + probabilities| A
  A --> F[fetch_posting]
  A --> P[parse_posting]
  A --> S[score_fit: 5 noul questions, 1 Decider call]
  A --> T[tailor_resume]
  A --> K[skip_job / ask_user]
  P -->|max 400 out tokens| N1[Bedrock Nova Micro]
  T -->|max 900 out tokens| N2[Bedrock Nova Lite]
  A -. p_top 0.40-0.70 .-> N1
```

- **Decider 2B** runs via `strands-decider serve` on a CPU EC2 instance, bound to `127.0.0.1` (no inbound ports; SSM only).
- **Amazon Bedrock Nova Micro** turns raw posting text into compact JSON requirements (max 400 output tokens).
- **Amazon Bedrock Nova Lite** writes the tailored resume and cover note from profile facts only (max 900 output tokens), and only for jobs that pass the fit check.
- Every run writes a JSONL trace and a token ledger to `runs/`.

## How Decider 2B picks tools
At every step the agent sends one `choice` question with six options, each with a one-line criterion:

```python
"fetch_posting": "the posting text has not been retrieved yet",
"parse_posting": "raw posting text exists but has not been summarized into requirements",
"score_fit":     "requirements are known but the candidate's fit has not been checked",
"tailor_resume": "fit was checked, no dealbreaker was found, and the job is worth applying to",
"skip_job":      "a dealbreaker was found or the fit is clearly poor",
"ask_user":      "required input is missing or unreadable and only the user can supply it",
```

The Decider returns calibrated probabilities; plain Python applies the policy:

```
p_top >= 0.70 -> run it      0.40-0.70 -> Nova Micro gets a 20-token vote      < 0.40 -> ask_user
```

Hard rules sit on top (never tailor with a dealbreaker, never tailor before a fit check, no profile -> ask
the user), and steps that already ran are masked out of the option set.

The fit check is the Decider again: five `noul` (yes/no probability) questions in **one request** over the
same state: remote OK from the candidate's country, US work authorization required, clearance required,
must-haves met, seniority match. Dealbreakers are caught before a single resume token is spent.

### The reusable pattern
`patterns/tool_selection/` is a drop-in Decider 2B pattern with two wirings:
1. `DeciderRouter.next_tool(state)` for a deterministic loop (the Apply-or-Skip default).
2. `ToolSelectionGate`, a Strands `InterventionHandler`. Add it to any `Agent(interventions=[gate])`; before
   each tool call it asks the Decider and returns `Guide("call X instead")` only when the Decider confidently
   disagrees (at most twice in a row). If the Decider is down, the agent proceeds.

Adapt it in three steps: write your option set, render a short state, tune the threshold on a labeled set.

## Where Amazon Bedrock fits
Bedrock does only the work that needs words: parsing postings (Nova Micro), writing tailored resumes (Nova
Lite), and casting a 20-token tie-break vote when the Decider is unsure. Each call reserves budget in a
per-run `TokenLedger` first, every model has `max_tokens`, and the EC2 instance role may only call
`bedrock:InvokeModel*` on the two Nova model ARNs. At list prices a triaged job costs about $0.0002 in Bedrock.

## Results
Setup: 40 labeled **synthetic** postings (fictional companies: 23 skip, 13 tailor, 4 ask_user), one
fictional profile, real `strands-decider-2B-hobson-v21` checkpoint on CPU.

| Metric | Decider router |
|---|---|
| Final-tool accuracy | **80% (32/40)** |
| Routing LLM calls | **0** (1 low-confidence step went to `ask_user`; fallback was off) |
| Decision latency on CPU | **~1.0 s p50**, 1.45 s p95 |

**Honest note:** in this run Bedrock was **mocked** (the parse step was templated, not Nova Micro), so the
Decider saw thinner summaries than it will with real parsing. Seven of the eight misses came from the
fit-check `noul` answers, not from next-tool routing. Treat this as a smoke test, not a benchmark. On a GPU
you'd expect decisions closer to ~100 ms.

## What I learned
- **State phrasing matters more than anything.** My first state used terse flags (`fetched: no`). On CPU the
  Decider picked the wrong tool in every test case, all with low confidence. Rewriting progress as plain
  sentences ("The job posting text has NOT been retrieved yet.") made it right in each case. The 0.70
  threshold caught the bad version: none of the wrong answers would have been acted on.
- Describe every option, and always include an escape option (`ask_user`).
- Phrase questions positively, keep the state short, and batch several questions over one state.
- Use `noul` for yes/no fit checks instead of asking a model for a score.
- Keep the policy in code: the model gives probabilities, Python decides.

## Try it
```bash
git clone https://github.com/taswarshah/apply-or-skip && cd apply-or-skip
python3.12 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
apply-or-skip triage examples/postings/02-redfield-sre-us-only.md --mock   # offline
```
The README covers the real EC2 + Decider 2B + Bedrock setup. Next steps: host the Decider on AgentCore, add
more ATS sources, and re-run the eval with real Bedrock parsing.

Repo: https://github.com/taswarshah/apply-or-skip · #strands-decider-2b #tool-selection
