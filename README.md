# Apply-or-Skip

**A job triage agent where Strands Decider 2B picks every next tool, and Amazon Bedrock Nova only writes text.**
`#strands-decider-2b` `#tool-selection` · Apache-2.0 · built for the AWS *Agent Experience Hackathon: Strands Decider*

Give it a job posting (public Greenhouse / Lever / Ashby URL, any web page, or a saved file) and a short
candidate profile. At each step a 2B decision model answers one typed question, *"which tool next?"*, with
calibrated probabilities. Plain Python applies the threshold. Dealbreakers (location, US work authorization,
security clearance) are caught by the Decider **before** any resume tokens are spent.

> Reusable asset: **[`patterns/tool_selection/`](patterns/tool_selection/README.md)**, the *Reusable Decider 2B pattern: Tool Selection*
> (option set, `DeciderRouter`, `ToolSelectionGate` Strands InterventionHandler). Copy the folder into any Strands project.

## Architecture

```mermaid
flowchart LR
  U[CLI / local web UI] -->|posting + profile.yaml| A[Apply-or-Skip agent loop]
  subgraph EC2["EC2 us-east-1 (CPU, 16 GB, auto-stops when idle)"]
    A -->|state + choice question| D[(strands-decider serve<br/>Decider 2B @127.0.0.1:8099)]
    D -->|tool + probabilities| A
    A --> F[fetch_posting]
    A --> P[parse_posting]
    A --> S[score_fit<br/>5 noul questions, 1 Decider call]
    A --> T[tailor_resume]
    A --> K[skip_job / ask_user]
    S --> D
  end
  P -->|max 400 out tokens| N1[Bedrock Nova Micro]
  T -->|max 900 out tokens| N2[Bedrock Nova Lite]
  A -. p_top in 0.40-0.70 .-> N1
  A --> R[(runs/*.jsonl trace + token ledger)]
```

| Tool | What runs | Bedrock |
|---|---|---|
| `fetch_posting` | HTTP GET (public ATS JSON APIs mapped automatically) or local file | none |
| `parse_posting` | Nova Micro extracts JSON requirements | `amazon.nova-micro-v1:0`, 400 out |
| `score_fit` | Decider answers 5 yes/no (`noul`) questions in one request: remote OK for candidate's country, US auth required, clearance required, must-haves met, seniority match | none |
| `tailor_resume` | Nova Lite writes a resume + cover note from profile facts only | `amazon.nova-lite-v1:0`, 900 out |
| `skip_job` | records the reason | none |
| `ask_user` | escape option: missing or unreadable input | none |

**Policy:** `p_top >= T_ACT` run the Decider's tool · `T_FLOOR <= p_top < T_ACT` ask Nova Micro (20 out tokens) · below that `ask_user`.
Hard rules veto unsafe picks (never tailor with a dealbreaker or before a fit check; no profile -> ask user). Steps already done are masked.

Two wirings ship:
- **Loop (default):** `apply_or_skip/triage.py`. The Decider drives; no LLM routing calls at all.
- **Gate:** `--mode gate`. Nova Lite drives a normal `strands.Agent`; `ToolSelectionGate` (an `InterventionHandler`) checks each proposed call and returns `Guide` when the Decider confidently disagrees.

## Quick start (offline, no AWS)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q                                                    # all tests mock Decider + Bedrock
apply-or-skip triage examples/postings/02-redfield-sre-us-only.md --mock
apply-or-skip batch examples/postings --mock
apply-or-skip serve-ui --mock                                # http://127.0.0.1:8080
python -m evals.run_eval --router both --mock                # plumbing check only
```
`--mock` uses a keyword heuristic shaped like Decider answers and template text. It is **not** Decider 2B
and its numbers mean nothing about model quality; the CLI prints a banner whenever it is on.

## Real run (EC2 + Decider 2B + Bedrock)

1. Set an AWS Budget alert first (the author uses $15/month).
2. Review and run `deploy/launch.sh` (creates an instance role limited by `deploy/iam-policy-bedrock-nova.json`, a security group with **no inbound rules**, and an `r7i.large` with `deploy/user-data.sh`).
3. Connect with SSM Session Manager (no SSH port): `aws ssm start-session --target <instance-id>`.
4. `sudo -iu aos && cd apply-or-skip && . .venv/bin/activate && apply-or-skip health`
5. `apply-or-skip triage https://boards.greenhouse.io/<board>/jobs/<id> --profile my-profile.yaml`
6. `python -m evals.run_eval --router both` -> `evals/results.md`
7. UI via port forwarding: `aws ssm start-session --target <id> --document-name AWS-StartPortForwardingSession --parameters portNumber=8080,localPortNumber=8080`
8. Optionally `INSTANCE_ID=<id> deploy/cloudwatch-idle-alarm.sh`. Stop the instance when done; terminate after the hackathon.

The Decider server has no authentication: it binds `127.0.0.1` and must never be exposed.

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `AOS_PARSE_MODEL_ID` / `AOS_TAILOR_MODEL_ID` / `AOS_ROUTER_MODEL_ID` | `amazon.nova-micro-v1:0` / `amazon.nova-lite-v1:0` / `amazon.nova-micro-v1:0` | models (use `us.` inference-profile IDs if on-demand IDs are refused; update the IAM policy too) |
| `AOS_PARSE_MAX_TOKENS` / `AOS_TAILOR_MAX_TOKENS` / `AOS_ROUTER_MAX_TOKENS` | 400 / 900 / 20 | output caps |
| `MAX_BEDROCK_TOKENS_PER_RUN` | 200000 | hard per-process token budget; the run stops with `budget_stop` |
| `MAX_TAILORED_PER_RUN` | 10 | max resumes per run |
| `DECIDER_BACKEND` | `http` | `http`, `inprocess`, or `mock` |
| `DECIDER_URL` | `http://127.0.0.1:8099` | `strands-decider serve` address |
| `DECIDER_CHECKPOINT` | `StrandsAgents/strands-decider-2B-hobson-v21` | for `inprocess` / user-data |
| `T_ACT` / `T_FLOOR` | 0.70 / 0.40 | confidence policy |
| `DEALBREAKER_THRESHOLD` / `FIT_THRESHOLD` | 0.60 / 0.50 | fit policy |
| `AOS_LLM_FALLBACK` | 1 | set 0 to disable the LLM fallback |
| `MAX_STEPS` | 8 | per job |

## Cost guardrails
- Bedrock: every call reserves budget in a `TokenLedger` first; `max_tokens` on every model; ~$0.0002 per triaged job at list prices (check the Bedrock pricing page).
- IAM: instance role may only `bedrock:InvokeModel*` on the two Nova model ARNs in us-east-1. No access keys on disk.
- EC2: idle auto-stop (systemd timer powers off after 30 idle minutes, instance launched with shutdown-behavior `stop`) plus an optional CloudWatch CPU alarm that stops the instance.
- Expected total for the hackathon: a few dollars of EC2 time; under $1 of Bedrock.

## Eval
CPU smoke test of the real checkpoint (real Decider, templated Bedrock): 0.80 final-tool accuracy, ~1.0 s p50 per decision; see [`docs/decider-smoke-test.md`](docs/decider-smoke-test.md).

`evals/labels.jsonl` labels 40 **synthetic** postings (`examples/postings/`, all fictional) with the correct final tool
(23 skip, 13 tailor, 4 ask_user including a missing file). `evals/run_eval.py` compares the Decider router against an LLM-router baseline
(Nova Micro picks every step) on final-tool accuracy, routing LLM calls/tokens, latency, and Bedrock cost.

## Repo layout
```
patterns/tool_selection/   reusable pattern (stdlib + optional strands)
apply_or_skip/             agent: tools, loop, gate agent, CLI, web UI, Bedrock ledger, offline mocks
evals/                     labels + harness
examples/                  synthetic postings + fictional profile
deploy/                    EC2 user-data, idle stop, IAM policy, launch reference
docs/                      article draft, demo script
tests/                     offline unit tests (mocked Decider + Bedrock)
```

## License
Apache-2.0. See `LICENSE` and `NOTICE`. Uses `strands-agents` and `strands-decider` (Apache-2.0) as dependencies.
