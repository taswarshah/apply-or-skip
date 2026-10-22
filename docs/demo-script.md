# Demo video shot list (target 2:40, hard limit < 3:00)

Record 1080p, terminal font 18pt, captions on, upload to YouTube as **Public**. Use only the fictional
`examples/profile.example.yaml`; no AWS account IDs on screen. If Bedrock is still unavailable, say
"Bedrock calls are mocked in this recording" on screen.

| # | Time | On screen | Voice-over |
|---|---|---|---|
| 1 | 0:00-0:15 | Title card, then terminal | "I'm Taswar, a cloud engineer in Pakistan. Most 'remote' jobs aren't open to me. Apply-or-Skip triages them without paying an LLM to route." |
| 2 | 0:15-0:35 | README architecture diagram | "Strands Decider 2B picks every next tool. Bedrock Nova Micro and Nova Lite only write text." |
| 3 | 0:35-1:10 | `apply-or-skip triage examples/postings/01-northwind-devops-remote.md` | "Each step is one choice question with probabilities. Code applies the 0.7 threshold. Fetch, parse, fit, tailor: two Bedrock calls total." |
| 4 | 1:10-1:40 | `triage examples/postings/02-redfield-sre-us-only.md` (score_fit, skip_job) | "Five yes/no fit questions in one Decider call. US-only: skipped before any resume tokens." |
| 5 | 1:40-2:00 | Same run with `T_ACT=0.95` (step marked L) | "When the Decider is unsure, Nova Micro gets a 20-token vote. Hard rules block tailoring past a dealbreaker." |
| 6 | 2:00-2:20 | `docs/decider-smoke-test.md` results | "On 40 synthetic postings: 80% final-tool accuracy, zero routing LLM calls, about one second per decision on CPU." |
| 7 | 2:20-2:40 | `patterns/tool_selection/gate.py`, repo URL | "The pattern is a drop-in Strands InterventionHandler. Repo below. #strands-decider-2b #tool-selection." |

Before recording: `apply-or-skip health` OK, `runs/` cleared.
