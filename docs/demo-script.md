# Demo video script (target 2:45, hard limit < 3:00)

DRAFT. Not recorded or published. Record at 1080p (OBS), captions on, upload to YouTube as **Public**.
Numbers in [brackets] must be filled from the real EC2 run (`evals/results.md`), never from the mock run.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00-0:15 | Title card "Apply-or-Skip", then terminal | "I'm Taswar, a cloud engineer in Pakistan. Most job postings aren't open to me, and paying an LLM to read and route every one is slow and wasteful. This is Apply-or-Skip." |
| 0:15-0:40 | README mermaid diagram | "It's a Strands agent on EC2. Strands Decider 2B, a two-billion-parameter decision model that never writes text, chooses every next tool. Amazon Bedrock Nova Micro and Nova Lite only write the parts that need words." |
| 0:40-1:20 | `apply-or-skip triage examples/postings/01-northwind-devops-remote.md`: trace shows fetch -> parse -> score_fit -> tailor_resume with p and ms per step, then the resume | "Each line is one typed choice question with calibrated probabilities. The code, not the model, applies the 0.7 threshold. Only two Bedrock calls for the whole job." |
| 1:20-1:50 | `triage examples/postings/02-redfield-sre-us-only.md`: score_fit shows `requires_us_work_authorization=[0.9x]`, then skip_job | "Five yes-or-no fit questions answered in one Decider request. US citizens only: dealbreaker. Skipped before a single resume token was spent." |
| 1:50-2:10 | Run with `T_ACT=0.95` so a step falls to `llm_fallback` (marked L) | "When the Decider isn't sure, Nova Micro gets a 20-token vote. Hybrid, not a gamble. And hard rules make sure it can never tailor a resume past a dealbreaker." |
| 2:10-2:30 | `evals/results.md` table (Decider vs LLM router) | "On 40 labeled synthetic postings: [N] routing LLM calls versus [M], [x]% accuracy, [y] ms per decision on CPU." |
| 2:30-2:45 | `patterns/tool_selection/` folder, gate.py | "The tool-selection pattern is a drop-in Strands InterventionHandler. Grab it from the repo; links below. #strands-decider-2b, #tool-selection." |

Checklist before recording: instance running, `apply-or-skip health` OK, terminal font 18pt, `runs/` cleared,
real profile not shown (use the fictional `examples/profile.example.yaml`), no AWS account IDs on screen.
