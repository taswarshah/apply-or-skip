# CPU smoke test of the real Decider checkpoint (Oct 17, 2026)

Run on a shared 8-vCPU / 15 GB Linux box, **not on AWS**, with `strands-decider 0.1.0`,
`strands-agents 1.61.1`, torch 2.14.1 CPU, checkpoint `StrandsAgents/strands-decider-2B-hobson-v21`
loaded in-process (`InProcessDeciderClient`, about 8 GB RSS). transformers warned that the
`causal_conv1d` / `flash-linear-attention` kernels were missing and fell back to slower reference code.

## Finding 1: state phrasing decides everything
| State format | fresh | fetched | unreadable | no profile |
|---|---|---|---|---|
| terse flags (`fetched: no`, `parsed: no`, ...) | score_fit 0.56 ✗ | score_fit 0.39 ✗ | fetch_posting 0.44 ✗ | score_fit 0.47 ✗ |
| plain sentences ("The job posting text has NOT been retrieved yet.") | fetch_posting 0.88 ✓ | parse_posting 0.66 ✓ | ask_user 0.82 ✓ | (n/a) dealbreaker → skip_job 0.72 ✓ |

Every wrong answer from the flag format sat below `T_ACT=0.70`, so the policy would have routed it to
the fallback or to `ask_user` instead of acting on it. The agent now renders progress as sentences
(`TriageState.narrative()`).

## Finding 2: 40-posting eval, real Decider, templated Bedrock, no LLM fallback
`AOS_LLM_FALLBACK=0 DECIDER_BACKEND=inprocess python -m evals.run_eval --router decider --mock-bedrock`
→ `evals/results-cpu-smoke.md`

- Final-tool accuracy **0.80** (32/40). Decision latency p50 **1.04 s**, p95 **1.45 s** on CPU.
- 0 routing LLM calls. 1 step fell below the threshold (posting 02, `parse_posting` at p=0.69 → `ask_user`; with the fallback on, Nova Micro would have voted).
- Seven misses came from the fit-check `noul` answers (location/US-auth/clearance/seniority), not from
  next-tool routing. The parse step was **templated** in this run (regex, not Nova Micro), so the
  Decider saw thinner summaries than it will with real Bedrock parsing. Re-measure on EC2 with real Bedrock.

These numbers are from a single small run on a labeled **synthetic** set; treat them as a smoke test, not a benchmark.
