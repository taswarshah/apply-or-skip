# Eval results

Dataset: 40 **labeled synthetic** postings (`examples/postings`, `evals/labels.jsonl`), fictional candidate profile.

> **Real Decider 2B, template Bedrock.** Routing and fit decisions come from the real checkpoint; parse/tailor text is templated, so Bedrock token and cost numbers are not real.

| metric | decider |
|---|---|
| final_tool_accuracy | 0.8 |
| avg_steps | 3.75 |
| routing_llm_calls | 0 |
| routing_llm_tokens | 0 |
| llm_fallback_steps | 0 |
| decision_latency_ms_p50 | 1036.31 |
| decision_latency_ms_p95 | 1448.48 |
| bedrock_calls_total | 53 |
| bedrock_tokens_total | 21639 |
| est_bedrock_cost_usd | 0.001512 |

## Misses

- [decider] 02-redfield-sre-us-only.md: expected skip_job, got ask_user via fetch_posting > ask_user
- [decider] 08-quarry-junior-support.md: expected skip_job, got tailor_resume via fetch_posting > parse_posting > score_fit > tailor_resume
- [decider] 10-cobalt-devops-uk-only.md: expected skip_job, got tailor_resume via fetch_posting > parse_posting > score_fit > tailor_resume
- [decider] 17-ember-cloud-hybrid-nyc.md: expected skip_job, got tailor_resume via fetch_posting > parse_posting > score_fit > tailor_resume
- [decider] 24-lattice-infra-remote-global.md: expected tailor_resume, got skip_job via fetch_posting > parse_posting > score_fit > skip_job
- [decider] 26-nimbus-devops-austin-onsite.md: expected skip_job, got tailor_resume via fetch_posting > parse_posting > score_fit > tailor_resume
- [decider] 29-quill-sre-greencard.md: expected skip_job, got tailor_resume via fetch_posting > parse_posting > score_fit > tailor_resume
- [decider] 34-vesper-sre-ts-sci.md: expected skip_job, got tailor_resume via fetch_posting > parse_posting > score_fit > tailor_resume
