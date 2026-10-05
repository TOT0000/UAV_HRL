# Complete-episode LLM feature search

`run_llm_episode_search.py` is the non-Lipschitz automated search entry point. A
single selected model proposes four candidates in one request. The host validates
each slot independently, locks passing slots, repairs only failed slots, screens
all four on complete recorded episodes, and trains only the highest unrounded
ordering score that is strictly above the baseline score.

This workflow uses method ID `td3_dinkelbach_llm_search`. It is independent from
`td3_dinkelbach_llm`, `run_llm_design.py`, and `run_llm_agent.py`; those empirical
Lipschitz workflows and their artifacts remain available and unchanged. Old
proposer/reviewer run contracts cannot be resumed as episode-search runs.

## Inputs and ranking

Pass every sampling source separately with `--episode-source`. Sources must be
complete `run_llm_sampling.py` outputs. Every episode must contain all movement
steps in order, all current auxiliary rows must be valid, and source contracts
must match. Episode identity is `(source_id, episode_id)`, so trajectories for the
same scenario from different checkpoints remain distinct.

The fixed screening lambda is either `--evaluation-lambda` or the arithmetic mean
of the final 100 committed `dinkelbach_lambda_used` rows in an explicitly supplied
`td3_dinkelbach` run (`--lambda-training-run`). The post-episode lambda is never
substituted. The value, history hash, episode range, and method are persisted.

For every episode, the host computes undiscounted `G_base`, each beta-weighted
feature contribution, and `G_candidate`. EE-tied pairs within the saved tolerance
are excluded. For every other pair, matching reward order scores 1, a reward tie
scores 0.5, and reverse order scores 0. Candidate and baseline use identical
pairs. A baseline tie does not pass; candidate slot order is the stable tie-break
between passing candidates.

## Commands (PowerShell)

Dry-run (builds and saves the actual first prompt and budget; no model request):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --episode-source '<COMPLETE_SAMPLE_SOURCE_1>' `
  --episode-source '<COMPLETE_SAMPLE_SOURCE_2>' `
  --provider lmstudio `
  --base-url 'http://127.0.0.1:1234/v1' `
  --model '<LM_STUDIO_API_IDENTIFIER>' `
  --context-length 50000 `
  --max-output-tokens 16384 `
  --lambda-training-run '<TD3_DINKELBACH_1500_EPISODE_RUN>' `
  --dry-run
```

Formal search (the three evaluation inputs must describe the same 100-scenario,
8-RoI, 1000 x 1000 m comparison):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --episode-source '<COMPLETE_SAMPLE_SOURCE_1>' `
  --episode-source '<COMPLETE_SAMPLE_SOURCE_2>' `
  --provider openai `
  --model '<OPENAI_MODEL_ID>' `
  --context-length 50000 `
  --max-output-tokens 16384 `
  --lambda-training-run '<TD3_DINKELBACH_1500_EPISODE_RUN>' `
  --baseline-run '<COMPATIBLE_TD3_DINKELBACH_RUN>' `
  --baseline-evaluation '<BASELINE_FIXED_8_ROI_EVALUATION>' `
  --evaluation-manifest '<SHARED_100_EPISODE_FIXED_8_ROI_MANIFEST>'
```

Resume:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --resume '<RESULTS_LLM_EPISODE_SEARCH_RUN>'
```

The four complete stage templates are under `prompts/llm_episode_search_*.txt`;
the common background is added to every request. Each call saves its expanded
prompt, request plan, token budget, and raw response. State distinguishes
program-valid, pretraining-selected, training-complete, evaluation-complete, and
historical-best states.

Training retains the formal Dinkelbach update; the fixed screening lambda is not
imposed on training. LLM training runs additionally write
`llm_training_episode_metrics.jsonl` with base/extra/combined reward and
beta-weighted per-feature episode sums. This does not replace canonical history.
