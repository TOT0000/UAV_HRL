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

Runs paused in generation/repair at the recorded `1460eff` numeric-operation
contract can be migrated explicitly after the validator update. This rechecks
the latest saved version in all four slots on the original complete-episode
dataset and then stops; it performs no model call, pre-evaluation, or training:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --resume '<RESULTS_LLM_EPISODE_SEARCH_RUN>' `
  --revalidate-only
```

The command preserves candidate version numbers, model/repair counts, and prior
reports. It writes a separate `revalidations/revalidation_NN` record containing
the old/new Git revisions and operation-rule versions. A normal `--resume`
continues only after that bounded transition has been recorded. Incompatible
dataset, interface, baseline, evaluation, or already-trained state is rejected.

Search-selected training runs use the short, collision-resistant repository
root `results/t/<search-and-round-hash>/`; the search state records this root
and the actual training run directory. Before training starts, the runner checks
the run artifact plus the final and atomic-temporary `models` and `full`
checkpoint artifact paths against the Windows legacy path limit. A run stopped
at the `ef10fa3` first-round training initialization failure can perform the one
bounded compatibility migration and continue without regenerating candidates
or rerunning pretraining ordering:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --resume '<RESULTS_LLM_EPISODE_SEARCH_RUN>' `
  --recover-training-initialization
```

The migration verifies the saved dataset, constants, baseline contract,
selected candidate, pretraining report, and approved-artifact identity. The old
nested `PREPARING` directory remains as an initialization-failure record. A
directory is resumed only when it contains training progress and a full-resume
checkpoint; progress without a checkpoint is reported and never silently
retrained.

The `fa948f1` run that completed episode 50 but failed before producing any
checkpoint cannot resume from episode 50. Its one bounded recovery explicitly
preserves that failed run and starts the already selected candidate again at
episode 1 under the shorter root, without model calls or pretraining reranking:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --resume '<RESULTS_LLM_EPISODE_SEARCH_RUN>' `
  --restart-failed-training
```

Ordinary `--resume` still refuses to discard progress without a valid full
checkpoint. If a compatible replacement run has a full checkpoint, it is
resumed even when the older search record still names an abandoned shell. The
explicit restart authorization is persisted and bound to its search run, round,
candidate artifact, and failed training run. If execution stops after recording
that authorization, ordinary `--resume` finishes that one operation; it never
applies the authorization to candidates in later search rounds. Once a
replacement run exists, subsequent resumes follow only that run and will not
discard replacement progress that lacks a full checkpoint.
Older initialization-only directories under prior or legacy output roots remain
in the recovery diagnostics but are not mistaken for the authorized replacement.
If the authorized failed run was already marked abandoned while launch was still
pending, ordinary resume may complete that pending launch exactly once.

Omitting the two limits on resume keeps their saved values. To add budget
without resetting the already-used rounds or repairs, pass larger totals:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_episode_search.py `
  --resume '<RESULTS_LLM_EPISODE_SEARCH_RUN>' `
  --max-search-rounds 8 `
  --max-repairs-per-round 10
```

Saved limits cannot be reduced. Extending a completed search starts the next
round from the saved pre-evaluation or training/evaluation feedback and does not
repeat an earlier round.

The four complete stage templates are under `prompts/llm_episode_search_*.txt`;
the common background is added to every request. Each call saves its expanded
prompt, request plan, token budget, and raw response. State distinguishes
program-valid, pretraining-selected, training-complete, evaluation-complete, and
historical-best states.

Training retains the formal Dinkelbach update; the fixed screening lambda is not
imposed on training. LLM training runs additionally write
`llm_training_episode_metrics.jsonl` with base/extra/combined reward and
beta-weighted per-feature episode sums after every completed episode. Resume
reconciles this file to the selected checkpoint before continuing, so rows after
that checkpoint are re-created exactly once. A legacy interrupted run that never
saved its earlier episode rows cannot fabricate them: complete 100-episode block
summaries stop with an explicit missing-range error. This does not replace
canonical history.

Before a formal model request, the search binds the explicit baseline run,
episode-1500 checkpoint provenance, 100-row fixed-8-RoI evaluation, and shared
manifest/scenario order. Dry-run remains prompt-only and records that this
preflight was not performed.
