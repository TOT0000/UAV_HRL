# Multi-candidate trained-feedback LLM reward search

`run_llm_training_search.py` implements the independent
`td3_dinkelbach_llm_train_search` comparison. It does not replace TD3, DDPG,
task-allocation, routing, either Lipschitz flow, or the complete-episode
ordering search.

Each round makes four separate model requests. Round 1 supplies only short,
explicitly untrained summaries of earlier designs in that round. Later rounds
freeze the historical best candidate at the start of the round and request,
in order, add, remove, reweight, and redesign variants of that same candidate.
All four executable, nonduplicate candidates are locked before any candidate
is trained. There is no Lipschitz, ordering-consistency, reviewer-model, or EE
improvement gate.

The fixed search contract is four candidates per round, 800 training episodes,
training seed 20260817, episode-800 checkpoints, and 100 shared evaluation
episodes at 8 RoIs, 1000 x 1000 m, and 60 seconds. Candidate ranking is the
arithmetic mean of the 100 per-episode EE values. The baseline evaluation must
be explicitly named and pass the same checkpoint, manifest hash and order,
scenario, environment, and completed-output checks before model work begins.

## Prompts and source context

The stage templates are:

- `prompts/llm_training_search_common.txt`
- `prompts/llm_training_search_initial.txt`
- `prompts/llm_training_search_feedback.txt`
- `prompts/llm_training_search_direction_{add,remove,reweight,redesign}.txt`
- `prompts/llm_training_search_{validation,duplicate}_repair.txt`

The host extracts the observation/validity, movement/energy, visual sensing,
communication/service, and delivery/base-reward excerpts from the current
source tree. Every excerpt records file, symbol, line range, SHA-256, call
relationships, and the current Git revision in
`prompt_source_context/environment_source_context.json`. Candidates still see
only the documented `obs` and `constants` interface.

Each model call writes the fully expanded prompt, request plan, token budget,
raw provider stream, and response under the saved call directory. Repairs reuse
COMMON plus the original stage instruction, failed candidate, and latest check
report; they do not recursively nest older repair prompts.

## PowerShell commands

Dry-run builds the real first prompt and source bundle without baseline, model
request, training, or evaluation work (provider/model values are still used to
construct and budget the request):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_training_search.py `
  --validation-source '.\results\llm_samples\diverse-1800-20261005-124933-41db181e\ep100_noise0.1\td3d-ep100-20261005T070345Z-c96408f5' `
  --provider lmstudio `
  --base-url 'http://127.0.0.1:1234/v1' `
  --model '<LM_STUDIO_API_MODEL_ID>' `
  --context-length 131072 `
  --max-output-tokens 16384 `
  --dry-run
```

Formal search (run only after the named baseline directory has a complete,
validated `paper_evaluation_metadata.json` and 100 per-episode rows):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_training_search.py `
  --validation-source '.\results\llm_samples\diverse-1800-20261005-124933-41db181e\ep100_noise0.1\td3d-ep100-20261005T070345Z-c96408f5' `
  --baseline-run '.\results\td3_dinkelbach\run-seed20260817-359edbc-20260922T132509272242Z-8601a052' `
  --baseline-evaluation 'C:\Users\user\Desktop\uav_eval\base800-20261010-121604' `
  --provider openai `
  --model '<OPENAI_MODEL_ID>' `
  --context-length 131072 `
  --max-output-tokens 16384 `
  --max-rounds 5 `
  --max-repairs-per-candidate 5 `
  --output-root '.\results\llm_training_searches'
```

API keys are read only by the existing provider client from its environment
variable. Model identifiers are passed to the provider; there is no LM Studio
model whitelist.

Resume without repeating completed generation, training, or evaluation:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_training_search.py `
  --resume '<SEARCH_OUTPUT_DIRECTORY>'
```

Increase total rounds or a previously exhausted correction budget:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_llm_training_search.py `
  --resume '<SEARCH_OUTPUT_DIRECTORY>' `
  --max-rounds 8 `
  --max-repairs-per-candidate 10
```

Continue one chosen candidate from its episode-800 full checkpoint to episode
1500 without involving the search controller:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' .\run_experiment.py resume `
  '<CANDIDATE_TRAINING_RUN_DIRECTORY>' `
  --target-episodes 1500
```

The saved search state names each candidate's actual training run directory.
Deep candidate-version, artifact, model-call, checkpoint, and evaluation work
uses stable hashed paths below `--runtime-root` (default
`C:\Users\user\uav_hrl_rt`) to retain Windows path headroom. Search state and
provenance retain every mapping.

## Persistence and resume

The v1 search schema is intentionally incompatible with old search directories.
The state records the model configuration, beta, fixed training/evaluation
contract, baseline preflight and manifest hashes, source bundle hash, candidate
versions and fingerprints, requested/actual direction, invocation settings,
training recovery record, evaluation attempts, score table, and historical best.
Only total rounds and the per-candidate correction ceiling may be increased.

Training uses a distinct output root for every candidate. The existing full
checkpoint contains actor, critics, targets, optimizers, replay, RNG, routing,
lambda, and lifecycle state. A resumable interruption continues that run; an
evaluation failure creates a new evaluation attempt without retraining. A
failed initialization without a safe checkpoint is preserved and reported by
the existing recovery adapter rather than silently discarding progress.

`llm_training_episode_metrics.jsonl` stores the actual per-episode EE, timely
delivery, movement energy, lambda, RoI count, exploration state, base reward,
beta-weighted extra reward, combined reward, and every beta-weighted feature
contribution. The writer checks that feature contributions sum to extra reward
and that base plus extra equals combined reward. Search feedback summarizes
episodes 1-200, 201-400, 401-600, and 601-800 without recomputing history from a
later lambda.
