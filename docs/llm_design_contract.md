# Offline LLM state/reward design

`run_llm_design.py` is an offline design tool. It loads an existing
`fixed_samples.json`/`fixed_samples.npz`/`baseline_report.json` set, asks one
explicit LM Studio model for a current-only state/reward candidate, validates
the candidate in an isolated worker, evaluates it on the baseline-defined
fixed pair set, and either requests a complete revision or saves the first
approved design. It never imports a candidate as a trusted project module and
never invokes training.

The client uses LM Studio's OpenAI-compatible `GET /v1/models` and
`POST /v1/chat/completions` endpoints with streaming enabled and without a
`response_format` override. Responses remain subject to duplicate-key-aware
strict JSON, schema, AST, worker, and numeric validation. The requested model must exactly
match a model identifier visible through `/v1/models`; no model substitution is
performed. `LM_STUDIO_API_TOKEN` is used when present and is never saved.

`--context-length` is a client-side budget. The OpenAI-compatible chat endpoint
does not use it to modify the model load configuration. During a real run the
tool separately queries `/api/v1/models` for the model-supported maximum,
loaded instance context, quantization, and reasoning metadata when available.
The effective budget is the minimum known bound. When no matching tokenizer is
installed, prompt size is reported as an explicitly inexact character-based
interval with chat-template and output reservations; an upper-bound overflow
stops before generation. On a revision after invalid JSON, only the failed raw
final-content excerpt may be explicitly shortened, with truncation markers and
preference for the parser-error location. The environment interface, schema,
evaluation rules, and output reservation are never silently removed.

## Current-only candidate interface

The candidate function receives only information available before the current
movement action:

```python
compute_extra_state(obs, constants)
```

`obs` contains the original 531-D state, the current 16-D movement mask, and
the named current auxiliary snapshot fields. It excludes action, all `next_`
fields, delivered data, movement energy, C9/C10/COM reward components,
Dinkelbach lambda, and episode/scenario/checkpoint/source trace identifiers.
The prompt generated for each run is the authoritative field/shape/dtype/unit,
mask, compact-row/ID, link-axis, sensing, capacity, and reward description.

`constants.json` records every exposed value together with its type, unit,
meaning, and source. Values are obtained from the fixed artifact's compatible
checkpoint/environment contracts. Only constants absent from per-run metadata
but authoritative in the checked-in environment configuration are added, with
that code source recorded.

Candidate JSON uses schema `uav-hrl-llm-shared-feature-candidate-v2` and
`reward_input_mode="current_only"`. Feature outputs are one-dimensional
`float32` arrays within their declared subranges of `[0,1]`. Each feature has
a finite signed `reward_weight`, and `sum(abs(reward_weight)) <= 1`. The exact
same unweighted feature vector is appended to state and weighted by the host,
without clipping or a second candidate call:

```text
s_aug = concat(s_original, features)
r_extra = dot(feature_reward_weights, features)
r_candidate(lambda) = r_base(lambda) + beta * r_extra
```

The worker statically excludes imports, arbitrary attributes/calls, reflection,
I/O, dynamic execution, randomness, and mutation. It exposes only the listed
NumPy subset and safe built-ins, runs in a terminable subprocess, checks every
fixed sample twice for determinism, compares inputs before/after, and exercises
an empty/missing-data probe. Fixed samples and the empty probe share the same
dtype, shape, finite-value, declared-range, global-range, determinism, mutation,
and weighted-extra-reward checks.

A limited AST redundancy check rejects direct scalar copies such as
`obs["state"][i]`, including simple straight-line local aliases, and rejects
multiple returned features whose statically resolved output expressions are
identical. It intentionally does not claim general algebraic equivalence or
full control-flow data-flow analysis. Numeric equality found only on the fixed
samples remains a diagnostic warning rather than an automatic rejection;
derived features may still use the original state.

For straight-line outputs that can be resolved reliably, source-field errors
identify the feature index, JSON path, field, and candidate-code locations,
including dependencies used by masks and conditions. For path-dependent
control flow, the report lists the certain field locations but explicitly does
not invent an output index. Failed candidates also receive a semantic
fingerprint over feature metadata, source fields, weights, and normalized code
AST. Changing only `candidate_name`, comments, or formatting is reported as a
repeat of the earlier failed attempt; this is deliberately not a claim of
general mathematical-equivalence detection.

## Fixed-pair acceptance

The baseline sample hash, pair hash, counts, lambdas, thresholds, and each saved
baseline `L_hat` are checked and recomputed before scoring. Candidate distances
are

```text
sqrt(original_distance^2 + extra_feature_distance^2)
```

but membership always comes from the saved original-state condition
`original_distance > distance_epsilon`. Original zero/near-zero pairs remain
excluded even if extra features separate them. They receive separate candidate
diagnostics. Every lambda must satisfy:

```text
margin = max(absolute_tolerance, relative_tolerance * abs(L_baseline))
L_candidate < L_baseline - margin
```

The defaults are `1e-12` absolute and `1e-6` relative. Acceptance means only
that code checks and this finite fixed-pair metric passed; it is not a claim of
global correctness or improved training.

## Commands

Dry-run against the current formal fixed baseline (no API request):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --model qwen/qwen3.5-9b `
  --context-length 40000 --max-output-tokens 4096 --dry-run
```

Qwen:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --base-url http://127.0.0.1:1234/v1 `
  --model qwen/qwen3.5-9b `
  --context-length 40000 --max-output-tokens 4096 `
  --temperature 0.3 --seed 20260927 --max-attempts 5 `
  --beta 1.0 --batch-size 128 `
  --timeout 600 --worker-timeout 120
```

Gemma uses the same task, schema, evaluation, and thresholds:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --base-url http://127.0.0.1:1234/v1 `
  --model google/gemma-4-e4b `
  --context-length 40000 --max-output-tokens 4096 `
  --temperature 0.3 --seed 20260927 --max-attempts 5 `
  --beta 1.0 --batch-size 128 `
  --timeout 600 --worker-timeout 120
```

`--connect-timeout` applies only to TCP/TLS establishment (default 30 seconds).
After a connection exists, `--timeout` limits inactivity while sending the
request, waiting for response headers, the first SSE bytes, and subsequent SSE
bytes (default 600 seconds). `--total-timeout` is an absolute per-generation
limit covering all those phases (default 1800 seconds). Timeout/cancellation
forcibly shuts down the transport and uses bounded cleanup; it does not enter a
candidate revision round. `--worker-timeout` independently limits untrusted
candidate execution and defaults to 120 seconds.

The default root is
`results/llm_designs/<model-slug>/<unique-run-name>/`. Every attempt keeps its
full prompt, request, raw response, finish reason/usage, extracted reasoning,
candidate/code, validation, evaluation, and feedback. A failed run keeps
history and `run_metadata.json` but has no `approved/` directory. A revision
uses only the immediately preceding output: a successfully parsed full
candidate plus its validation/evaluation feedback, or the failed raw final
content plus its parse error. Reasoning-only text is stored locally but is not
replayed as candidate code. Full validation reports remain on disk. If parsed
candidate feedback is too large for the fixed context budget, a stable summary
keeps representative distinct error types/root locations and records original,
included, and omitted counts in `prompt_feedback.json`. The candidate itself
is never truncated or rewritten; if even minimum feedback cannot fit, the run
stops with `context_budget_exceeded`. The first
passing attempt creates:

```text
approved/artifact.json
approved/candidate.json
approved/candidate.py
approved/constants.json
approved/validation_report.json
approved/evaluation_report.json
```

`artifact.json` includes content/file hashes, field order, weights, interface
version, model/sample/pair provenance, and approved status. Later integration
can load and compute the isolated outputs without changing training yet:

```python
from llm_baseline import load_fixed_samples
from llm_candidate import load_approved_design

fixed_arrays, _ = load_fixed_samples("results/llm_baselines/<baseline-run>")
design = load_approved_design("results/llm_designs/<model>/<run>/approved")
extra_state, r_extra = design.evaluate_fixed_samples(
    fixed_arrays, timeout=120
)
```

The formal `td3_dinkelbach_llm` training method is registered separately; this
offline tool never starts it. Approved dual-function v1 artifacts are not
silently converted because their state and reward functions may have different
semantics. The v2 runtime rejects them with a redesign/retraining message.

LM Studio interface references used by this implementation:

- [OpenAI-compatible model listing](https://lmstudio.ai/docs/developer/openai-compat/models)
- [OpenAI-compatible chat completions and supported parameters](https://beta.lmstudio.ai/docs/developer/openai-compat/chat-completions)
- [Native model/load inventory and context metadata](https://lmstudio.ai/docs/developer/rest/list)
