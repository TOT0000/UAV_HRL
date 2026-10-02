# Offline LLM state/reward design

`run_llm_design.py` is an offline design tool. It loads an existing
`fixed_samples.json`/`fixed_samples.npz`/`baseline_report.json` set, asks one
explicit LM Studio or OpenAI model for a current-only state/reward candidate, validates
the candidate in an isolated worker, evaluates it on the baseline-defined
fixed pair set, and either requests a complete revision or saves the first
approved design. It never imports a candidate as a trusted project module and
never invokes training.

The default `--provider lmstudio` client uses LM Studio's OpenAI-compatible
`GET /v1/models` and `POST /v1/chat/completions` endpoints. The
`--provider openai` client uses the official
`https://api.openai.com/v1/chat/completions` endpoint and reads
`OPENAI_API_KEY` only from the process environment; dry-run does not require a
key. OpenAI generation never calls LM Studio's native inventory endpoint and
does not automatically retry a paid request. Both providers stream and omit a
`response_format` override. Responses remain subject to duplicate-key-aware
strict JSON, schema, AST, worker, and numeric validation. The requested model must exactly
match a model identifier visible through LM Studio's `/v1/models`; no model
substitution is performed. For OpenAI, `gpt-4o` may report one of the explicitly
documented GPT-4o snapshots, while an explicitly requested snapshot must match
exactly. This does not admit `gpt-4o-mini` or arbitrary prefixes.
`LM_STUDIO_API_TOKEN`, `OPENAI_API_KEY`, and Authorization headers are never
saved and are removed from the isolated candidate worker environment.

`--context-length` is a client-side budget. Neither provider uses it to modify
server-side model capacity. During a real LM Studio run the
tool separately queries `/api/v1/models` for the model-supported maximum,
loaded instance context, quantization, and reasoning metadata when available.
The effective budget is the minimum known bound. When no matching tokenizer is
installed, prompt size is reported as an explicitly inexact character-based
interval with chat-template and output reservations; an upper-bound overflow
stops before generation. On a revision after invalid JSON, only the failed raw
final-content excerpt may be explicitly shortened, with truncation markers and
preference for the parser-error location. The environment interface, schema,
evaluation rules, and output reservation are never silently removed.

Qwen, Gemma, GPT-OSS, and GPT-4o use the same provider-neutral English master prompt
(`uav-hrl-llm-design-prompt-v7`). Provider adapters change only transport and
provider-specific request fields; they do not maintain separate task prompts.
The saved prompt for each attempt is the exact fully assembled request content.
Its environment interface is generated from the fixed artifact's authoritative
movement-state feature schema plus the replay auxiliary field specifications,
and its baseline values, tolerances, beta, schema/example, supported operations,
and round-specific request are injected before the context budget is computed.

Candidate parsing accepts either a plain JSON object or a whole response made
of exactly one `json`/unlabelled Markdown code fence containing one JSON object.
It removes only that outer fence and records the action in
`candidate_parse_metadata.json`. Explanatory text, multiple/incomplete fences,
other language labels, invalid JSON, duplicate keys, and non-finite JSON values
remain errors; no punctuation, field, number, weight, formula, or Python code is
repaired.

Successful JSON parsing and simplified-candidate schema validation are separate
recorded stages. A syntactically valid response with several bad fields keeps
the parsed object and reports every independently checkable field problem with
its JSON path; static and runtime checks remain explicitly not run until that
schema passes.

## Current-only candidate interface

The candidate function receives only information available before the current
movement action:

```python
compute_extra_state(obs, constants)
```

`obs` contains the original state in the dimension and order declared by the
fixed artifact's authoritative movement-state feature schema, the current
movement mask, and the named current auxiliary snapshot fields. For the current
formal contract these are 531 and 16 dimensions, respectively. It excludes
action, all `next_` fields, delivered data, movement energy, C9/C10/COM reward components,
Dinkelbach lambda, and episode/scenario/checkpoint/source trace identifiers.
The prompt generated for each run is the authoritative field/shape/dtype/unit,
mask, compact-row/ID, link-axis, sensing, capacity, and reward description.

`constants.json` records every exposed value together with its type, unit,
meaning, and source. Values are obtained from the fixed artifact's compatible
checkpoint/environment contracts. Only constants absent from per-run metadata
but authoritative in the checked-in environment configuration are added, with
that code source recorded.

The model-facing JSON contains only ordered `features` entries (`name`,
`description`, finite signed `reward_weight`) and `code`. The host
deterministically adds schema/version, candidate name, current-only mode,
indices, float32 runtime dtype, fixed `[0,1]` range, and dependency diagnostics
before saving the existing `uav-hrl-llm-shared-feature-candidate-v2` internal
artifact. Raw model JSON and the enriched candidate are saved separately.
Feature code may return a one-dimensional numeric list or NumPy array; raw
values are checked before conversion to float32. There is no sum or
absolute-sum restriction on finite weights, and the host never normalizes them.
The exact same unweighted vector is appended to state and weighted once:

```text
s_aug = concat(s_original, features)
r_extra = dot(feature_reward_weights, features)
r_candidate(lambda) = r_base(lambda) + beta * r_extra
```

The worker statically excludes imports, arbitrary attributes/calls, reflection,
I/O, dynamic execution, randomness, and mutation of input-backed data. It exposes only the listed
NumPy subset and safe built-ins, runs in a terminable subprocess, checks every
fixed sample twice for determinism, compares inputs before/after, and exercises
an empty/missing-data probe. Fixed samples and the empty probe share the same
dtype, shape, finite-value, declared-range, global-range, determinism, mutation,
and weighted-extra-reward checks.

The sensing contract defines raw `image_quantity` as RoI area divided by the
canonical oblique camera-footprint area, so it may exceed one. Valid captures
use `packet_max_bits * clip(image_quantity, 0, 1)` for physical packet size;
timely useful VS bits additionally use the coverage frozen at capture. C10
incompleteness is not a separate hard packet-generation gate. These formulas
appear once in the generated prompt; the input table focuses on field shape,
unit, indexing, and validity.

Indexed assignment may fill an independently allocated local array created by
the documented NumPy operations (`zeros`, `ones`, `arange`, or copying
`array`, together with other whitelisted array-producing expressions). Local
aliases and slices retain that ownership. Direct input aliases, input slices,
`asarray` views that may share input storage, rebinding to an input, and a
control-flow merge with any input-backed path remain read-only. An unresolved
write target is rejected as unconfirmed rather than falsely reported as proven
input mutation. Proven immutable numeric locals and independently allocated
NumPy arrays may use augmented assignment. Loop bindings are joined to a finite
fixed point so a later iteration cannot reuse an earlier local-only result;
`and`/`or` preserve their possible operand sources. Python containers are not
treated as proof that every nested member is independent: storing a mutable or
input-backed reference for later nested mutation is conservatively rejected.
This limited ownership analysis is shared by design-time
validation, isolated execution, and approved-artifact loading; the worker's
read-only arrays and before/after comparison remain independent defenses.

The prompt's parseable example is deliberately a zero-weight interface
example, not an approved design recommendation. It selects movement-controlled
UAV rows only when their queue summaries are valid, distinguishes observed
empty queues from missing summaries, defines the empty applicable set as zero,
uses a fixed bounded fraction, and returns a numeric list. Tests run that example
through schema, static, fixed-sample, and empty-probe validation.

A limited AST redundancy check rejects direct scalar copies such as
`obs["state"][i]`, including simple straight-line local aliases, and rejects
multiple returned features whose statically resolved output expressions are
identical. It intentionally does not claim general algebraic equivalence or
full control-flow data-flow analysis. Numeric equality found only on the fixed
samples remains a diagnostic warning rather than an automatic rejection;
derived features may still use the original state.

Allowed literal field accesses are retained as host diagnostics. They are not
model declarations and unresolved per-feature data flow does not fail a
candidate. Access to a field outside the supplied current-only interface still
fails static validation. Failed candidates also receive a semantic
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
  --model qwen/qwen3.8-27b `
  --context-length 40000 --max-output-tokens 4096 --dry-run
```

Qwen3.8-27B (use the exact ID returned by the current LM Studio inventory):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --base-url http://127.0.0.1:1234/v1 `
  --model qwen/qwen3.8-27b `
  --context-length 40000 --max-output-tokens 4096 `
  --temperature 0.3 --seed 20260927 --max-attempts 5 `
  --beta 1.0 --batch-size 128 `
  --timeout 600 --worker-timeout 120
```

GPT-OSS uses the same non-agent workflow (reasoning settings remain provider
capabilities rather than a hard-coded Qwen/Gemma option):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --base-url http://127.0.0.1:1234/v1 `
  --model openai/gpt-oss-20b `
  --context-length 40000 --max-output-tokens 4096 `
  --temperature 0.3 --seed 20260927 --max-attempts 5 `
  --beta 1.0 --batch-size 128 `
  --connect-timeout 30 --timeout 600 --total-timeout 1800 `
  --worker-timeout 120
```

OpenAI GPT-4o dry-run (no API key or network request required):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --provider openai --model gpt-4o `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --context-length 40000 --max-output-tokens 4096 --dry-run
```

One GPT-4o generation attempt (the key remains in `OPENAI_API_KEY`, not the
command line):

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_design.py `
  --provider openai --model gpt-4o `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --context-length 40000 --max-output-tokens 4096 `
  --temperature 0.3 --seed 20260927 --max-attempts 1 `
  --beta 1 --batch-size 128 `
  --connect-timeout 30 --timeout 600 --total-timeout 1800 `
  --worker-timeout 120
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
always uses the immediately preceding output as its correction target: a
successfully parsed full candidate, or the failed raw final content plus its
parse error. It also carries earlier actionable issues whose relevant check
could not run successfully in the latest attempt. Such issues are marked
`not_revalidated`, retain their original source attempt/candidate/location and
their latest actual confirmation separately, and leave the pending prompt only
after the corresponding fine-grained check completes without finding the root.
For example, a completed allowed-call check can resolve a removed `append` even
when a separate source-field check still fails. Python syntax failure leaves
AST-dependent roots `not_revalidated`. `issue_tracker.json` preserves
`confirmed_current`, `not_revalidated`, and `resolved` histories plus the
evidence for each status change. Reasoning-only
text is stored locally but is not replayed as candidate code. Each attempt's
`prompt_feedback.json` is explicitly
the incoming feedback used for that request and records its source attempt;
attempt 1 therefore correctly has `feedback: null`. Feedback generated after
checking the response is saved separately in that attempt's `feedback.json`
with its source and intended next-attempt provenance. Full validation reports
remain on disk. If parsed
candidate feedback is too large for the fixed context budget, a stable summary
keeps independently actionable roots and records original, included, omitted,
and repeated-occurrence counts in `prompt_feedback.json`. Legacy full-metadata
candidates retain structured source-declaration diagnostics, keyed by feature
index and source field; simplified submissions do not ask the model to declare
those fields.
Repeated occurrences of the same root retain representative locations. A
semantically repeated failed candidate carries the original concrete issues as
a flat list alongside the immediately preceding feedback attempt, duplicate
matched attempt, and reused-validation-report attempt.
It does not overwrite the original source with the matched attempt, nest
earlier feedback, or leave only a generic duplicate warning. The candidate itself is
never truncated or rewritten; if even minimum actionable feedback cannot fit,
the run stops with `context_budget_exceeded`.

When implementation checks pass but the fixed-pair Lipschitz criterion does
not, the revision request uses a dedicated evaluation review. The full report
deduplicates maximum-ratio pairs and samples across lambdas and records each
feature value, signed `i-j` difference, `w*f` contribution (before beta), base/
extra/combined rewards, original/augmented distances, the ratio on that exact
pair, and the separate global baseline/candidate estimates. Current-only input
fields are selected from host-recorded literal field accesses in the validated
code (with explicit dependency-analysis limitations); validity masks,
movement mask, constants, axis/ID semantics, and any dependency limitation are
kept separate from post-action reward outcomes. Large input arrays may be
represented relationally: Boolean masks use lossless true coordinates, while
values governed by an authoritative validity mask retain all valid values at
their original coordinates, including valid zeros. Empty valid sets remain
distinct from unavailable fields. SR/RoI/task/link diagnostics add only the
observable ID maps and validity fields needed to interpret the accessed data;
S2U coordinates retain their `[compact_sr_row, receiver_uav]` axis meaning.
State values are not projected through a UAV mask or partially selected unless
their exact dependency indices are known. The complete diagnostic stays
in `evaluation_report.json` and `feedback.json`, while
`prompt_feedback.json` records the exact compact variant sent. Candidate code is
never shortened, and a revision that cannot retain the minimum diagnostic stops
with `context_budget_exceeded`.

Diagnostic contract v2 also derives structured numerical findings from the
complete, uncompressed maximum-pair data. It distinguishes exact equality of
the full added feature vectors, near-but-nonzero differences under a recorded
absolute tolerance, equal weighted extra rewards produced by different feature
vectors, and pairs whose two movement masks are empty. Exact feature equality
supports the narrow conclusion that changing only fixed weights cannot alter
that pair while preserving the feature outputs. Equal reward alone does not:
different features may still change the state distance. Empty movement masks
are reported as an observed condition, not as proof that every observable field
or feature is invalid. These findings guide revision only and do not add a new
candidate rejection rule.

The first passing attempt creates:

```text
approved/artifact.json
approved/candidate.json
approved/candidate.py
approved/constants.json
approved/validation_report.json
approved/evaluation_report.json
```

Runtime exception feedback includes a numbered excerpt from the generated
candidate, safe summaries of candidate-local NumPy array shapes/dtypes, and
relevant structural interface metadata when it can be matched reliably. For
the common unbounded `state[offset::per_uav_width]` error, the worker derives
the UAV block boundary, UAV count, and block width from the saved authoritative
movement-state schema; it does not maintain a second hard-coded layout. A
specific slice correction is emitted only when the exception, local shapes,
mask shape, traceback subscript, and limited straight-line alias provenance all
agree. An unrelated state slice elsewhere in the function is not attribution
evidence, and reassignment invalidates the old alias. Otherwise the report
keeps the original exception and source excerpt without guessing a cause. No array
contents, arbitrary object representations, environment variables, or
credentials are captured.

Execution issues use a normalized AST operation plus its function-structural
path as their cross-attempt identity. Comments and blank lines therefore do not
change identity, while two equivalent-looking operations at different places
remain separate. If an operation cannot be identified reliably, tracking falls
back to the saved source location and conservatively retains separate issues;
an old location always remains attached to its source attempt/candidate.

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

OpenAI references used by the OpenAI provider:

- [Chat Completions API](https://developers.openai.com/api/reference/resources/chat)
- [GPT-4o model and documented snapshots](https://developers.openai.com/api/docs/models/gpt-4o)
