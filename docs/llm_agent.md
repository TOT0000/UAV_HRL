# Offline LLM feature-design agent

`run_llm_agent.py` is an optional autonomous alternative to the fixed revision
loop in `run_llm_design.py`. It uses Hugging Face `smolagents==1.26.0` and its
`ToolCallingAgent`. The original design CLI, training, and evaluation modules do
not import smolagents and continue to work when this optional dependency is not
installed.

The agent may choose the order in which it reads the contract or fixed samples,
submits immutable candidate versions, runs isolated validation over the complete
loaded fixed-sample artifact, performs formal fixed-pair scoring, and retrieves prior reports. It receives no
shell, arbitrary file, project-write, sample-write, evaluator-write, or training
tool. Sample queries expose only fields produced by the existing current-only
adapter. Sample IDs, baseline rewards, and scoring results are labelled as
diagnostics and are not candidate inputs.

The host—not the model or framework final answer—approves a candidate only after
full validation and the existing all-lambda Lipschitz rule pass. The resulting
`approved/` directory uses the existing artifact contract consumed by
`td3_dinkelbach_llm`. No training starts automatically.

`submit_candidate` accepts the simplified `{features, code}` object directly.
Each ordered feature contains only `name`, `description`, and finite signed
`reward_weight`; no weight-sum restriction applies. The model must not serialize
that object into a second `candidate_json` string. The host deterministically
adds internal artifact metadata and saves both forms. The `code` member remains
ordinary decoded Python source. It may return a numeric list or NumPy array;
the host validates raw `[0,1]` values before float32 conversion.

## Install

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe -m pip install -r requirements-llm-agent.txt
```

The version is pinned because the smolagents API is documented as experimental.
Official references used for this integration are the Hugging Face
[`ToolCallingAgent`, `Tool`, and model API documentation](https://huggingface.co/docs/smolagents/reference/agents),
the [`smolagents` 1.26.0 release](https://pypi.org/project/smolagents/1.26.0/),
and LM Studio's [OpenAI-compatible tool-use documentation](https://lmstudio.ai/docs/developer/openai-compat/tools).
The shared streamed-tool adapter also follows OpenAI's
[function-calling stream aggregation contract](https://developers.openai.com/api/docs/guides/function-calling);
GPT-4o is documented as supporting both streaming and function calling.

## Commands

Dry-run with the fixed baseline (writes the full task prompt, framework system
prompt, tool schemas, settings, and conservative context estimate, but sends no
model request):

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --provider lmstudio `
  --base-url http://127.0.0.1:1234/v1 `
  --model openai/gpt-oss-20b `
  --context-length 50000 `
  --max-output-tokens 8192 `
  --max-model-calls 20 `
  --reasoning-effort low `
  --dry-run
```

Run the currently loaded GPT-OSS model:

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --fixed-sample results/llm_baselines/baseline-20260927T130708Z-ba4bf287 `
  --provider lmstudio `
  --base-url http://127.0.0.1:1234/v1 `
  --model openai/gpt-oss-20b `
  --context-length 50000 `
  --max-output-tokens 8192 `
  --max-model-calls 20 `
  --reasoning-effort low
```

Resume an interrupted run without repeating completed validation or evaluation
cache entries:

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --resume <RUN_DIRECTORY>
```

If a run is `paused_budget_exhausted`, explicitly add another 20 model calls.
The saved cumulative usage is retained: a 20-call run that used all 20 calls
will have a total budget of 40 and 20 calls remaining after this command.

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --resume <RUN_DIRECTORY> `
  --additional-model-calls 20
```

`--additional-model-calls` is valid only with `--resume`. A resume dry-run may
show the proposed effective budget, but it neither sends a request nor records
the extension. It validates an in-memory copy and writes any preview prompt and
metadata to a separate `results/llm_agent_previews/...` directory (or the
explicit `--output-dir`), never beneath or over the source run. The source
run's files, running-operation states, timestamps, candidate history, and
budget remain byte-for-byte unchanged, including on compatibility or context
budget failure. A real resume records the timestamp, old and new total, added
amount, and cumulative calls already used. Resuming without an extension uses
the saved remaining budget; an exhausted run stays paused without a model call.

Use the exact ID returned by LM Studio's model list for another local model. At
the time this contract was verified, visible IDs included
`qwen/qwen3.8-27b`; do not assume it is loaded at a later time. The common LM
Studio adapter accepts any exact inventory ID. Do not apply GPT-OSS-only
`--reasoning-effort` to Qwen, Gemma, or other adapters:

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --fixed-sample <FIXED_BASELINE_DIRECTORY> `
  --provider lmstudio `
  --base-url http://127.0.0.1:1234/v1 `
  --model qwen/qwen3.8-27b `
  --context-length <ACTUAL_LOADED_CONTEXT_LENGTH>
```

For GPT-4o, the API key is read only from `OPENAI_API_KEY`; the official OpenAI
base URL is enforced and paid requests are not retried automatically:

```powershell
$env:OPENAI_API_KEY = '<API_KEY>'
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --fixed-sample <FIXED_BASELINE_DIRECTORY> `
  --provider openai `
  --model gpt-4o `
  --context-length <CLIENT_BUDGET>
```

`--context-length` is a client-side request budget. It does not load, reload, or
change a model's LM Studio context setting.

## Persistence, budgets, and recovery

Each run gets a unique directory under `results/llm_agents/<model>/` unless
`--output-dir` is supplied. `agent_state.json` stores candidate versions,
parents, issue evidence, completed test/evaluation cache entries, framework tool
call IDs, tool results, model-call count, and stop reason. Complete prompts,
requests, SSE events, reasoning/final content, reports, and generated candidates
are stored separately in the same run.

Tool results are committed to framework memory one call at a time. If a later
call in the same model response has an unknown name or invalid arguments, prior
successful results remain visible on the next model turn and are not replayed.
The failed call receives an actionable result keyed by its tool-call ID; later
calls in that response are explicitly recorded as `skipped_after_tool_error`.
Approval remains stronger: once a formal evaluation approves, every remaining
call is `skipped_after_approval` and generation stops.

`test_candidate` accepts only `candidate_id` and always evaluates every sample
in the loaded fixed-sample artifact. Its report distinguishes the fixed-sample
count, attempted samples, successful outputs, and whether complete outputs were
obtained. `query_samples` remains a
bounded inspection tool and is not validation. Numeric constant diagnostics are
produced only after complete outputs exist; fewer than two evaluated samples are
reported as insufficient for a variation judgment. Empty-probe checks remain
separate from fixed-sample distribution statistics. A complete candidate test
still does not approve a design.

The host, not model prose, controls completion. Before an approved artifact
exists, `final_answer` and complete plain-text replies become recorded
`completion_rejected` observations containing current validation/evaluation
state, unresolved issues, and remaining call budget. The next model request sees
that observation and continues. Repeated stop requests consume the normal model
call budget and eventually pause the run; no extra final-answer generation is
added. A formal evaluation that saves the approved artifact remains the sole
successful terminal condition.

`--max-model-calls` counts actual agent generations. Planning is disabled,
provider generation retries are zero, and the SDK's extra max-step final-answer
generation is replaced by a deterministic host result, so it cannot exceed the
budget. Tool executions have no separate count limit; formal evaluation remains
cacheable but unrestricted. A legal tool call produced by the last allowed
model generation is executed before stopping.

The terminal run states distinguish `approved`, `paused_budget_exhausted`,
`interrupted`, transport/provider failures, and context-budget failures. A
paused run is unfinished and has no approved artifact. Within one model
response, tool calls run sequentially in the provided order. If formal
evaluation approves a candidate, later calls from that same response are saved
as `skipped_after_approval`, receive no side effects, and no further generation
is requested.

Before each generation, the client budgets the complete framework system
prompt, task, tool definitions, memory, and output reservation. When necessary,
whole older, already-delivered action groups are replaced with a host-generated
structured summary. A completed tool result remains pending until it is included
in an actual request that receives a complete accepted model response; a failed
or cancelled request leaves it pending for resume and never re-executes the tool.
Pending tool-call/result pairs are not split or silently discarded, and the
current candidate is not truncated. Each raw result is preserved separately
from its bounded model-facing view with a record ID and content hash.

`query_samples`, `inspect_interface`, and `get_history` return stable pages with
the returned range, remaining count, `has_more`, and exact
`next_query_arguments`. Sample pages preserve the requested field order and
filter, and never offer an unchanged zero-progress page. Candidate tests and
formal evaluations still run over their complete data; only their model-facing
reports are summarized, with status, test scope, every lambda result, and a
`get_history` lookup retained. The compact work state records recent query
indexes and any still-pending result, so resume can deliver it without rerunning
the operation. “Delivered” means included in a request that completed; it does
not claim that the model understood the result.

Formal-evaluation delivery is checked again after every reduction step against
the serialized representation actually placed in the tool-result message. Its
bounded form keeps every lambda score and pass decision, expands a shared
maximum pair only once, and limits feature/finding evidence by deterministic
rules. Full traces, sample inputs, contracts, and omitted feature details remain
in the registered report. Candidate-history results are the intentional
exception to the general tool-result character limit: candidate source remains
complete and the whole request budget decides whether it can be delivered.

Evaluation-history pages summarize each registered report before applying the
page-size limit. Every entry keeps the candidate/report IDs, approval status,
all per-lambda baseline and candidate estimates, improvement, required margin,
pass result, and a directly executable `details_query`. Detailed reports are
returned as lossless ordered text segments with character and line positions;
concatenating each segment's `text` reconstructs the canonical report JSON.
This lets a single unusually long JSON line be paged without silent truncation
or a zero-progress page.

The estimate uses the same serialized messages and tool schemas sent to the
provider, plus the configured output reservation. If the minimum state and a
usable pending result page still do not fit, the run is saved and stops without
consuming another model call; it does not continue with the result removed.
The compact work state includes the full current candidate, a deterministic
bounded grouping of unresolved or unverified issues, exactly one current or
nearest-ancestor formal-evaluation summary, compact history indexes, and
cumulative call-budget state. Repeated issues retain source counts and recent
source IDs; omitted details remain available through `get_history`. Older
evaluation reports are indexed instead of expanded, and run-history lookup is
paged. Shared maximum pairs are stored once and referenced by all applicable
lambda entries. If removing delivered history is insufficient, the host uses a
second bounded work-state level before failing; neither level truncates the
candidate. Context compaction rebuilds the common task around this one fresh
work state, so the original stale summary is not retained alongside a new copy.
Saved compaction diagnostics separate serialized character counts from inexact
token estimates and identify system, task/interface, tools, candidate,
evaluation, issue/query summaries, pending results, retained history, and the
exact output-token reservation. The unified full-request estimate remains the
accept/reject decision.
On a real resume, compatibility hashes and settings are checked, running operations are
marked interrupted rather than passed, completed candidate tests/evaluations use
their content-keyed caches, and requests of unknown completion status are not
automatically resent. The original provider endpoint, context/output budgets,
generation settings, and timeout settings are restored from the run state so a
plain `--resume` does not silently change the request contract. Resume also
requires the saved prompt/tool/interface contracts and Git revision; a changed
implementation must start a separate run instead of mixing evaluation evidence.

Provider responses are not exposed to tool execution until streaming completed
and the termination reason is accepted. Standard `tool_calls` is accepted;
LM Studio `stop` is accepted only when it includes complete tool calls. Truncated
(`length`), refused, content-filtered, interrupted, missing-terminal, and unknown
termination responses are preserved as failures and are not retried or treated
as candidate-validation feedback.

Agent runs created by another Git revision cannot be resumed by design. This
prevents old framework-memory, tool, or validation evidence from being silently
mixed with a changed implementation. Start a new run when compatibility checks
reject an older directory; the old directory remains intact for inspection.
This includes runs created with the former string-valued `candidate_json` tool
contract: their candidates and approved artifacts remain readable, but the agent
conversation cannot be resumed under the object-valued tool contract. Runs from
the earlier structured-candidate contract that exposed small-test arguments are
also incompatible with the complete-fixed-sample-only tool contract; their
saved candidates and approved artifacts remain readable. Tool contract v8 now
exposes only the simplified `{features, code}` model object; older verbose-object
agent conversations are likewise inspectable but must start a new run, while
their v2 internal candidates and approved artifacts remain loadable.
