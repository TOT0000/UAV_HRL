# Offline LLM feature-design agent

`run_llm_agent.py` is an optional autonomous alternative to the fixed revision
loop in `run_llm_design.py`. It uses Hugging Face `smolagents==1.26.0` and its
`ToolCallingAgent`. The original design CLI, training, and evaluation modules do
not import smolagents and continue to work when this optional dependency is not
installed.

The agent may choose the order in which it reads the contract or fixed samples,
submits immutable candidate versions, runs small or full isolated validation,
performs formal fixed-pair scoring, and retrieves prior reports. It receives no
shell, arbitrary file, project-write, sample-write, evaluator-write, or training
tool. Sample queries expose only fields produced by the existing current-only
adapter. Sample IDs, baseline rewards, and scoring results are labelled as
diagnostics and are not candidate inputs.

The host—not the model or framework final answer—approves a candidate only after
full validation and the existing all-lambda Lipschitz rule pass. The resulting
`approved/` directory uses the existing artifact contract consumed by
`td3_dinkelbach_llm`. No training starts automatically.

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
  --max-output-tokens 4096 `
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
  --max-output-tokens 4096 `
  --max-model-calls 20 `
  --reasoning-effort low
```

Resume an interrupted run without repeating completed validation or evaluation
cache entries:

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --resume <RUN_DIRECTORY>
```

Use the exact ID returned by LM Studio's model list for another local model. At
the time this contract was verified, visible IDs included
`qwen/qwen3.5-9b` and `google/gemma-4-e4b`; do not assume either is loaded at a
later time. Omit `--reasoning-effort` for these adapters:

```powershell
C:\Users\user\anaconda3\envs\LLM_HRL\python.exe run_llm_agent.py `
  --fixed-sample <FIXED_BASELINE_DIRECTORY> `
  --provider lmstudio `
  --base-url http://127.0.0.1:1234/v1 `
  --model qwen/qwen3.5-9b `
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

`--max-model-calls` counts actual agent generations. Planning is disabled,
provider generation retries are zero, and the SDK's extra max-step final-answer
generation is replaced by a deterministic host result, so it cannot exceed the
budget. Tool executions have no separate count limit; formal evaluation remains
cacheable but unrestricted. A legal tool call produced by the last allowed
model generation is executed before stopping.

Before each generation, the client budgets the complete framework system
prompt, task, tool definitions, memory, and output reservation. When necessary,
whole older action groups are replaced with a host-generated structured summary;
tool-call/result pairs are never split and the current candidate is not
truncated. If the minimum state still does not fit, the run is saved and stops.
On resume, compatibility hashes and settings are checked, running operations are
marked interrupted rather than passed, completed candidate tests/evaluations use
their content-keyed caches, and requests of unknown completion status are not
automatically resent. The original provider endpoint, context/output budgets,
generation settings, and timeout settings are restored from the run state so a
plain `--resume` does not silently change the request contract. Resume also
requires the saved prompt/tool/interface contracts and Git revision; a changed
implementation must start a separate run instead of mixing evaluation evidence.
