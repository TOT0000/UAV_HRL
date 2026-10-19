# LLM-guided TD3 and safe-DDQN exploration

`td3_dinkelbach_llm_explorer` is an additive comparison method based on
`td3_dinkelbach`. It keeps the same observations, rewards, constraints,
Dinkelbach lifecycle, replay buffers, networks, optimizers, task assignment,
packet model, and update schedules. Its only training interventions are:

- a fixed per-episode `[16,3]` mean offset added to the TD3 Actor action before
  the existing Gaussian behavior noise is projected; and
- a fixed pre-mask `[16,17]` routing prior used only in safe-DDQN's epsilon
  branch. The host masks illegal actions and renormalizes 100% of the remaining
  prior. It uses legal-uniform only when legal prior mass is zero and records
  that fallback separately.

The first guided episode begins only after 10,000 joint movement transitions
have already completed and a previous episode is available. With the standard
60-transition horizon this is episode 168. The implementation checks the
transition count rather than the episode number. Each guided episode normally
makes one analysis request and one numerical request to `gpt-4o-mini`. The two
requests use independent system and user prompts. The numerical request carries
the complete analysis, exact current bounds, compact current state, application
status, and complete rounded historical matrices; it does not repeat the large
movement/routing tables, representative cases, or analysis system prompt. A
stage is attempted at most twice; the underlying OpenAI client has retries
disabled.

Prompts, responses, validation results, usage, timings, and an append-only event
journal are stored under `<run>/llm_explorer/`. Completed responses are reused
only when the run/episode, complete input hash, model, prompt version, and
configuration, summary, and cache identities all match. Final episode results are also cached for
full success, partial success, and fallback outcomes. The analysis payload uses
versioned columns/rows aggregates for all 16 UAVs and all routing senders,
complete 17-action counts, decision-time expected capacity, realized shared
service, WAIT-with-HOL interval unions, immutable S2U completion endpoints, and
at most eight deterministic cases. General LLM-input values use a detached
at-most-three-decimal presentation; counts, IDs, booleans, validity, exact
validation bounds, small non-zero multipliers, raw records, applied guidance,
and checkpoint state keep their required semantics and precision. Every
analysis, numerical, and retry
request is checked with the `o200k_base` tokenizer against the 128,000-token
context limit, including reserved output and a safety margin. Token budgets are
saved beside each attempt. The payload contains filtered observable state and
never serializes a scenario manifest or undiscovered RoI and SR ground truth.
The first-generation controller schema is `uav-hrl-llm-explorer-firstgen-v6`,
prompt schema is v9, summary schema is `complete-core-feedback-v4`, and cache
schema is `episode-fixed-dual-guidance-cache-v6`. The first-generation numerical
Chat Completions request uses a strict JSON Schema for exactly one object with
complete `[16,3]` and `[16,17]` numeric matrices. Host validation remains
authoritative: probability rows must sum to one within `0.000001`, and grossly
invalid rows are rejected rather than normalized. The parser accepts bare JSON
or a complete outer plain/`json` Markdown fence, but does not extract JSON from
explanatory text or repair truncated output. Representative-case service
durations and HOL remaining deadlines use explicitly named millisecond fields,
while absolute `time_s` remains in seconds. The analysis dictionary contains
the runtime-owned C9/C10/COM formulas, populations, weights, and epsilon, plus
one U2U/U2G routing-state-normalizer mapping instead of repeating constants in
each representative link. Slot-start HOL fields remain
observation counts; actual WAIT time is accumulated on the PacketEngine event
clock and unavailable legacy values stay unknown. Movement feedback separates
the requested matrix, Actor pre-projection offset, movement-mask-qualified
application, and final projected/executed action. Exact resume permits the known
format-only v8/v5 first-generation protocol to move to v9/v6 without resetting
the saved training, optimizer, replay, RNG, guidance, feedback, request, or
fallback state. The migration and effective episode are recorded in resolved
configuration and later checkpoints. Other old or unknown protocol combinations
remain incompatible. Numerical template hashes and the output protocol are part
of the cache identity, and the v9 numerical stage uses a versioned directory, so
old numerical artifacts are preserved but never silently reused.

## PowerShell commands

Live training:

```powershell
python -m pip install -r requirements-llm-explorer.txt
$env:OPENAI_API_KEY = "<your-key>"
$episodes = [int](Read-Host "Training episode count")
& "C:\Users\user\anaconda3\envs\LLM_HRL\python.exe" .\run_experiment.py td3_dinkelbach_llm_explorer --episodes $episodes --seed 20260817 --llm-explorer-model gpt-4o-mini --output-root C:\uav_llm_runs
```

Exact resume uses the persisted run configuration and response cache:

```powershell
python run_experiment.py resume results\td3_dinkelbach_llm_explorer\<run-directory>
```

Formal evaluation never calls the LLM and does not require `OPENAI_API_KEY`:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
$checkpointEpisode = [int](Read-Host "Checkpoint episode")
python run_experiment.py evaluate results\td3_dinkelbach_llm_explorer\<run-directory> --checkpoint-episode $checkpointEpisode
```

One-episode offline smoke (no key and no network request):

```powershell
python run_experiment.py td3_dinkelbach_llm_explorer --smoke --llm-explorer-mock --output-root tmp\llm-explorer-smoke
```

Targeted offline tests:

```powershell
python -m pytest tests\test_llm_explorer.py tests\test_experiment_framework.py tests\test_joint_td3.py tests\test_ddqn_routing.py -q
```

The mock client is deterministic and returns neutral biases and uniform priors.
It exists for smoke/integration validation only; it is not a scientific LLM
baseline. A live remote response is not guaranteed reproducible from a local
seed, while a saved matching response is replayable and revalidated.

## Compact annealed v3 comparison

`td3_dinkelbach_llm_explorer_v3` is isolated from the original method above.
It sends verified full-episode summaries plus at most eight deterministic
illustrative cases, accepts named per-sender routing probabilities, and caps
each request at 20,000 input tokens. Its host-side weight is fixed by the
1-based episode number: after the 10,000-transition warmup, `w(168)=1`,
`w(384)=0.5`, and `w(600)=0`. Starting at episode 600 it creates no client and
makes no LLM request; training continues through the original no-prior paths.
Artifacts are stored under `<run>\llm_explorer_v3\` and do not share caches or
checkpoint identity with the original explorer.

An 800-episode PowerShell run using the project Conda environment is:

```powershell
& "C:\Users\user\anaconda3\envs\LLM_HRL\python.exe" run_experiment.py td3_dinkelbach_llm_explorer_v3 --episodes 800 --seed 20260817 --checkpoint-interval 800 --llm-explorer-model gpt-4o-mini --output-root C:\uav_llm_runs
```

The command prints the exact run directory. Evaluate its episode-800 model
checkpoint without an LLM call by replacing `<run-directory>` below:

```powershell
& "C:\Users\user\anaconda3\envs\LLM_HRL\python.exe" run_experiment.py evaluate "C:\uav_llm_runs\td3_dinkelbach_llm_explorer_v3\<run-directory>" --checkpoint-episode 800
```

Training metadata/history and checkpoints are below the run directory; v3 LLM
prompts, responses, token budgets, timings, summaries, and event journal are in
`llm_explorer_v3`. Evaluation output is created below `evaluation\ep_800`.
Offline tests write only to pytest temporary directories:

```powershell
& "C:\Users\user\anaconda3\envs\LLM_HRL\python.exe" -m pytest tests\test_llm_explorer.py tests\test_llm_explorer_v3.py tests\test_experiment_framework.py tests\test_joint_td3.py tests\test_ddqn_routing.py -q
```

## TD3-only annealed comparison

`td3_dinkelbach_llm_explorer_td3_only` uses the v3 warmup, two-stage request,
episode-fixed raw bias, and linear `w` schedule, but its numerical response has
only the complete `[16,3]` `td3_bias` matrix. The host applies `w * raw_bias`
once. Safe-DDQN receives no LLM prior and remains on its original legal-mask,
epsilon-greedy, legal-uniform exploration, and greedy-selection paths.

The movement-focused analysis input includes complete per-UAV movement
statistics and a compact communication bottleneck summary. The latter is
computed locally from all recorded routing decisions: queue backlog, observed
head-of-line deadline, legal non-WAIT action availability, GS availability,
and legal-link reference-capacity summaries, with explicit counts and units.
System QoS counts remain system-wide. The records do not reliably provide
per-UAV timely-GS-delivery attribution, per-UAV final deadline violations, or
realized per-link throughput, so those fields are marked unavailable rather
than inferred. Full raw movement and routing diagnostics remain in each
episode's feedback artifact.

An 800-episode PowerShell run with periodic checkpoints is:

```powershell
& "C:\Users\user\anaconda3\envs\LLM_HRL\python.exe" .\run_experiment.py td3_dinkelbach_llm_explorer_td3_only --episodes 800 --seed 20260817 --checkpoint-interval 100 --llm-explorer-model gpt-4o-mini --output-root C:\uav_llm_runs
```

Artifacts use the independent `llm_explorer_td3_only` directory and schema.
Resume the exact printed run directory with `run_experiment.py resume`; a v3
checkpoint or response cache is not accepted as a TD3-only checkpoint/cache.
The deterministic `--llm-explorer-mock` option is for offline plumbing tests,
not a substitute for a live-model or 800-episode scientific result.
