# LLM-guided TD3 and safe-DDQN exploration

`td3_dinkelbach_llm_explorer` is an additive comparison method based on
`td3_dinkelbach`. It keeps the same observations, rewards, constraints,
Dinkelbach lifecycle, replay buffers, networks, optimizers, task assignment,
packet model, and update schedules. Its only training interventions are:

- a fixed per-episode `[16,3]` mean offset added to the TD3 Actor action before
  the existing Gaussian behavior noise is projected; and
- a fixed pre-mask `[16,17]` routing prior used only in safe-DDQN's epsilon
  branch as `0.8 * masked_prior + 0.2 * legal_uniform`.

The first guided episode begins only after 10,000 joint movement transitions
have already completed and a previous episode is available. With the standard
60-transition horizon this is episode 168. The implementation checks the
transition count rather than the episode number. Each guided episode normally
makes one analysis request and one guidance request to `gpt-4o-mini`. A stage
is attempted at most twice; the underlying OpenAI client has retries disabled.

Prompts, responses, validation results, usage, timings, and an append-only event
journal are stored under `<run>/llm_explorer/`. Completed responses are reused
only when the run/episode, complete input hash, model, prompt version, and
configuration identity all match. Final episode results are also cached for
full success, partial success, and fallback outcomes. The request payload uses
versioned compact tables built from a copy of the raw diagnostics; raw training
records retain their original precision. Every analysis, guidance, and retry
request is checked with the `o200k_base` tokenizer against the 128,000-token
context limit, including reserved output and a safety margin. Token budgets are
saved beside each attempt. The payload contains filtered observable state and
never serializes a scenario manifest or undiscovered RoI and SR ground truth.
The compact-input/cache schema is v2; v1 explorer checkpoints are rejected
instead of mixing old prompt identities or inventing historical counters. Start
a new comparison run when migrating from a v1 explorer run.

## PowerShell commands

Live training:

```powershell
python -m pip install -r requirements-llm-explorer.txt
$env:OPENAI_API_KEY = "<your-key>"
$episodes = [int](Read-Host "Training episode count")
python run_experiment.py td3_dinkelbach_llm_explorer --episodes $episodes --seed 20260817 --output-root results
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
