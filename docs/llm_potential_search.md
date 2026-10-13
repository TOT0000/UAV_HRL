# LLM potential-based reward shaping search

`td3_dinkelbach_llm_potential` is an independent comparison method. It keeps the
original centralized TD3 state and actor/critic dimensions unchanged. An approved
candidate computes bounded, unweighted potential components for one observation;
the host computes

```text
Phi(s) = sum_k w_k f_k(s)
F_t = beta * (gamma * Phi(s_next) - Phi(s_current))
r_train,t = r_base,t + F_t
```

`beta` is fixed at 1. The artifact's `gamma` must exactly equal the TD3 target
discount. The current TD3 configuration resolves this value to `1.0` from the
movement hyperparameters in `experiment_config.py`; TD3 uses
`reward + not_done * gamma * next_q`.

## Transition and terminal convention

The environment builds the current observation before movement and the next
observation after movement, sensing, service, delivery, energy, and penalty
updates. Potential shaping uses these two snapshots before any reset. The current
60-second horizon is represented as `done=True` on the final movement transition,
so its replay bootstrap mask is zero and the host substitutes a zero effective
next potential. The current environment does not expose a separate time-limit
`truncated` transition. The potential helper nevertheless requires a real next
observation when `terminated=False` and bootstrapping is retained; it never uses a
reset observation as the previous episode's next state.

Only the shaped training reward enters the potential method's TD3 replay update.
Timely delivery, movement energy, Dinkelbach lambda updates, and reported base
reward remain based on the existing unshaped quantities. If replay reconstructs
the base reward under a later lambda, it adds the stored shaping value exactly
once afterward.

## Search contract

`run_llm_potential_search.py` generates four candidates using four separate model
calls per round. Every executable, distinct candidate is trained independently
and evaluated on the exact baseline manifest. Round one requests distinct initial
designs. Later rounds use a fixed historical-best reference and request ADD,
REMOVE, REWEIGHT, and REDESIGN variants. The default contract is 400 training
episodes, 100 fixed-8-RoI evaluation episodes, 50-episode summary blocks, five
rounds, and at most five repair calls after each initial response.

Selection uses the arithmetic mean of per-episode energy efficiency in Mbit/J,
not a ratio of totals and not the shaped training return. Increasing
`--max-rounds` on resume schedules only new rounds. Training/evaluation budgets,
seeds, manifests, artifacts, and completed phases remain compatibility-checked.

The baseline evaluation path may be the directory containing the selected
`paper_evaluation_metadata.json`, or a higher root only when it resolves to one
unambiguous result. The saved point supplies the exact `roi_8` manifest path.
Preflight requires checkpoint episode 400, method
`td3_dinkelbach`, exactly 100 episodes, the evaluation result files, and the saved
scenario manifest. It also cross-checks the referenced source run and checkpoint.
An incomplete baseline pauses the formal search; the command does not start a
replacement evaluation.

## PowerShell examples

Set paths for the current checkout and the completed baseline evaluation leaf:

```powershell
$Python = 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe'
$Repo = 'C:\Users\user\Desktop\project\YM_Post_Disaster_Training\UAV_HRL'
$Validation = Join-Path $Repo 'results\llm_samples\td3d-ep250-20260927T105459Z-83c75c4c'
$BaselineRun = Join-Path $Repo 'results\td3_dinkelbach\run-seed20260817-359edbc-20260922T132509272242Z-8601a052'
$BaselineEval = 'C:\uav_eval\baseline400\fixed_roi\td3_dinkelbach\<completed-evaluation-id>'
$SearchRoot = 'C:\uav_potential_search'
$RuntimeRoot = 'C:\uav_potential_runtime'
Set-Location $Repo
```

GPT-4o through OpenAI (the key is read only from the environment):

```powershell
$env:OPENAI_API_KEY = Read-Host 'OPENAI_API_KEY'
& $Python .\run_llm_potential_search.py `
  --validation-source $Validation `
  --baseline-run $BaselineRun `
  --baseline-evaluation $BaselineEval `
  --provider openai --model gpt-4o `
  --max-rounds 5 `
  --output-root $SearchRoot --runtime-root $RuntimeRoot
```

LM Studio, limited to three rounds:

```powershell
& $Python .\run_llm_potential_search.py `
  --validation-source $Validation `
  --baseline-run $BaselineRun `
  --baseline-evaluation $BaselineEval `
  --provider lmstudio --model '<loaded-model-id>' `
  --endpoint 'http://127.0.0.1:1234/v1' `
  --max-rounds 3 `
  --output-root $SearchRoot --runtime-root $RuntimeRoot
```

Continue a three-round search through round five, then later through round eight:

```powershell
$SearchDir = 'C:\uav_potential_search\<search-directory>'
& $Python .\run_llm_potential_search.py --resume $SearchDir --max-rounds 5
& $Python .\run_llm_potential_search.py --resume $SearchDir --max-rounds 8
```

The search stops after its configured round budget. To continue the selected
candidate manually from its complete episode-400 checkpoint, take that candidate's
`training.run_directory` from `state.json` and run:

```powershell
$SelectedTrainingRun = 'C:\uav_potential_runtime\train\<candidate-training-run>'
& $Python .\run_experiment.py resume $SelectedTrainingRun --target-episodes 1500
```

This continuation retains the candidate code, component order, fixed weights,
normalization, beta, gamma, replay, optimizers, RNG state, and training counters.
