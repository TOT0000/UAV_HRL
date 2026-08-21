# Manifest-driven comparison experiments

`run_experiment.py METHOD` runs exactly one controlled trajectory method per
process. The registry keys are `td3_dinkelbach`, `ddpg_dinkelbach`,
`td3_ratio`, `ddpg_ratio`, `random_action`,
`td3_dinkelbach_no_task_potential`, and
`ddpg_dinkelbach_no_task_potential`. All seven share current K-KM assignment,
16 UAVs, the common Simulator and synchronous projection/movement flow,
safe-DDQN routing, energy/delivery accounting, evaluation, and logging.

```powershell
python -X utf8 run_experiment.py td3_dinkelbach
python -X utf8 run_experiment.py ddpg_ratio --smoke
```

Each invocation creates a new leaf such as
`results/td3_dinkelbach/run-seed20260817-e354dbd-<unique-id>/`.
The scenario manifest, resolved config, per-episode CSV/JSONL history, metadata,
and checkpoints stay below that leaf. Existing run leaves are never reused.

The same simple runner infers method, seed, reward, agent, task-potential flag,
manifest, and formal configuration from an existing run:

```powershell
python -X utf8 run_experiment.py resume results/<method>/<run-id>
python -X utf8 run_experiment.py evaluate results/<method>/<run-id>
```

Evaluation defaults to the formal `ep_2500` checkpoint. Each invocation creates
`<run>/evaluation/ep_2500/<unique-eval-id>/`; results, metadata, its evaluation
manifest, and plots never overwrite a prior evaluation. `--smoke` explicitly
marks a non-formal evaluation and may be combined with
`--checkpoint-episode N` for lifecycle checks.

`comparison_experiment.py` remains available for manifest-driven evaluation,
design-dataset collection, aggregation, and exact-resume workflows.

## Formal protocol

- 2,500 training episodes for the centrally configured seed `20260817`
- inclusive RoI count range 2 through 8
- 60 seconds per episode
- four 0.25-second routing slots per movement interval
- 100 evaluation episodes
- formal model checkpoint `ep_2500`
- TD3 noise, DDQN epsilon, and DDQN logits noise disabled in evaluation
- no network, optimizer, target-network, replay, or Dinkelbach update in evaluation
- direct-ratio methods store zero objective reward on every non-terminal
  movement transition and the single episode value
  `sum(timely delivered Mbit) / sum(all-UAV mobility energy J)` on the terminal
  transition; task-potential shaping remains transition-local
- Dinkelbach lambda starts at `0.0` and is fixed within each non-overlapping
  50-episode outer block
- after every complete block, lambda becomes
  `sum(timely delivered Mbit) / sum(all-UAV mobility energy J)`; no scaling,
  clipping, or moving average is applied (2,500 episodes contain 50 complete
  block updates)
- model-only checkpoint every 50 episodes, including exactly one final checkpoint
- full-resume checkpoint every 50 episodes, retaining only the latest two

Run one method and the configured training seed per training/evaluation job.
Evaluation episodes are averaged within the seed; additional explicitly run
seeds may be aggregated as independent trained-policy seed means
independent policies. The cross-seed report uses sample standard deviation and
the Student-t 95% interval `mean +/- t(0.975, n-1) * s / sqrt(n)`. For five
seeds, the critical value is approximately `2.776`.

## Scenario manifests

The `uav-hrl-scenario-v2` JSON schema records the split, manifest seed, episode
count, generation profile, generator/config fingerprint, content hash, and one
entry per scenario. Schema v1 is obsolete and must be regenerated. Each entry
contains its profile-aware ID and seed, GT/RoI data, UAV initial state, SR
initial state and deterministic motion primitive, and traffic/load primitives.

Manifest generation uses local `random.Random` and
`numpy.random.Generator` instances. It does not consume global RNG state.
Train, validation, and test IDs/seeds are disjoint when generated with their
respective split names. Mixed and fixed-`num_GT` profiles also derive distinct
scenario IDs/seeds when the split and manifest seed are identical. JSON load
validates both the canonical content hash and the current environment
fingerprint.

Without `--num-gt`, a mixed manifest draws `num_GT` in the inclusive range
2–9. With `--num-gt N`, every episode uses the same supported value. Use one
fixed manifest across every compared method and trained seed for a Figure whose
x-axis is RoI/GT count.

The randomness audit separates:

- Exogenous scenario inputs: GT count/placement, UAV initial altitude/state, SR
  initial state/motion primitive, and traffic/load primitives.
- Agent/training randomness: network initialization, exploration, replay
  sampling/minibatch order, target-policy smoothing, and PyTorch/CUDA RNG.
- Policy-dependent outcomes, which are never stored in a manifest: task
  assignment, actions, coverage, discovery, FOV validity, sources, packets,
  routing decisions, and delivered bits.

Traffic packet creation remains gated by the policy-dependent task assignment
and FOV state; the manifest stores only the underlying demand primitive.

## Commands

Generate separate manifests (runtime artifacts should remain outside Git):

```powershell
python -X utf8 comparison_experiment.py generate-manifest --split train --manifest-seed 101 --episodes 2500 --manifest runs/comparison/manifests/train.json
python -X utf8 comparison_experiment.py generate-manifest --split validation --manifest-seed 202 --episodes 100 --manifest runs/comparison/manifests/validation.json
python -X utf8 comparison_experiment.py generate-manifest --split test --manifest-seed 303 --episodes 100 --manifest runs/comparison/manifests/test.json
python -X utf8 comparison_experiment.py generate-manifest --split test --manifest-seed 303 --episodes 100 --num-gt 4 --manifest runs/comparison/manifests/test-num-gt-4.json
```

Run the 60-second manifest smoke:

```powershell
python -X utf8 comparison_experiment.py smoke --training-seed 1 --output-dir runs/comparison/smoke
```

Collect deterministic centralized joint transitions for offline LLM design
analysis with the explicit, evaluation-only collector. This command currently
supports only `td3_dinkelbach`; every other controlled method is rejected during
preflight before checkpoint loading, Simulator construction, or output creation:

```powershell
python -X utf8 comparison_experiment.py collect-design-dataset --method td3_dinkelbach --split validation --manifest runs/comparison/manifests/validation.json --training-seed 20260817 --episodes 100 --checkpoint runs/comparison/td3_dinkelbach/train/<train-hash-8>/seed-20260817/checkpoints/models/ep_2500 --output-dir runs/design --reference-per-episode runs/comparison/td3_dinkelbach/evaluate/validation/<validation-hash-8>/seed-20260817/per_episode.csv
```

The collector writes to an isolated
`<root>/<method>/design/<split>/<manifest-hash-8>/seed-<seed>/` directory. It
uses the ordinary deterministic evaluation dataflow with no exploration or
action perturbation and never updates learning or Dinkelbach state. Its atomic
artifacts are `design_transitions.npz`, `design_dataset_metadata.json`, and a
design-local `per_episode.csv`/`per_episode.jsonl`. The NPZ stores each complete
532-D state, projected 48-D raw actor action, 532-D next state, terminal flags,
delivery/energy and potential reward components, checkpoint lambda, and
episode/step/scenario identity. Metadata records the authoritative state/action
schema from `centralized_movement.py`; it deliberately does not select
Lipschitz hyperparameters or compute a Lipschitz constant.

`--output-dir` is an output root. Train/evaluate commands derive collision-safe
run directories from method, manifest, split, and seed:

```text
<root>/<method>/train/<training-manifest-hash-8>/seed-<seed>/
<root>/<method>/evaluate/<split>/<evaluation-manifest-hash-8>/seed-<seed>/
```

An existing non-empty run directory is never overwritten. Training may reuse it
only with `--resume` pointing to a compatible checkpoint inside that exact run.
Evaluation reruns must use a different identity/root or explicitly archive the
old runtime output first.

Train and evaluate commands complete a read-only preflight before creating the
canonical run directory. The preflight validates the method, manifest
schema/hash/split/count, requested episode count, formal configuration, COM
calibration, canonical identity, and applicable checkpoint metadata. Evaluation
checks the model-only checkpoint metadata before loading weights. A failed
preflight creates no run directory, identity marker, history, or checkpoint and
does not initialize the simulator.

After preflight, `run_status.json` records atomic lifecycle transitions. Fresh
runs move through `PREPARING`, `RUNNING`, and `COMPLETED`. An execution exception
records `FAILED` with concise exception metadata; a fresh rerun remains blocked.
A matching exact resume of an interrupted/failed training run records
`RESUMING`, then `RUNNING` and `COMPLETED`. Completed evaluation directories
remain collision protected.

Train the formal seed and evaluate its episode-2500 model-only checkpoint:

```powershell
python -X utf8 comparison_experiment.py train --manifest runs/comparison/manifests/train.json --training-seed 20260817 --episodes 2500 --output-dir runs/comparison
python -X utf8 comparison_experiment.py evaluate --split test --manifest runs/comparison/manifests/test.json --training-seed 20260817 --episodes 100 --checkpoint runs/comparison/<method>/train/<train-hash-8>/seed-20260817/checkpoints/models/ep_2500 --output-dir runs/comparison
```

Resume uses a retained full checkpoint from the same canonical run:

```powershell
python -X utf8 comparison_experiment.py train --manifest runs/comparison/manifests/train.json --training-seed 20260817 --episodes 2500 --resume runs/comparison/<method>/train/<train-hash-8>/seed-20260817/checkpoints/full/ep_2450 --output-dir runs/comparison
```

Exact resume must select the latest valid full-resume checkpoint in the
canonical run. Requesting an older checkpoint is rejected if a newer valid full
checkpoint exists. If no newer valid full checkpoint exists, valid model-only
directories newer than the resume boundary can be artifacts of a previously
interrupted full save. They are moved, never deleted, to
`recovery/resume-from-ep_NNNN-<transaction-id>/models/`; the accompanying
`recovery_manifest.json` records source/destination paths, checkpoint metadata,
resume provenance, timestamp, and transaction ID. Reconciliation scans only
direct canonical `checkpoints/models/ep_N` children. Hidden, temporary,
incomplete, invalid, other-run, other-seed, history, and full-resume artifacts
are not quarantined.

Aggregate all seed directories below an evaluation root:

```powershell
python -X utf8 comparison_experiment.py aggregate --input-dir runs/comparison/evaluation --output-dir runs/comparison/aggregate
```

Exact-resume checkpoints validate the method fingerprint, training-manifest
hash, training seed, the complete Dinkelbach block state, and its configuration.
Checkpoint schema v3 stores the terminal direct-ratio objective in joint replay.
Schema-v2 Dinkelbach TD3/DDPG checkpoints remain loadable; schema-v2 ratio
checkpoints are explicitly rejected because their per-transition `B/E` values
cannot be migrated without episode-boundary information. A partial Dinkelbach
block is persisted exactly and resumes with
the same lambda, completed-episode count, numerator sum, denominator sum, block
index, input-validity status, and successful-update count. An incomplete final
block never triggers a forced update. Full-resume logging schema v1 separately
persists `lambda_used_log` and `lambda_after_episode_log`; a legacy ambiguous
single-lambda log is rejected for exact resume without changing model-only
checkpoint compatibility. Formal evaluation validates model-only type, schema,
532/48/126 dimensions, movement-agent/DDQN gamma, COM calibration, method/seed,
the formal core configuration, and exactly 2,500 completed training episodes before
loading weights. A distinct validation/test manifest is expected; output
metadata records both training/evaluation manifest hashes and checkpoint
provenance, including the fixed Dinkelbach configuration and state. Evaluation
does not mutate that state.

Checkpoint directories are written through a same-parent temporary directory
and atomically renamed only after every file succeeds. A 2,500-episode run has
50 model-only checkpoints (`ep_0050` through `ep_2500`). Full-resume saves use
the same 50-episode schedule but retain only the latest two directories. If a
custom run ends off schedule—for example at episode 75—it saves episodes 50 and
75 exactly once. Retention never removes model-only checkpoints or files outside
that run's `checkpoints/full` directory. Formal evaluation always uses
`ep_2500`; test performance is not used to select a checkpoint.

## Metrics and artifacts

Every evaluation episode writes method/seed/scenario identity plus:

- timely goodput and raw final-hop throughput in Mbit
- total mobility energy and `timely_goodput_mbits / mobility_energy_j`
- FOV/COM timely deliveries and deadline violations
- total violations, coverage, and found-GT ratio
- routing Waits, partial transmissions, and slot-budget violations

Zero or invalid energy produces an EE value of `0.0`, never NaN or infinity.
Each evaluation directory contains `per_episode.csv`, `per_episode.jsonl`,
`per_training_seed_summary.csv`, `per_training_seed_summary.json`, and
`run_metadata.json`. Formal aggregation defaults to exactly five seeds and 100
rows per seed, requires identical scenario sets and compatible identities, and
rejects duplicate reruns or non-finite values. For smaller deterministic tests,
override `--expected-seed-count` and `--expected-episodes-per-seed` explicitly.
Aggregation writes `cross_seed_summary.csv`, JSON, and Student-t methodology in
`aggregation_metadata.json`.

Every canonical training run writes canonical `training_history.jsonl`, its
derived tabular projection `training_history.csv`, and
`training_history_commit.json`. They contain one identical row per completed
episode with method/seed/training-manifest identity, finite reward, timely
goodput, mobility energy, and per-episode EE. Dinkelbach methods also record the
finite lambda used during that episode, the lambda after the episode, whether
an update occurred, update status, block index/position, and the block
numerator/denominator sums so far. Direct-ratio methods store those inapplicable
Dinkelbach fields as null rather than a fake lambda of zero. Boundary rows
record the completed block's sums and resulting lambda; the following episode
uses that new value. Invalid or non-finite block input, or a non-positive
denominator, records an explicit status and preserves the old finite lambda
while still completing the block. The commit file records identity, row count,
last episode, both SHA-256 hashes, and a transaction ID. JSONL and CSV are
replaced from one normalized row set, and
the commit marker is replaced last; readers reject partial or hash-inconsistent
transactions. The full checkpoint embeds the canonical rows. Exact resume can
repair an interrupted dual-format transaction from those checkpoint rows,
validates any committed prefix, truncates to the checkpoint boundary, and then
continues without duplicate episodes. A fresh run never silently accepts
partial history. `run_metadata.json` records the history files, canonical
format, row count, last episode, and identity.

The legacy `HRL_task_aware.py --mode smoke` and explicit
`--mode train --episodes N` interfaces remain available. Its optional legacy
training CSV names the corresponding columns `lambda_used` and
`lambda_after_episode`; it no longer emits an ambiguous `lambda` column.
