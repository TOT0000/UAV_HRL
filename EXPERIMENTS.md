# Manifest-driven comparison experiments

`comparison_experiment.py` is the single entry point for reproducible training
and evaluation. This framework version intentionally exposes only the corrected
current method:

```text
current K-KM + centralized TD3 + safe-DDQN + Dinkelbach + no LLM
```

Unsupported method components fail immediately; there is no fallback to DDPG,
DQN, KM/random assignment, fixed lambda, or an LLM agent.

## Formal protocol

- 1,500 training episodes per training seed
- 5 independent training seeds
- 60 seconds per episode
- four 0.25-second routing slots per movement interval
- 100 evaluation episodes per trained seed
- TD3 noise, DDQN epsilon, and DDQN logits noise disabled in evaluation
- no network, optimizer, target-network, replay, or Dinkelbach update in evaluation

Run one method and one training seed per training/evaluation job. Uncertainty is
computed from the five trained-policy seed means. Evaluation episodes are
averaged within each seed first; the 500 episodes are not pooled as 500
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
python -X utf8 comparison_experiment.py generate-manifest --split train --manifest-seed 101 --episodes 1500 --manifest runs/comparison/manifests/train.json
python -X utf8 comparison_experiment.py generate-manifest --split validation --manifest-seed 202 --episodes 100 --manifest runs/comparison/manifests/validation.json
python -X utf8 comparison_experiment.py generate-manifest --split test --manifest-seed 303 --episodes 100 --manifest runs/comparison/manifests/test.json
python -X utf8 comparison_experiment.py generate-manifest --split test --manifest-seed 303 --episodes 100 --num-gt 4 --manifest runs/comparison/manifests/test-num-gt-4.json
```

Run the 60-second manifest smoke:

```powershell
python -X utf8 comparison_experiment.py smoke --training-seed 1 --output-dir runs/comparison/smoke
```

Train one formal seed and evaluate its model-only checkpoint:

```powershell
python -X utf8 comparison_experiment.py train --manifest runs/comparison/manifests/train.json --training-seed 1 --episodes 1500 --output-dir runs/comparison/seed-1
python -X utf8 comparison_experiment.py evaluate --split test --manifest runs/comparison/manifests/test.json --training-seed 1 --episodes 100 --checkpoint runs/comparison/seed-1/checkpoints/models/ep_1500 --output-dir runs/comparison/evaluation/seed-1
```

Aggregate all seed directories below an evaluation root:

```powershell
python -X utf8 comparison_experiment.py aggregate --input-dir runs/comparison/evaluation --output-dir runs/comparison/aggregate
```

Exact-resume checkpoints validate the method fingerprint, training-manifest
hash, and training seed. Formal evaluation validates model-only type, schema,
532/48/126 dimensions, TD3/DDQN gamma, COM calibration, method/seed, the formal
core configuration, and exactly 1,500 completed training episodes before
loading weights. A distinct validation/test manifest is expected; output
metadata records both training/evaluation manifest hashes and checkpoint
provenance.

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

The legacy `HRL_task_aware.py --mode smoke` and explicit
`--mode train --episodes N` interfaces remain available.
