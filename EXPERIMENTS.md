# Manifest-driven comparison experiments

`run_experiment.py METHOD` runs exactly one controlled trajectory method per
process. The registry keys are `td3_dinkelbach`, `ddpg_dinkelbach`,
`td3_ratio`, `ddpg_ratio`, `random_action`,
`td3_dinkelbach_no_task_potential`,
`ddpg_dinkelbach_no_task_potential`, `td3_dinkelbach_wo_ta`,
`td3_dinkelbach_dqn`, `kkm_random_action_random_routing`,
`km_td3_dinkelbach`, `random_assignment_td3_dinkelbach`,
`km_ddpg_dinkelbach`, `ddpg_dinkelbach_wo_ta`,
`td3_dinkelbach_random_routing`, and `td3_dinkelbach_dqn_wo_ta`. All sixteen methods share
10 UAVs, the common Simulator and synchronous movement flow,
energy/delivery accounting, evaluation, and logging.

The added baselines are orthogonal configurations of that shared flow. The
`wo_ta` method keeps the 429-D/90-D layouts but zeros named task-assignment
observation fields. The controlled DQN method replaces safe-DDQN with a masked
standard DQN. The combined random baseline uses K-KM, the common projected
continuous movement domain, and uniform random routing over each slot's current
effective mask. The assignment baselines use one KM round and one seeded
shuffle-and-pair random round, respectively.

All assignment solvers exclude Search and Hovering. Below 0.99 coverage UAVs
0 and 9 are reserved for Search, while the other UAVs solve only discovered FOV
and COM service tasks; unassigned UAVs fall back to Search. At release, all 10
UAVs are immediately reassigned and only unassigned UAVs Hover. FOV raw utility
uses global feasible-pair min/max normalization (equal values map to 0.5), while
COM uses canonical S2U capacity at the candidate's actual 3-D geometry,
computed with `10 MHz / 18`, divided by the same channel model's fixed
best-feasible 50 m AGL capacity. This denominator is independent of candidates
and traffic rate. A separate feasibility mask and
explicit dummy choices keep rows unmatched when no service task is available;
solver-only infinities never enter the domain utility matrix. K-KM uses at most
two rounds, and FOV+COM is its only legal two-task combination, with current
horizontal targets no more than 200 m apart.

FOV assignment uses the production pair geometry from
`centralized_movement.fov_task_metrics`. Its raw utility is
`coverage * q(I)`, where `q(I)=0` for non-finite or non-positive `I`, `q(I)=I`
for `0<I<=1`, and `q(I)=1/I` for `I>1`. Geometry feasibility remains a separate
mask, so a finite pair with `I>1` stays feasible. No FOV/COM blend, clipping,
bonus, penalty, or all-zero fallback is applied. The seeded random-assignment
baseline bypasses these utilities entirely. Equal dummy candidates are reported
as `dummy_1`, `dummy_2`, ... in deterministic row/ID order.

```powershell
python -X utf8 run_experiment.py td3_dinkelbach
python -X utf8 run_experiment.py ddpg_ratio --smoke
python -X utf8 run_experiment.py td3_dinkelbach_wo_ta --smoke
python -X utf8 run_experiment.py td3_dinkelbach_dqn --smoke
python -X utf8 run_experiment.py kkm_random_action_random_routing --smoke
python -X utf8 run_experiment.py km_td3_dinkelbach --smoke
python -X utf8 run_experiment.py random_assignment_td3_dinkelbach --smoke
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

Evaluation defaults to the formal `ep_1500` checkpoint. Each invocation creates
`<run>/evaluation/ep_1500/<unique-eval-id>/`; results, metadata, its evaluation
manifest, and plots never overwrite a prior evaluation. `--smoke` explicitly
marks a non-formal evaluation and may be combined with
`--checkpoint-episode N` for lifecycle checks.

`comparison_experiment.py` remains available for manifest-driven evaluation,
design-dataset collection, aggregation, and exact-resume workflows.

## Formal protocol

- 1,500 training episodes for the centrally configured seed `20260817`
- inclusive RoI count range 2 through 8
- 60 seconds per episode
- one projected movement command held across four synchronous 0.25-second
  movement/channel/routing substeps per one-second movement interval
- physical and effective routing masks recomputed in every routing slot for
  safe-DDQN, controlled DQN, and random routing
- safe-DDQN target violation probability `0.1`, initial `lambda_cost=0`, and
  `eta_c=0.01`; every episode uses one frozen multiplier for both executed and
  target actions, then updates it once with
  `max(0, lambda_cost + eta_c * (violations / eligible_packets - 0.1))`.
  Eligible packets are generated FOV plus S2U-admitted COM; an episode with no
  eligible packets does not update the multiplier, and SR admission drops are
  excluded from both numerator and denominator
- learned routing uses the common local reward
  `capacity_norm - 0.5 * (transmission_delay_norm + queue_delay_norm)` with
  fixed canonical U2U/U2G reference capacities and no distance/progress bonus
- production FOV/COM deadlines `1.5 s`/`1.0 s`; completion at the exact deadline
  is timely
- production packet injection cutoff `58.5 s`
- Search footprints advance after every accepted physical transition, including
  transitions that add no map cells; FOV EMA advances only on map change/reset.
  Each overlap sample uses an immutable previous/current pair, and routing state
  getters are pure reads. Full checkpoints persist EMA values, initialization,
  previous footprints, footprint/EMA markers, and EMA update count.
  SR routes omit the duplicated start point, include the exact target, and mark
  arrival on the update that consumes the final waypoint. The only mutable SR
  lifecycle fields are `assigned_gt_id` and `arrived`; movement and COM-source
  enablement are read-only derived state, and assigned SRs keep generating COM
  after arrival. These lifecycle fields remain episode-boundary snapshots in
  full checkpoints
  mid-episode checkpointing is explicitly unsupported.
- 100 evaluation episodes
- formal model checkpoint `ep_1500`
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
  clipping, or moving average is applied (1,500 episodes contain 30 complete
  block updates)
- propulsion energy uses the fixed `canonical-3d-quadrotor-v1` vector-power
  model and actual boundary-projected displacement; communication energy is not
  part of the EE denominator
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
python -X utf8 comparison_experiment.py generate-manifest --split train --manifest-seed 101 --episodes 1500 --manifest runs/comparison/manifests/train.json
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
python -X utf8 comparison_experiment.py collect-design-dataset --method td3_dinkelbach --split validation --manifest runs/comparison/manifests/validation.json --training-seed 20260817 --episodes 100 --checkpoint runs/comparison/td3_dinkelbach/train/<train-hash-8>/seed-20260817/checkpoints/models/ep_1500 --output-dir runs/design --reference-per-episode runs/comparison/td3_dinkelbach/evaluate/validation/<validation-hash-8>/seed-20260817/per_episode.csv
```

The collector writes to an isolated
`<root>/<method>/design/<split>/<manifest-hash-8>/seed-<seed>/` directory. It
uses the ordinary deterministic evaluation dataflow with no exploration or
action perturbation and never updates learning or Dinkelbach state. Its atomic
artifacts are `design_transitions.npz`, `design_dataset_metadata.json`, and a
design-local `per_episode.csv`/`per_episode.jsonl`. The NPZ stores each complete
429-D state, projected 30-D raw actor action, 429-D next state, terminal flags,
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
schema/hash/split/count, requested episode count, formal configuration, fixed
COM reference-capacity metadata, canonical identity, and applicable checkpoint
metadata. Evaluation
checks the model-only checkpoint metadata before loading weights. A failed
preflight creates no run directory, identity marker, history, or checkpoint and
does not initialize the simulator.

After preflight, `run_status.json` records atomic lifecycle transitions. Fresh
runs move through `PREPARING`, `RUNNING`, and `COMPLETED`. An execution exception
records `FAILED` with concise exception metadata; a fresh rerun remains blocked.
A matching exact resume of an interrupted/failed training run records
`RESUMING`, then `RUNNING` and `COMPLETED`. Completed evaluation directories
remain collision protected.

Train the formal seed and evaluate its episode-1500 model-only checkpoint:

```powershell
python -X utf8 comparison_experiment.py train --manifest runs/comparison/manifests/train.json --training-seed 20260817 --episodes 1500 --output-dir runs/comparison
python -X utf8 comparison_experiment.py evaluate --split test --manifest runs/comparison/manifests/test.json --training-seed 20260817 --episodes 100 --checkpoint runs/comparison/<method>/train/<train-hash-8>/seed-20260817/checkpoints/models/ep_1500 --output-dir runs/comparison
```

Resume uses a retained full checkpoint from the same canonical run:

```powershell
python -X utf8 comparison_experiment.py train --manifest runs/comparison/manifests/train.json --training-seed 20260817 --episodes 1500 --resume runs/comparison/<method>/train/<train-hash-8>/seed-20260817/checkpoints/full/ep_1450 --output-dir runs/comparison
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
Checkpoint schema v10 is the current utility/QoS/routing-causality,
actual-HOL-wait reward, pooled-QoS aggregation, propulsion, and four-substep
movement-channel contract. Schema v9 and every older
schema is rejected before weights or replay state are restored and must be
retrained; no legacy checkpoint migration is attempted. The current schema
stores the 429/30/90 dimensions, movement feature schema, direct-ratio bit/J
objective, shared channel/packet contracts, adaptive routing lifecycle, and
FOV-EMA state. The movement feature schema remains unchanged, and scenario
manifests remain `uav-hrl-scenario-v3`. A partial Dinkelbach block is persisted
exactly and resumes with
the same lambda, completed-episode count, numerator sum, denominator sum, block
index, input-validity status, and successful-update count. An incomplete final
block never triggers a forced update. Full-resume logging schema v1 separately
persists `lambda_used_log` and `lambda_after_episode_log`; a legacy ambiguous
single-lambda log is rejected for exact resume without changing model-only
checkpoint compatibility. Formal evaluation validates model-only type, schema,
429/30/90 dimensions, movement-agent/DDQN gamma, COM normalization, method/seed,
the formal core configuration, and exactly 1,500 completed training episodes before
loading weights. A distinct validation/test manifest is expected; output
metadata records both training/evaluation manifest hashes and checkpoint
provenance, including the fixed Dinkelbach configuration and state. Evaluation
does not mutate that state.

Checkpoint directories are written through a same-parent temporary directory
and atomically renamed only after every file succeeds. A 1,500-episode run has
30 model-only checkpoints (`ep_0050` through `ep_1500`). Full-resume saves use
the same 50-episode schedule but retain only the latest two directories. If a
custom run ends off schedule—for example at episode 75—it saves episodes 50 and
75 exactly once. Retention never removes model-only checkpoints or files outside
that run's `checkpoints/full` directory. Formal evaluation always uses
`ep_1500`; test performance is not used to select a checkpoint.

## Metrics and artifacts

Every evaluation episode writes method/seed/scenario identity plus:

- timely goodput and raw final-hop throughput in Mbit
- total mobility energy and `timely_goodput_mbits / mobility_energy_j`
- FOV/COM timely deliveries and deadline violations
- canonical eligible count, violation count/probability, SR admission drops,
  coverage, and found-GT ratio
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

Formal training uses `packet_outcome_artifact_mode=disabled` and
`collect_packet_outcomes=false`. It never retains a cross-episode list of raw
packet dictionaries; episode-local outcomes are released after the existing
QoS, timely-goodput, task diagnostic, reward, energy, and training-history
aggregates have been computed. `run_metadata.json` and the resolved config
record the mode, the in-memory collection flag, and the bounded collection
limit. Full-resume and model checkpoints contain no raw packet-outcome list.
The `bounded_memory` mode exists only as an explicit test/debug facility,
requires `collect_packet_outcomes=true`, and accepts at most 16 episodes. It is
rejected for formal or resumed training.

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

## Semantic paper evaluation and figure build

Paper evaluation remains one method per process and never trains or resumes a
model. Learned methods require the completed formal `ep_1500` run. The pure
random `kkm_random_action_random_routing` baseline must omit `--run-dir`; it
loads no model, creates no fake `models.pt`, and records
`checkpoint_required=false`.

Each production paper sweep point streams its raw packet diagnostics directly
to `packet_outcomes.jsonl`, flushing one complete episode before starting the
next. The in-memory all-episode artifact field remains disabled. Every JSONL
record contains `scenario_id`, the episode packet summary, the original raw
`packet_outcomes`, and
`artifact_schema_version=uav-hrl-packet-outcomes-jsonl-v1`. The collection
mode, artifact schema, output path, and written episode count are recorded in
metadata. Writer/open/serialization errors fail the evaluation, and the stream
is closed on both successful and exceptional exits.

The canonical evaluation suites are `training_ee_vs_episode`,
`uav_trajectory_snapshots`, `task_type_delay_vs_arrival_rate`,
`task_type_delay_violation_vs_target_delay`, and `fixed_roi`. Deprecated
`fig*` aliases are accepted only by internal Python compatibility boundaries;
they do not appear in CLI help, output paths, or metadata.

```powershell
python -X utf8 run_paper_evaluation.py td3_dinkelbach --run-dir results/td3_dinkelbach/<run-id> --suite uav_trajectory_snapshots --manifest runs/comparison/manifests/trajectory-test.json --target-uav-id 0
python -X utf8 run_paper_evaluation.py td3_dinkelbach_random_routing --run-dir results/td3_dinkelbach_random_routing/<run-id> --suite task_type_delay_vs_arrival_rate --manifest runs/comparison/manifests/test.json
python -X utf8 run_paper_evaluation.py td3_dinkelbach --run-dir results/td3_dinkelbach/<run-id> --suite task_type_delay_violation_vs_target_delay --manifest runs/comparison/manifests/test.json
python -X utf8 run_paper_evaluation.py td3_dinkelbach --run-dir results/td3_dinkelbach/<run-id> --suite fixed_roi
python -X utf8 run_paper_evaluation.py kkm_random_action_random_routing --suite fixed_roi --manifest-seed 20260817
```

The trajectory manifest contains exactly the requested evaluation episode
count (one by default). Its artifact records all 10 UAVs and paths, explicit
target UAV and phase, RoIs/detection, SR paths, GS, actual selected U2U/U2G
links, sensing footprint geometry, requested/actual times, scenario identity,
method configuration, checkpoint provenance, and Git SHA.

Checkpoint provenance is deliberately split into three fields. The existing
`checkpoint_metadata_fingerprint` remains the SHA-256 of the canonical
checkpoint metadata and retains its previous schema and meaning.
`checkpoint_models_sha256` is a streaming SHA-256 of the exact `models.pt`
bytes after the file has passed a structural PyTorch-ZIP integrity check.
`checkpoint_artifact_fingerprint` is the SHA-256 of this canonical JSON object
(serialized with sorted keys and compact separators):

```json
{
  "schema": "uav-hrl-checkpoint-artifact-v1",
  "checkpoint_metadata_fingerprint": "<metadata sha256>",
  "checkpoint_models_sha256": "<models.pt sha256>"
}
```

The hash path never calls `torch.load()` and reads `models.pt` in bounded
chunks. Missing, empty, corrupt, truncated, or replaced payloads fail closed.
Learned paper evaluations persist all three fields in top-level metadata, each
evaluation point, each point's `run_metadata.json`, and every trajectory
artifact. The figure builder recomputes all three from the selected checkpoint
and requires exact agreement in those layers, the resolved specification, the
method-to-checkpoint map, and final figure metadata. Pure-random methods require
the checkpoint path and all three fields to be null. For schema-v10
checkpoints, evaluation derives the payload and combined hashes from the actual
`models.pt`; schema-v9 checkpoints are rejected under the causality/QoS
contract above. Newly generated paper artifacts without the complete
provenance triplet are intentionally rejected.

Each sweep point writes per-episode data plus `aggregated_plot_data.csv/json`.
Its metadata also persists the actual manifest path and canonical hash,
scenario IDs, evaluation count/horizon/seed, 10-UAV count, and fully resolved
traffic-rate/deadline/cutoff overrides. The deadline-violation (Fig. 6) sweep
uses a scoped `57.0 s` packet-injection cutoff so its maximum `3.0 s` deadline
can resolve within the 60-second horizon; other training and evaluation suites
retain the production `58.5 s` cutoff.
Delay is pooled as total delivered E2E delay divided by total delivered packet
count. FOV and COM violation rows remain diagnostics. The formal `ALL` row pools
the two raw violation counts over the two raw eligible counts, and Fig. 6 reads
only `ALL` while labeling the task whose deadline is swept. A delay with no
delivered packets, or violation probability with no eligible packets, is
`null` with `missing=true`. EE
comparison points use pooled timely Mbit divided by pooled mobility joules.

Every non-trajectory point has exactly the following six canonical aggregate
rows, keyed by `(method_id, point_id, metric, task_type)`:

```text
energy_efficiency_mbit_per_j / null
average_e2e_delay_seconds   / FOV
average_e2e_delay_seconds   / COM
violation_probability       / FOV
violation_probability       / COM
violation_probability       / ALL
```

The shared production helper in `paper_metrics.py` both computes and validates
these rows. Before figure-specific filtering, the builder rejects missing,
duplicate, extra, or semantically invalid rows (including rows unused by the
requested figure). It then requires exact canonical agreement among the
top-level `aggregated_plot_data.json`, the union of all point-level aggregate
files, and a fresh recomputation from every point's `per_episode.jsonl`.
Numerator, denominator, value, unit, and missing status are all checked. Delay
with no delivered packets and violation probability with no eligible packets
remain missing rather than becoming fake zeros; violation numerators may not
exceed their denominators.

The figure spec explicitly maps training and evaluation directories:

```json
{
  "target_uav_id": 0,
  "training_runs": {
    "td3_dinkelbach": {"run_dir": "results/td3_dinkelbach/<run-id>"}
  },
  "evaluation_runs": {
    "uav_trajectory_snapshots": {
      "td3_dinkelbach": {"evaluation_dir": "results/paper_evaluations/uav_trajectory_snapshots/td3_dinkelbach/<evaluation-id>"}
    },
    "fixed_roi": {
      "td3_dinkelbach": {"evaluation_dir": "results/paper_evaluations/fixed_roi/td3_dinkelbach/<evaluation-id>"}
    }
  }
}
```

For `--figure all`, include every required method mapping listed in
`paper_figure_registry.py`. The builder validates exact methods, formal
checkpoint/no-checkpoint provenance, sweep values, units, and common scenario
manifest hashes before rendering. It reloads every manifest and inspects the
actual `ep_1500/metadata.json` plus `models.pt` without loading weights, then
recomputes and compares the canonical checkpoint fingerprint. It never starts
training.

```powershell
python -X utf8 build_paper_figures.py --spec paper_runs.json --figure all
python -X utf8 build_paper_figures.py --spec paper_runs.json --figure training_ee_vs_episode
```

All twelve semantic figures produce PNG, PDF, source CSV, source JSON, and a
resolved semantic specification in a collision-safe
`results/paper_figures/<timestamp>_<git-sha>/` directory. Every formal output
contains exactly one axes. The four trajectory times, three EE comparisons,
and two arrival-task charts are standalone files; deprecated family aliases
expand to those files and never emit a composite. See
`docs/legacy_figure_inventory.md` for Drive file IDs, content fingerprints,
screenshot references, visual contracts, and intentional changes.

Each standalone trajectory JSON uses schema
`uav-hrl-standalone-trajectory-v1` and contains the complete scene needed for
an independent redraw: requested/actual time and phase, scenario/manifest and
checkpoint provenance, Git SHA, GS and ground targets, all 10 UAV snapshots and
their assignments, every UAV path truncated at the selected actual time, SR
snapshots and truncated paths, active links, sensing coverage, camera, axes,
labels, and registry style. Its long-form CSV uses `record_type` values
`uav_path`, `uav_snapshot`, `sr_path`, `sr_snapshot`, `ground_target`,
`ground_station`, `active_link`, and `sensing_coverage`. Rendering that JSON
does not reopen the evaluation artifact; titles use the actual time and phase,
and FOV assignments keep the legacy display label `VS`.
