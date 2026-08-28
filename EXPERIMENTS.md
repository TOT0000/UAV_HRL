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
and COM service tasks; unassigned UAVs fall back to Search. UAV 0 is permanently
excluded from service assignment: it is Search plus GS gateway below the release
threshold and Hovering plus GS gateway after release. UAV 9 keeps the ordinary
reserved-Search lifecycle and may be reassigned after release. FOV raw utility
uses global feasible-pair min/max normalization (equal values map to 0.5), while
COM uses canonical S2U capacity at the candidate's actual 3-D geometry,
computed with `10 MHz / 18` from the sampled one-second A2G LoS/NLoS state and
the corresponding deterministic Rician/Rayleigh expected capacity. It is
divided by the fixed 50 m AGL LoS-Rician expected-capacity reference. This
denominator is independent of candidates and traffic rate. A separate feasibility mask and
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
python -X utf8 run_experiment.py resume results/<method>/<run-id> --target-episodes 3000
python -X utf8 run_experiment.py evaluate results/<method>/<run-id>
```

Without `--target-episodes`, resume retains its exact existing horizon and
behavior. The option is the only supported horizon-extension entry point: its
value must exceed both the current planned horizon and the latest completed
full-resume checkpoint. The registry-selected method, seed, hyperparameters,
warmup, fixed exploration-decay contract, optimizer/replay state, Dinkelbach
state, safe-DDQN constraint state, RNG state, counters, and training-history
prefix are restored unchanged. The new horizon does not rescale either
exploration schedule.

An extension deterministically generates a new immutable training manifest
under `<run>/scenario_manifests/`. Before activation, its first N scenarios
must reproduce the previous N-entry manifest and canonical prefix hash exactly.
`resolved_config.json` switches to that manifest atomically and records the old
and new horizons, both manifest hashes, preserved prefix length, checkpoint,
and timestamp. The original manifest remains untouched. Existing model
checkpoints and evaluation outputs are never rewritten; periodic checkpointing
continues at the configured cadence and always includes the new target episode.

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
  movement/channel/routing substeps per one-second movement interval; each
  routing slot contains exactly fifty 5 ms fading blocks
- physical and effective routing masks recomputed in every routing slot for
  safe-DDQN, controlled DQN, and random routing
- safe-DDQN target violation probability `0.1`, initial `lambda_cost=0`, and
  `eta_c=0.01`; every episode uses one frozen multiplier for both executed and
  target actions, then updates it once with
  `max(0, lambda_cost + eta_c * (violations / eligible_packets - 0.1))`.
  This multiplier uses only packets that received a stable routing transition
  ID and their routing-attributable violations. Pre-routing S2U failures remain
  formal system QoS violations but cannot update the multiplier or replay cost;
  an episode with no routing-credit-eligible packets does not update it
- learned routing uses the common local reward
  `capacity_norm - 0.5 * (transmission_delay_norm + queue_delay_norm)` with
  fixed canonical U2U/U2G reference capacities and no distance/progress bonus.
  For the frozen slot-start aggregate FIFO, transmission delay is the HOL
  remaining bits divided by the actual equal-FDMA/fading slot capacity and
  queue delay is all remaining bits behind that HOL divided by the same
  capacity. Elapsed wall-clock HOL wait is diagnostic only
- production FOV/COM deadlines `1.5 s`/`1.0 s`; completion at the exact deadline
  is timely
- production packet injection cutoff `58.5 s`
- every assigned FOV source runs its existing rate integrator beginning with the
  first normal post-assignment injection event. Generation is not gated by full
  RoI coverage, geometry validity, or image-quality validity, and assignment
  does not create a burst. Every generated FOV packet freezes the authoritative
  `fov_task_metrics` overlap ratio as `capture_coverage_ratio`, clipped to
  `[0,1]`; invalid geometry and non-finite values map to `0.0`. The full physical
  `size_bits` remains the queue, service, partial-hop, and routing-reward load.
  Timely FOV useful bits are `size_bits * capture_coverage_ratio`; timely COM
  useful bits are the full `size_bits`; late packets contribute zero useful bits
- At each one-second Search boundary every FOV-observation participant computes
  raw unvisited, frontier, overlap, and `map_changed` from the same immutable
  pre-commit bitmap. Observation participation is separate from coverage
  contribution: only Search-UAV footprints enter the atomic coverage union.
  The frozen all-UAV samples update EMA after coverage commit and before
  Search-to-Hover conversion, assignment, or state construction. No post-commit
  bitmap fallback is permitted. Routing state
  getters are pure reads. Full checkpoints persist EMA values, initialization,
  previous footprints, footprint/EMA markers, and EMA update count.
  SR routes omit the duplicated start point, include the exact target, and mark
  arrival on the update that consumes the final waypoint. The only mutable SR
  lifecycle fields are `assigned_gt_id` and `arrived`. A COM session produces
  no packets until its assigned UAV first enters the inclusive 400 m S2U
  range; activation is permanent for the episode and packet generation then
  continues at 50 packets/s even while the UAV is out of range. Each activated
  COM packet is immediately `qos_eligible=True` at SR generation, while
  `routing_eligible=False` until S2U completes. S2U completion changes only
  routing eligibility and never increments the QoS denominator again. SR
  timeout or terminal settlement is one formal E2E violation measured from
  generation time; without a routing transition ID it increments the explicit
  unattributed diagnostic and never writes safe-DDQN replay. These lifecycle
  fields remain episode-boundary snapshots in
  full checkpoints
  mid-episode checkpointing is explicitly unsupported.
- 100 evaluation episodes
- formal model checkpoint `ep_1500`
- TD3 noise, DDQN epsilon, and DDQN logits noise disabled in evaluation
- no network, optimizer, target-network, replay, or Dinkelbach update in evaluation
- direct-ratio methods store zero objective reward on every non-terminal
  movement transition and the single episode value
  `sum(timely useful Mbit) / sum(all-UAV mobility energy J)` on the terminal
  transition; task-potential shaping remains transition-local
- Dinkelbach lambda starts at `0.0` and is fixed within each non-overlapping
  50-episode outer block
- after every complete block, lambda becomes
  `sum(timely useful Mbit) / sum(all-UAV mobility energy J)`; no scaling,
  clipping, or moving average is applied (1,500 episodes contain 30 complete
  block updates)
- propulsion energy uses the fixed `canonical-3d-quadrotor-v1` vector-power
  model and actual boundary-projected displacement; communication energy is not
  part of the EE denominator
- model-only checkpoint every 50 episodes, including exactly one final checkpoint
- full-resume checkpoint every 50 episodes, retaining only the latest two

## Movement task-potential contract

Movement shaping remains a potential difference with unit movement discount
and unit task weights: `F_x = beta_x * (Phi_x(s_next) - Phi_x(s))`, where
`beta_search = beta_vs = beta_com = 1` and `gamma = 1`. An unchanged potential
therefore contributes zero, approaching a task target contributes positively,
and retreating contributes negatively. Terminal transitions retain the existing
terminal-zero potential lifecycle. No absolute per-step proximity bonus and no
delivery or connectivity potential is present; routing reward is unchanged.

Search remains exactly the global `mean(visited_bitmap)`. It does not use the
Search target position, a frontier target, or any target-distance term, and the
pre-commit coverage lifecycle is unchanged.

For every assigned FOV task, VS progress is an equal blend of the existing
`coverage_ratio * clip(image_score, 0, 1)` sensing progress and horizontal
target proximity. Horizontal proximity is
`1 - clip(d_xy / sqrt(W^2 + H^2), 0, 1)` and deliberately ignores altitude.
Invalid sensing geometry contributes zero sensing progress while finite UAV
and target coordinates still produce distance progress. `Phi_VS` is the
arithmetic mean over FOV tasks, or zero when none exist.

For every assigned COM task, progress is an equal blend of the existing
deterministic expected S2U capacity normalization and 3-D S2U range-gap
proximity. With inclusive `R_S2U = 400 m`, the gap is
`max(d_3D - R_S2U, 0)` and is normalized by
`max(sqrt(W^2 + H^2 + (z_max - z_ground)^2) - R_S2U, epsilon)`. The distance
term is exactly one throughout the legal S2U range and decreases continuously
only outside it. The prospective fading-aware capacity component deliberately
does not apply the hard service cutoff, so it continues to provide continuous
channel-quality information outside 400 m. VS remains based only on FOV sensing
and horizontal target geometry and is independent of the communication range.
`Phi_COM` is the arithmetic mean over COM tasks, or zero when none exist.
Both blends use weights `0.5/0.5` and are finite in `[0,1]`.

The contract is `vs-horizontal-proximity-com-400m-range-gap-blend-v2`. TD3 and DDPG
consume the same potential definitions, random-movement methods publish the
same environment/reward contract, and `*_no_task_potential` methods disable all
Search/VS/COM shaping. Existing observations already contain UAV and task
positions plus COM capacity, so movement/routing state dimensions remain
429/90 and the joint action dimension remains 30.

## Stochastic channel contract

Every registered method uses the same `Simulator`-owned channel lifecycle.
Equation (13), with `a=11.95`, `b=0.136`, elevation in degrees, and the 2 GHz
A2G carrier, supplies the Bernoulli LoS probability. At episode reset the state
is initialized before task assignment; it is then updated only at explicit
one-second movement boundaries. The sampled state selects conditional
`FSPL + 2 dB` LoS or `FSPL + 20 dB` NLoS loss. Equation (14)'s expected path
loss is retained only as a deterministic reference/diagnostic and is not used
after a state has been sampled. This is a one-second large-scale block-state
Monte Carlo approximation, not a claim that buildings physically change every
second.

U2U links retain the directed sender-absolute-AGL, 3-D-distance, 2.4 GHz paper
path-loss model and are always LoS. U2U and sampled A2G LoS links use normalized
Rician fading with linear `K=10` (`10 dB`); sampled A2G NLoS links use normalized
Rayleigh fading. S2U, U2G, and U2U all use the same hard, inclusive, Euclidean
3-D slot-start range of 400 m. Out-of-range links have zero observable and service
capacity, are absent from the physical mask, and consume no FDMA bandwidth.
The full potential-link fading profile is still generated before the range
mask, preserving policy-independent RNG draw counts. No `C_min`, outage/PER,
HARQ, shadowing, or retransmission model is added.

The range decision is frozen at the start of each 0.25 s routing slot and is
shared by all fifty 5 ms service blocks in that slot. All sixteen registered
methods use this same channel, range mask, COM activation, packet-service, and
evaluation contract. `FOV_COM_PAIR_MAX_DISTANCE_M = 200 m` remains unchanged:
it is a horizontal task-target compatibility rule for pairing FOV and COM, not
a radio range.

Routing decisions remain every 0.25 seconds. Before each decision the private
channel lifecycle vectorizes fifty 5 ms gains for every potential directed U2U,
U2G, and S2U link in fixed type/sender/receiver order, whether or not the link
will be selected. Separate named training/evaluation channel RNG streams make
these common random numbers independent of policy action counts. The routing
agent, task assignment, and movement observation never receive those future
samples: their CSI is the deterministic distributional expectation
`B E_G[log2(1 + mean_snr*G)]`, evaluated with cached order-64 Gauss-Laguerre
quadrature and normalized by link-type physical LoS-Rician reference capacity.

After all routing actions are chosen, active S2U/U2U/U2G links share the common
10 MHz pool by equal FDMA. Noise and capacity are recomputed with the allocated
bandwidth while reusing the slot's cached gains; full-band capacity is never
linearly scaled. Packet service consumes the fifty block budgets in order. A
packet that finishes inside a block is timestamped at the block start plus its
remaining bits divided by that block's capacity. FIFO, start-of-slot
eligibility, next-slot S2U relay causality, partial-hop receiver locking, Wait,
deadlines, and the communication-energy model are unchanged.

The 5 ms duration follows Ghazikor et al., *IEEE Transactions on
Communications* (2026), DOI `10.1109/TCOMM.2025.3650376`, for block-duration
scale only; its interference, queue, threshold, bandwidth, and K-factor models
are not adopted. Five milliseconds is a fixed block-fading approximation, not
an asserted exact coherence time for every link.

Episode initialization and every non-terminal one-second boundary use one
authoritative ordering. Episode objects and geometry are created first; the
canonical slot-0 SR movement then completes before interval-0 A2G state is
sampled exactly once and initial task assignment runs. At a later boundary, the
fourth routing slot and packet service finish, boundary UAV/SR geometry is made
current, and the next interval's complete A2G state is sampled once. The expired
slot profile is cleared by that state transition. Boundary task assignment,
movement potentials, and movement `next_state` are then built from the new
geometry/state. The next loop iteration reuses this prepared state, so every
non-terminal movement replay `next_state`, projection mask, and `phi_*_t1`
exactly equal the next policy observation, mask, and `phi_*_t`.

Every routing decision, including Wait, receives a global monotonic transition
ID. Packets retain the exact ID of their last routing action. Routing
transitions first remain causality-pending instead of recording an end-of-slot
observation. At the next slot, packet injection,
expiration/cleanup, start-of-slot HOL eligibility, effective masks, and routing
observations are resolved in their canonical order. Boundary expiration cost is
attached by packet transition ID, never by the sender's latest replay index.
After the actual next observation is fixed, a transition remains credit-pending
while any active packet still references it and is not sampleable. Delivery,
expiration, or replacement by a newer action releases that reference. Packets
with no routing transition still contribute to system QoS and increment an
explicit missing-ID diagnostic without receiving fabricated replay credit.
System violations therefore equal routing-attributable violations plus
unattributed pre-routing violations. System evaluation uses every activated
COM sample, while the safe-DDQN constraint uses only stable-ID credit-eligible
samples.
Formal routing replay is one-step. Episode-terminal packet outcomes and costs
are settled before the ledger is drained.

Run one method and the configured training seed per training/evaluation job.
Evaluation episodes are averaged within the seed; additional explicitly run
seeds may be aggregated as independent trained-policy seed means
independent policies. The cross-seed report uses sample standard deviation and
the Student-t 95% interval `mean +/- t(0.975, n-1) * s / sqrt(n)`. For five
seeds, the critical value is approximately `2.776`.

## Scenario manifests

The `uav-hrl-scenario-v7` JSON schema records the split, manifest seed, episode
count, generation profile, generator/config fingerprint, content hash, permanent
gateway contract, and one entry per scenario. Schemas v1-v6 are obsolete and
must be regenerated. Each
entry contains its profile-aware ID and seed, GT/RoI data, UAV initial state,
SR initial state and deterministic motion primitive, and traffic/load
primitives.

The ground station is fixed at `(0, 0, 0)`. Canonical UAV 0 starts at
`(50, 50, z)` while UAVs 1-9 retain `(300,250)`, `(500,250)`, `(700,250)`,
`(900,250)`, `(100,750)`, `(300,750)`, `(500,750)`, `(700,750)`, and
`(900,750)` respectively. Every UAV still consumes exactly one deterministic
`z ~ Uniform(80,120)` draw and starts with 10,000 J. This is only an initial
geometry contract. UAV 0 is the permanent GS gateway. The single common movement
proposal/execution path applies a continuous, hard-only 3-D GS projection to
every movement method in training and evaluation. Every airspace-legal proposed
position at distance `<= 400 m` is retained exactly, including the full
`360-400 m` band. Only points beyond 400 m are projected to the inclusive hard
boundary. Feasible UAV altitude is retained and the horizontal component is
shortened to at most `sqrt(400^2 - dz^2)`, avoiding naïve radial scaling below
minimum altitude; the final point must also obey environment XY bounds. The
legacy 360 m soft radius remains metadata-only with
`gs_gateway_soft_radius_operational=false`. UAV 9 is not projected.
Replay records the executed net movement, state/channel/task geometry observes
the projected position, and propulsion energy uses the executed displacement.
The fixed movement warm-up remains 10,000 joint transitions.

Every manifest entry and formal preflight validates the finite 3-D initial
range graph using the common inclusive 400 m Euclidean 3-D boundary for U2G and
U2U.
At least one UAV must have a U2G edge and the GS connected component must
contain at least two UAVs. Validation uses range geometry, never positive
capacity, and runs before run-directory creation or learned-weight loading.
Failure reports the scenario ID, nearest U2G distance, U2G range, and the UAV
IDs in the GS component. The manifest generator and Simulator fallback share
the same canonical XY layout. Training, validation, test, and all 16 methods
therefore receive identical initial-geometry rules, although later movement
may disconnect the component.

Manifest generation uses local `random.Random` and
`numpy.random.Generator` instances. It does not consume global RNG state.
Train, validation, and test IDs/seeds are disjoint when generated with their
respective split names. Mixed and fixed-`num_GT` profiles also derive distinct
scenario IDs/seeds when the split and manifest seed are identical. JSON load
validates both the canonical content hash and the current environment
fingerprint.

Without `--num-gt`, a mixed manifest draws `num_GT` in the inclusive range
2–8. With `--num-gt N`, every episode uses the same supported value. Use one
fixed manifest across every compared method and trained seed for a Figure whose
x-axis is RoI/GT count.

The randomness audit separates:

- Exogenous scenario inputs: GT count/placement, UAV initial altitude/state, SR
  initial state/motion primitive, and traffic/load primitives.
- Agent/training randomness: network initialization, exploration, replay
  sampling/minibatch order, target-policy smoothing, and PyTorch/CUDA RNG.
- Channel randomness: one-second A2G state and 5 ms small-scale fading, with
  independent named training/evaluation streams and fixed potential-link draws.
- Policy-dependent outcomes, which are never stored in a manifest: task
  assignment, actions, coverage, discovery, FOV validity, sources, packets,
  routing decisions, and delivered bits.

Traffic packet creation remains gated by the policy-dependent task assignment.
Once assigned, a FOV source is not gated by FOV validity; the manifest stores
only the underlying demand primitive and no policy-dependent coverage snapshot.

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
relationship, training seed, the complete Dinkelbach block state, and its configuration.
Checkpoint schema v20 is the current continuous-hard-only-gateway,
unified-400-m communication, permanent-gateway, coverage-weighted useful
goodput, boundary-aligned stochastic-channel,
movement/routing replay, utility/QoS, routing-ID causality/credit,
frozen-backlog reward, GS-reachable initial topology, distance-aware VS/COM
task potentials, hard-range/COM-session, atomic-FOV, seed-ratio
aggregation, propulsion, and four-slot/fifty-block movement-channel contract.
Schema v19 and every older schema is rejected before weights or replay state are
restored and must be retrained; no legacy checkpoint migration is attempted.
The current schema
stores the 429/30/90 dimensions, movement feature schema, direct-ratio bit/J
objective, shared gateway/channel/packet contracts, active FOV capture-coverage
snapshots, raw/useful counters, inject buffers, adaptive routing lifecycle, and
FOV-EMA state. The movement feature schema remains unchanged; the scenario
schema is v7. Schema-19 checkpoints use the discontinuous 360-400 m soft
compression and must be retrained under the hard-only projection dynamics.
Schema-18 checkpoints use the previous split 200 m A2G / 400 m
A2A environment and older COM range-gap semantics, so they must be retrained.
Schema-17 checkpoints predate the permanent UAV 0 gateway,
assigned-source FOV generation, and useful-goodput numerator and must be
retrained. Schema-16 checkpoints use the prior sensing-only VS and capacity-only COM
reward semantics and must be retrained. Scenario v3 and schema-15 checkpoints
also remain incompatible with the current canonical initial-topology contract;
none are silently migrated. A partial
Dinkelbach block is persisted
exactly and resumes with
the same lambda, completed-episode count, numerator sum, denominator sum, block
index, input-validity status, and successful-update count. An incomplete final
block never triggers a forced update. Full-resume logging schema v2 separately
persists `lambda_used_log` and `lambda_after_episode_log`; a legacy ambiguous
single-lambda log is rejected for exact resume without changing model-only
checkpoint compatibility. Formal evaluation validates model-only type, schema,
429/30/90 dimensions, movement-agent/DDQN gamma, COM normalization, method/seed,
the formal core configuration before loading weights. The checkpoint's own
planned horizon may be smaller than the current run horizon only when
`total_episodes` is the sole formal-config difference, the checkpoint episode is
within both horizons, and its training-manifest hash equals the canonical prefix
of the current training manifest at the old planned horizon. All non-horizon
method, seed, model, optimizer, warmup, exploration, reward, QoS, energy,
channel, routing, and task contracts remain strict. A distinct validation/test manifest is expected; output
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

- canonical timely useful goodput and raw final-hop throughput in Mbit
- total mobility energy and `total_timely_useful_mbits / mobility_energy_j`
- FOV/COM timely deliveries and deadline violations
- FOV generated raw bits, timely delivered raw bits, timely useful bits, mean
  capture coverage, zero-coverage packet count, timely COM bits, and total timely
  useful bits
- canonical system-QoS eligible count, violation count/probability, isolated
  non-formal SR admission diagnostics, coverage, and found-GT ratio
- routing Waits, partial transmissions, and slot-budget violations

Zero or invalid per-episode energy produces an episode diagnostic EE value of
`0.0`, never NaN or infinity. Canonical aggregation treats a zero pooled
denominator as missing.
Each evaluation directory contains `per_episode.csv`, `per_episode.jsonl`,
`per_training_seed_summary.csv`, `per_training_seed_summary.json`, and
`run_metadata.json`. Formal aggregation defaults to exactly five seeds and 100
rows per seed, requires identical scenario sets and compatible identities, and
rejects duplicate reruns or non-finite values. For smaller deterministic tests,
override `--expected-seed-count` and `--expected-episodes-per-seed` explicitly.
Canonical within-seed aggregation uses numerator/denominator sums for EE,
FOV/COM/ALL violation probability, and FOV/COM delay. Valid seed ratios receive
equal weight across seeds with sample SD and Student-t 95% CI; zero-denominator
seeds are missing. `canonical_aggregation()` in
`evaluation_aggregation.py` is the only statistical implementation. Generic
CSV/JSON canonical rows and paper JSON serialize or adapt the same returned
rows; writers do not recompute means, sample SD, valid-seed counts, or
Student-t intervals. The canonical JSON artifacts contain only those six
formal metrics, including their per-seed numerators, denominators, units,
schema, valid/missing seed counts, and CI provenance. The paper path consumes
only these canonical rows.

Generic `cross_seed_summary.csv` and `.json` use the independent
`uav-hrl-generic-cross-seed-aggregate-v3` artifact schema. They combine the six
`canonical_ratio` rows with 32 `descriptive_seed_mean` diagnostics. Descriptive
statistics first take the episode arithmetic mean within each training seed,
then give each valid seed equal weight with seed-level sample SD and Student-t
95% CI; missing seed values are excluded rather than replaced with zero. The
legacy `delay_violation_probability` row is a `canonical_alias` that directly
copies the canonical `(violation_probability, ALL)` result and records
`alias_of_metric`/`alias_of_task_type`; it is not independently calculated.
Consequently, the current 26-column episode metric registry produces 31 generic
rows per identity: six canonical, 24 descriptive, and one alias. CSV uses the
stable union of all row fields and JSON-serializes list/dict provenance. Method,
split, manifest, training seeds, and per-seed checkpoint fingerprints remain in
both generic formats. `aggregation_metadata.json` records the separate rules.
Existing `per_episode.csv` files can therefore be re-aggregated without
retraining.

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

The single-run paper CLI remains one method per invocation and never trains or
resumes a model. Learned methods use the completed formal `ep_1500` checkpoint by
default; explicit diagnostic selectors may use an existing intermediate
checkpoint from a longer completed run. The pure
random `kkm_random_action_random_routing` baseline must omit `--run-dir`; it
loads no model, creates no fake `models.pt`, and records
`checkpoint_required=false`.

Each production paper sweep point streams its raw packet diagnostics directly
to `packet_outcomes.jsonl`, flushing one complete episode before starting the
next. The in-memory all-episode artifact field remains disabled. Every JSONL
record contains `scenario_id`, the episode packet summary, the original raw
`packet_outcomes`, and
`artifact_schema_version=uav-hrl-packet-outcomes-jsonl-v3`. The collection
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

Fixed-RoI evaluation remains backward compatible: without checkpoint or RoI
selectors it loads `FORMAL_CHECKPOINT_EPISODE` and evaluates RoI 2 through 8.
Explicit selectors enable diagnostic checkpoint-progress evaluation without
changing simulator, frozen-policy, packet, energy, or aggregation semantics:

```powershell
python -X utf8 run_paper_evaluation.py td3_dinkelbach --run-dir results/td3_dinkelbach/<run-id> --suite fixed_roi --checkpoint-episode 500 --roi-count 4
python -X utf8 run_paper_evaluation.py td3_dinkelbach --run-dir results/td3_dinkelbach/<run-id> --suite fixed_roi --checkpoint-episodes 500 1000 1500 --roi-counts 2 4 8
```

The singular and plural forms are mutually exclusive. Selected checkpoints
must exist below `checkpoints/models/ep_NNNN`, match their metadata episode and
method/model contracts, and satisfy checkpoint episode <= checkpoint planned
horizon <= current run horizon. An older planned horizon additionally requires
canonical training-manifest prefix compatibility.
Evaluation and sweep metadata expose the checkpoint/current planned totals,
whether the monotonic extension policy was used, its allowed field list, both
training-manifest hashes, and the prefix-compatibility result.
Only `N == FORMAL_CHECKPOINT_EPISODE` is marked formal; earlier checkpoints are
labelled `diagnostic_checkpoint_progress_evaluation` and still use frozen
weights, disabled exploration, and no replay or optimizer updates.

Use the generic batch CLI for multiple training runs. It infers every method
from each run's resolved config and builds the full dynamic Cartesian product:

```powershell
python -X utf8 run_checkpoint_roi_sweep.py `
  --run-dir results/td3_dinkelbach/<run-id> `
  --run-dir results/ddpg_dinkelbach/<run-id> `
  --checkpoint-episodes 500 1000 1500 `
  --roi-counts 2 4 8 `
  --episodes 100 `
  --episode-seconds 60 `
  --manifest-seed 20260817 `
  --output-root results/checkpoint_roi_sweeps
```

The runner validates all runs, checkpoints, fingerprints, RoIs, and output
paths before starting the first simulation, prints the complete plan and total
point count, then executes sequentially. It generates exactly one canonical
test manifest per selected RoI and reuses that manifest hash and scenario-ID
set across every method and checkpoint at that RoI. Different RoIs intentionally
have different manifests. Each batch directory contains
`checkpoint_roi_sweep_metadata.json`, `checkpoint_roi_sweep_summary.csv`, and
`checkpoint_roi_sweep_summary.json`; point paths include method, training-run
ID, checkpoint episode, and RoI. A failed point is marked failed, later points
remain pending, and already completed point outputs are preserved.

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
Within each training seed, delay is total delivered E2E delay divided by total
delivered packet count. FOV and COM violation rows remain diagnostics. The
formal `ALL` seed row pools the two raw violation counts over the two raw
eligible counts, and Fig. 6 reads only `ALL` while labeling the task whose
deadline is swept. Across seeds, valid seed ratios are equally weighted with a
Student-t interval. A delay with no
delivered packets, or violation probability with no eligible packets, is
`null` with `missing=true`. EE
comparison points use each seed's summed timely useful Mbit divided by summed mobility
joules, followed by the same equal-weight cross-seed rule.

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

The shared canonical formulas live in `evaluation_aggregation.py`;
`paper_metrics.py` adapts and validates these rows. Before figure-specific
filtering, the builder rejects missing,
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
