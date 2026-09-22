# Canonical visual sensing change record

## Active deterministic Search/VS contract

The current visual contract is
`single-footprint-full-roi-overlap10-passive-contributors-v5`. Both modes use the
0.0156 m by 0.0235 m image plane. Search uses a 0.0175 m focal length and a
fixed nadir footprint; VS keeps the 0.035 m focal length and the existing
oblique camera aimed at the assigned ROI. The deterministic region/frontier
Search Path Manager owns Search UAV motion instead of TD3/DDPG, targets 120 m,
emits one command per second, and holds it across four 0.25-second routing
slots. Search coverage itself is sampled in every subslot.

Search, COM-only, and Hovering UAVs contribute a nadir Search footprint and
change `visited_bitmap`; FOV and FOV+COM UAVs use VS mode and do not contribute
Search coverage. An undiscovered circular RoI is detected from a single
current footprint when
`area(footprint intersect complete circular RoI) / area(complete circular RoI)`
is at least 10%. The comparison is inclusive and uses `gt.radius`; there is no
temporal accumulation, and empty or invalid effective geometry fails closed.
The mode follows current task types; there is no separate camera action or
camera-state transition.

FOV assignment utility, movement constraints, coverage, image quantity, packet
generation and packet size continue to use `VS_CAMERA` and retain the original
35 mm numerical geometry. Search and VS have no minimum-resolution hard
constraint. Resolution remains an input to the existing VS utility and
packet-size calculations.

All methods whose existing task-potential flag is enabled now use post-action
C9, C10, and assigned COM-range penalties with unit weights. Each task type is
averaged over its assigned pairs and is not normalized by episode length.
`no_task_potential` applies none of these penalties. Training, evaluation and
checkpoint validation use the same resolved configuration. The current
checkpoint schema is **33**; schema 32 uses incompatible reward, assignment,
Safe-DDQN cost, and dual-update semantics and requires retraining.

Search diagnostics schema `uav-hrl-search-diagnostics-v2` retains the legacy
field name `search_uav_ids`, whose actual meaning is all Search coverage
contributor IDs. For each subslot, simultaneous overlap is the sum of footprint
cell counts minus that subslot's multi-UAV union; interval overlap and gross are
the four-subslot sums. Interval union/new/historical-revisit counts remain
unique across the full interval, and per-UAV `footprint_cell_count` is the
four-subslot cell-sample denominator.

Original geometry change base: `feature/centralized-td3`, HEAD
`84531f859ed6ab3291d14091e67ba0c8cce6398d`.
The remote branch was fetched and matched this HEAD before editing. The existing
untracked ZIP and `tmp/` directory were preserved. No experiment results or PDFs
were edited, and no push is part of this change.

The valid-capture follow-up below starts from `85d8032`, fetched and confirmed
equal to `origin/feature/centralized-td3`. The contract descriptions in this
document reflect that follow-up; the original file inventory and validation
results are explicitly labeled as historical.

## Valid-capture follow-up

`PacketEngine.inject_packets` reads `fov_task_geometry` before adding credit.
An invalid result removes that assignment's credit and skips `create_packet`;
there is no packet object, queue entry, generated/eligible count, or physical
bit increment. Current assignments use `FOV:<UAV>:<ROI>:<task>` buffer keys.
At each injection event, keys absent from the current assignment set are
pruned, including when the source set is empty. Switching ROI or task identity
therefore starts at zero; returning to an old identity cannot recover its
discarded credit. `reset_packet_state` clears the whole buffer at episode reset.
COM buffer keys and generation remain unchanged.

Only valid intervals contribute to the existing rate integrator. At 5 packets/s
and 0.25 seconds, continuous-valid cumulative counts are 1, 2, 3, 5. Invalid
intervals discard fractions rather than deferring a burst. Partial coverage is
accepted, and coverage zero alone does not gate generation. Physical size is
`31600*min(I,1)`; positive finite image quantity/size are asserted before credit
changes. Packet fields freeze size, coverage, raw I, ROI ID and task ID, and
delivery uses the original coverage for useful bits even after reassignment.

Assignment feasibility and movement potential were not edited. In an ordinary
outside-range pose, Q=0 while G remains positive and increases on approach;
the pair score stays `0.8*Q+0.2*G`. All 16 registry methods retain the same
training/evaluation/paper-evaluation route through `HRL_task_aware.train`,
`_run_routing_slot` and this scheduler. No comparison-specific gate exists.

Historical visual contract: `nadir-search-oblique-vs-valid-capture-v2`.
FOV generation contract: `assigned-valid-sensing-rate-integrator-capture-snapshot-v3`.
Checkpoint schema: **27**, rejecting schema 26 and older before weight loading.
The shared metadata publishes the gate, credit lifetime and capture snapshot.
Seeds, CRN, manifests and evaluation pairing are unchanged.

Follow-up files:

- Production: `Packet_scheduler_v1.py`, `visual_sensing.py`,
  `experiment_config.py`, `training_checkpoint.py`.
- Documentation: `EXPERIMENTS.md`, `docs/visual_sensing_contract.md`.
- Behavior tests: `tests/test_vs_packet_generation_gate.py`,
  `tests/test_visual_sensing.py`, `tests/test_potentials_and_packets.py`,
  `tests/test_permanent_gateway_useful_goodput.py`.
- Schema expectations only: `tests/test_channel_boundary_alignment.py`,
  `tests/test_contract_alignment_v9.py`, `tests/test_design_dataset.py`,
  `tests/test_gs_progress_routing_contract.py`,
  `tests/test_initial_topology_contract.py`,
  `tests/test_training_checkpoints.py`, `tests/test_uav16_contract.py`.

The formal VS injection call is the only production caller of `create_packet`.
It cannot create a zero-bit VS capture: invalid sensing skips creation, and
inconsistent valid geometry raises. The generic constructor still permits
explicit zero-sized packets for compatibility; it is not an alternate formal
VS generation path. No FIFO rewrite was needed.

Follow-up validation used `anaconda3/envs/LLM_HRL/python.exe`:

- Targeted packet, visual sensing, assignment and movement potential tests:
  **129 passed, 51 subtests passed** in 7.21 seconds.
- Training/model-checkpoint/evaluation round-trip, paper registry and evaluation
  smoke tests: **28 passed, 26 subtests passed** in 9.99 seconds.
- Full suite: `python -u -m pytest tests -q -p no:cacheprovider --tb=short
  --disable-warnings --durations=5`: **637 passed, 373 subtests passed**, with
  228 non-failing warnings, in 359.59 seconds.
- `git diff --check` passed. `rg` found only the shared formal injection call
  and its single VS constructor call, and no obsolete active metadata claiming
  that invalid sensing continues injection. Production episode reset calls
  `reset_packet_state`, which clears all injection buffers.

The original untracked ZIP and `tmp/` are preserved. No PDF or existing
experiment result was changed, and no push was performed.

## Geometry and shared execution paths

`visual_sensing.py` is the single camera configuration and geometry source.
`SEARCH_CAMERA` and `VS_CAMERA` are immutable and share the 0.0156 m by
0.0235 m image plane. Target objects supply ROI radius (default 80 m) and
ground altitude. `CAMERA` remains a compatibility alias for `VS_CAMERA` and is
not used by Search production paths.

- `search_footprint`: a fixed nadir rectangle derived from actual altitude.
  `HRL_task_aware._mark_search_observations` freezes one continuous footprint
  per Search coverage contributor in each 0.25-second subslot and passes it to
  both discovery and bitmap sampling. Discovery uses deterministic analytic
  circle/rectangle intersection divided by the complete circular RoI area with
  an inclusive 10% threshold and no temporal accumulation. Search, COM-only,
  and Hover contribute; FOV and FOV+COM transitions contain no Search footprint.
- `vs_geometry`: target-pointing camera rays project all four sensor corners
  onto the ROI ground plane. Sensor width follows the tilt plane, sensor height
  is cross-track, and nadir uses +x bearing. The resulting polygon rotates with
  bearing. Circle/polygon overlap uses deterministic analytical triangle/sector
  edge integration, with no bitmap, Monte Carlo, or added dependency.
- The returned result contains horizontal distance, relative altitude, b1,
  model-range validity, sensing validity, polygon, area, raw I, coverage, G,
  Q, pair score, and geometry diagnostics. Invalid poses return I=c=Q=0.
  The exact inclusive `d=b1*z_relative` boundary has a horizontal corner ray;
  it is model-range-valid but sensing-invalid. A downward-ray margin of 1e-8
  guards the singularity; the distance boundary tolerance is 1e-9 m.
- `I=ROI_area/footprint_area` uses the same polygon as coverage and remains
  unsaturated. `Q=c*min(I,1)` and `G=min(1,b1*z_relative/(d+1e-12))` determine
  `pair_score=0.8*Q+0.2*G`. Assignment keeps this raw score. Movement C9 uses
  `1-G`; C10 uses the finite along-tilt margins only after C9 is satisfied.

`run_experiment.py`, `comparison_experiment.py`, `paper_evaluation.py` and their
CLI/thin training wrappers all enter `HRL_task_aware.train`. All 16 registered
methods use `Simulator`, `UAVAssigner`, `centralized_movement` and `PacketEngine`.
`Simulator_KM` and `Simulator_Rand` are direct aliases of `Simulator`.
TD3/DDPG, Dinkelbach/ratio, task observation/potential ablations and
safe-DDQN/DQN/random routing therefore share the visual model.

`centralized_movement.fov_task_geometry` resolves the assigned target object;
`fov_task_metrics` is its tuple adapter for observations; packet injection reads
the canonical geometry result directly.
Assignment uses the same geometry result. Eligibility retains valid positions,
altitude and target checks, plus existing orchestration/role/task-count rules;
it never masks a pair merely because `sensing_valid_now` is false. Outside the
sensing range, candidates remain ranked by G. Each task type divides its raw
feasible utilities by that call's feasible maximum; equal positive values map
to one. Random assignment bypasses ranking but shares sensing, movement
constraints, and packet generation after selection.

Evaluation trajectory artifacts export the actual active Search rectangle or
VS polygon. `paper_figures` draws oblique polygons at the ROI ground altitude.

## Packet and checkpoint contracts

`VS_PACKET_MAX_BITS=31600` is an independent traffic-model constant.
`Packet_scheduler_v1.fov_physical_packet_size_bits` applies `31600*min(I,1)`
with the existing finite-value sanitization. The formal injection path first
requires `sensing_valid_now`, then asserts positive finite image quantity and
physical size. Only valid sensing accrues the 5 packets/s accumulator; invalid
intervals clear fractional credit without creating packets or updating counters.
Credit is scoped to UAV/ROI/task and pruned on removal/reassignment; episode
reset clears it. Partial or zero coverage is not another generation gate.
The injection cutoff, generic FIFO, QoS eligibility definition and COM behavior
are unchanged. Each capture freezes physical size, coverage, raw image quantity
and ROI/task identity; the VS QoS denominator excludes invalid intervals;
timely useful VS bits equal timely physical bits times capture coverage.

Checkpoint schema 33 requires the complete visual and deterministic external
Search-controller contract for both full resume and model-only evaluation.
Schema 32 and older are rejected because they encode incompatible learning
contracts (and earlier schemas also encode earlier Search contracts).
Older schemas,
missing visual metadata, or changed camera/coverage/packet/weight/validity
metadata fail before loading weights. All affected methods must be retrained.
Training/evaluation configs and metadata publish the canonical configuration.
FOV EMA lifecycle v7, identified by
`nadir-contributor-footprints-empty-fov-samples-v7`, records nadir footprints
for Search, COM-only, and Hover coverage contributors. Non-contributors (FOV
and FOV+COM) use an empty footprint sample while complete per-UAV checkpoint
records and the existing transition/EMA cadence are preserved.

Scenario generation, manifest schema/content rules, seeds, CRN, pairing,
Dinkelbach/ratio objectives, routing reward and
energy formulas are unchanged. A short smoke episode can have no routing
packets.

## Legacy code audit

Full repository searches found no callers of the four legacy Simulator reward
methods (`calculate_fov_reward`, `calculate_fov_reward_wo_Dinkel`,
`calculate_search_reward`, `calculate_search_reward_wo_Dinkel`); they were
removed. `vs_data_valid`, `VS_COVERAGE_EPS`, the rectangle VS intersection and
the full-coverage discovery formula were removed. Packet-generation tests were
retained and updated.

`Fov_model_phase.FovModel` is retained as a delegating facade. `object.UAV`
uses its nadir dimensions for the existing movement-distance cap. The facade
has no independent camera constants or geometry formula. Its image-quantity
adapter also delegates to `vs_geometry`. The disabled legacy movement method
in `object.py` remains disabled; its camera construction now uses the facade's
canonical defaults. Historical packet-state helper field layouts are retained
and obtain image quantity through `fov_task_metrics`.

## Original geometry change: files

Production and documentation:

- `visual_sensing.py`
- `Fov_model_phase.py`
- `Simulator.py`
- `centralized_movement.py`
- `Task_assignment.py`
- `Packet_scheduler_v1.py`
- `HRL_task_aware.py`
- `object.py`
- `experiment_config.py`
- `training_checkpoint.py`
- `fov_ema_lifecycle.py`
- `paper_evaluation.py`
- `paper_figures.py`
- `EXPERIMENTS.md`
- `docs/visual_sensing_contract.md`

Tests (geometry/behavior assertions, current-schema fixtures, or schema version
expectations affected by this change):

- `tests/test_visual_sensing.py`
- `tests/test_assignment_strategies.py`
- `tests/test_channel_boundary_alignment.py`
- `tests/test_checkpoint_evaluation_provenance.py`
- `tests/test_com_qos_fov_canonical_contracts.py`
- `tests/test_contract_alignment_v9.py`
- `tests/test_design_dataset.py`
- `tests/test_distance_aware_task_potentials.py`
- `tests/test_experiment_preflight.py`
- `tests/test_formal_checkpoint_validation.py`
- `tests/test_gs_progress_routing_contract.py`
- `tests/test_initial_topology_contract.py`
- `tests/test_paper_figures.py`
- `tests/test_paper_method_smoke.py`
- `tests/test_potentials_and_packets.py`
- `tests/test_resume_recovery.py`
- `tests/test_training_checkpoints.py`
- `tests/test_uav16_contract.py`

## Original geometry change: validation

Deterministic geometry tests cover camera ownership, nadir size, inclusive
Search discovery, footprint identity, non-Search exclusion, bearing rotation,
original-model area/quantity equivalence, analytical zero/partial/full overlap,
tangency, singular rays, invalid coordinates, outside-range assignment,
quality/proximity ranking, packet saturation/rate/capture freezing, useful
bits, all 16 method configurations, and incompatible checkpoints.

Validation used the existing `anaconda3/envs/LLM_HRL/python.exe` environment.

- Focused Search/VS, assignment, packet and EMA checks: 61 tests and 46
  subtests passed in the initial targeted run.
- Final geometry and paper-method checkpoint smoke checks:
  66 tests and 4 subtests passed.
- Full suite: `python -u -m pytest tests -q -p no:cacheprovider --tb=short
  --disable-warnings --durations=5` — **623 passed, 373 subtests passed**, with
  228 non-failing warnings, in 354.05 seconds. This includes formal runner,
  evaluation, comparison-method, checkpoint round-trip, packet/COM,
  routing, energy, Search release and reproducibility regression tests.
- `git diff --check` passed. A Python-source `rg` audit found no old camera
  constructor parameters, old pixel-derived packet factor, `vs_data_valid`,
  `VS_COVERAGE_EPS`, dL/dR full-coverage discovery, or rectangle VS intersection.

No known failing test remains. Full 1500-episode retraining and publication
evaluation were not run; all affected checkpoints/results need fresh training
and evaluation under this contract. Earlier experiment artifacts are preserved.
