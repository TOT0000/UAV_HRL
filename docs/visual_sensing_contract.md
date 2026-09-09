# Canonical visual sensing change record

Base: `feature/centralized-td3`, HEAD `84531f859ed6ab3291d14091e67ba0c8cce6398d`.
The remote branch was fetched and matched this HEAD before editing. The existing
untracked ZIP and `tmp/` directory were preserved. No experiment results or PDFs
were edited, and no push is part of this change.

## Geometry and shared execution paths

`visual_sensing.py` is the single camera configuration and geometry source.
`CAMERA` is immutable: focal length 0.035 m, width 0.0156 m, height 0.0235 m.
Target objects supply ROI radius (default 80 m) and ground altitude.

- `search_footprint`: fixed nadir rectangle, 44.5714 by 67.1429 m at 100 m.
  `HRL_task_aware._mark_search_observations` freezes one continuous footprint
  per Search contributor and passes it to both discovery and bitmap sampling.
  Discovery tests the ROI center inclusively; it never uses full-ROI coverage.
  Non-Search transitions contain no footprint and zero raw Search samples.
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
  `pair_score=0.8*Q+0.2*G`. VS potential averages every assigned pair.

`run_experiment.py`, `comparison_experiment.py`, `paper_evaluation.py` and their
CLI/thin training wrappers all enter `HRL_task_aware.train`. All 16 registered
methods use `Simulator`, `UAVAssigner`, `centralized_movement` and `PacketEngine`.
`Simulator_KM` and `Simulator_Rand` are direct aliases of `Simulator`.
TD3/DDPG, Dinkelbach/ratio, task observation/potential ablations and
safe-DDQN/DQN/random routing therefore share the visual model.

`centralized_movement.fov_task_geometry` resolves the assigned target object;
`fov_task_metrics` is its tuple adapter for packet generation and observations.
Assignment uses the same geometry result. Eligibility retains valid positions,
altitude and target checks, plus existing orchestration/role/task-count rules;
it never masks a pair merely because `sensing_valid_now` is false. Outside the
sensing range, candidates remain ranked by G. Global min/max normalization and
the equal-value result of 0.5 are preserved. Random assignment bypasses ranking
but shares sensing, potential and packet generation after selection.

Evaluation trajectory artifacts export the actual active Search rectangle or
VS polygon. `paper_figures` draws oblique polygons at the ROI ground altitude.

## Packet and checkpoint contracts

`VS_PACKET_MAX_BITS=31600` is an independent traffic-model constant.
`Packet_scheduler_v1.fov_physical_packet_size_bits` applies `31600*min(I,1)`
with the existing finite-value sanitization. The 5 packets/s accumulator,
injection cutoff, zero-coverage/zero-bit injection, QoS eligibility and COM
behavior are unchanged. Each capture freezes physical size and coverage;
timely useful VS bits equal timely physical bits times capture coverage.

Checkpoint schema 26 requires the complete visual contract configuration and
version for both full resume and model-only evaluation. Schema 25 and older,
missing visual metadata, or changed camera/coverage/packet/weight/validity
metadata fail before loading weights. All affected methods must be retrained.
Training/evaluation configs and metadata publish the canonical configuration.
FOV EMA lifecycle v6 represents non-Search footprints as null while preserving
complete per-UAV checkpoint records and the existing transition/EMA cadence.

Scenario generation, manifest schema/content rules, seeds, CRN, pairing,
Dinkelbach/ratio objectives, Search/COM/Relay potential formulas, routing reward
and energy formulas are unchanged. The narrower Search footprint intentionally
changes discovery timing; a short smoke episode can have no routing packets.

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

## Changed files

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
- `tests/test_relay_task_contract.py`
- `tests/test_resume_recovery.py`
- `tests/test_training_checkpoints.py`
- `tests/test_uav16_contract.py`

## Validation

Deterministic geometry tests cover camera ownership, nadir size, inclusive
Search discovery, footprint identity, non-Search exclusion, bearing rotation,
original-model area/quantity equivalence, analytical zero/partial/full overlap,
tangency, singular rays, invalid coordinates, outside-range assignment,
quality/proximity ranking, packet saturation/rate/capture freezing, useful
bits, all 16 method configurations, and incompatible checkpoints.

Validation used the existing `anaconda3/envs/LLM_HRL/python.exe` environment.

- Focused Search/VS, assignment, packet and EMA checks: 61 tests and 46
  subtests passed in the initial targeted run.
- Final geometry, Relay boundary and paper-method checkpoint smoke checks:
  66 tests and 4 subtests passed.
- Full suite: `python -u -m pytest tests -q -p no:cacheprovider --tb=short
  --disable-warnings --durations=5` — **623 passed, 373 subtests passed**, with
  228 non-failing warnings, in 354.05 seconds. This includes formal runner,
  evaluation, comparison-method, checkpoint round-trip, packet/COM/Relay,
  routing, energy, Search release and reproducibility regression tests.
- `git diff --check` passed. A Python-source `rg` audit found no old camera
  constructor parameters, old pixel-derived packet factor, `vs_data_valid`,
  `VS_COVERAGE_EPS`, dL/dR full-coverage discovery, or rectangle VS intersection.

No known failing test remains. Full 1500-episode retraining and publication
evaluation were not run; all affected checkpoints/results need fresh training
and evaluation under this contract. Earlier experiment artifacts are preserved.
