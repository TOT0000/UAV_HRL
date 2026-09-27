# Enriched movement replay and independent sampling

The active auxiliary schema is `uav-hrl-joint-replay-aux-v1`. It extends
`joint_replay.npz` without changing the actor/critic state, action, reward, or
batch interface. Every auxiliary field has a numeric dtype and fixed shape, so
the archive is readable with `np.load(path, allow_pickle=False)`.

Each transition has a `current_` snapshot aligned with `state` immediately
before the movement decision, and a `next_` snapshot aligned with `next_state`
after the four routing subslots and the next movement-boundary reassignment (if
one occurs). Terminal `next_` snapshots describe the physical terminal boundary.
Arrays are copied into replay storage at the same circular-buffer index.

The complete field/shape/dtype/unit/mask dictionary is emitted under
`replay_auxiliary` in sampling `metadata.json`. Main groups are:

- SR FIFO summaries padded to eight rows. Only an SR whose RoI association is
  already discovered is observable. Deadlines are the packet's original
  absolute end-to-end deadline minus snapshot time.
- Per-UAV aggregate FIFO summaries, including unclipped remaining bits, typed
  counts/bits, the actual aggregate FIFO HOL packet, and typed minimum deadlines.
- Up to one assigned VS pair per UAV, with raw `I`, coverage, `h`, `d`, C9 margin
  `b1*h-d`, canonical `d_L`/`d_R`, and C10 margin
  `min(d_L,d_R)-roi_radius`. Positive margins satisfy the corresponding
  constraint. Geometry and capture validity have separate masks.
- Discovered RoIs, observable SRs, and up to two typed task assignments per UAV.
  Undiscovered RoI positions and SR-to-RoI mappings are never serialized.
- Decision-time U2U, U2G, and observable S2U distance, inclusive range status,
  and expected reference capacity. These are explicitly not scheduling or
  realized-service fields. Recording reads existing large-scale/channel state
  and consumes no RNG.
- Episode, movement step, global transition id, scenario index/stable id hash,
  snapshot times, and the Dinkelbach lambda used by the transition.

Missing auxiliary fields in an older full-resume replay remain invalid after
load; reusing an already allocated buffer first resets every auxiliary value to
its defined missing state, including `-1` IDs and false validity masks.
Zero-filled values are never marked as observations. New samples written after
resume use the active schema. Model-only checkpoints do not need replay fields,
so otherwise-compatible older models remain valid sampling sources. Snapshot
capture reads absent SR queues without inserting keys into the queue mapping.

At 50,000 transitions, the auxiliary fields require approximately 557 MiB of
uncompressed RAM (11,679 bytes per transition). The fixed-shape fields use
`float32`, `int32`/`int16`/`int8`, and Boolean arrays; fixed configuration is
stored once in metadata. NPZ output uses compression. A normal 100-by-60-second
sampling run preallocates 6,000 transitions, or about 66.8 MiB of auxiliary RAM.

## Independent sampling CLI

`run_llm_sampling.py` loads one model-only TD3 + Safe-DDQN checkpoint, starts an
empty collector, disables all optimizer/target/lambda updates, disables routing
exploration, and applies fixed Gaussian movement noise. Formal evaluation still
uses zero movement noise. Output defaults to a unique directory below
`results/llm_samples/` and contains `joint_replay.npz`, `metadata.json`,
`summary.json`, and `scenario_manifest.json`.

```powershell
python run_llm_sampling.py --checkpoint <CHECKPOINT_DIR> --episodes 2
python run_llm_sampling.py --checkpoint <CHECKPOINT_DIR> --episodes 100
python run_llm_sampling.py --checkpoint <CHECKPOINT_DIR> --episodes 37 --noise-std 0.05 --sampling-seed 20260927
```

To reuse exactly the same scenarios for multiple checkpoints, first save the
generated manifest, then pass it to later runs with the same episode count:

Generated balanced manifests encode both the balanced generation mode and each
forced RoI count in every scenario ID. Reusing the saved manifest preserves
those IDs exactly; legacy manifests remain loadable and are never rewritten.

```powershell
python run_llm_sampling.py --checkpoint <CHECKPOINT_250> --episodes 100 --manifest-output shared_sampling_manifest.json
python run_llm_sampling.py --checkpoint <CHECKPOINT_750> --episodes 100 --manifest shared_sampling_manifest.json
python run_llm_sampling.py --checkpoint <CHECKPOINT_1500> --episodes 100 --manifest shared_sampling_manifest.json
```

The episode counts and checkpoint episodes are not fixed by the program.

## Fixed offline baseline and empirical Lipschitz estimate

`run_llm_baseline.py` reads completed independent-sampling directories. It
validates the replay/auxiliary schemas, 531-D movement-state contract,
environment contract, manifest identities, monotonic replay order, finite
numeric fields, and current-snapshot validity before selecting any row. Legal
empty queues, undiscovered RoIs, and an all-false movement mask remain valid.

The fixed sampler selects the same number of rows from every source. Within a
source it uses seeded stable water-filling over the scenario's actual total RoI
count, then the current-time early/middle/late segment, then episode. Rows are
drawn without replacement. Any capacity-driven quota redistribution and all
excluded incomplete records are recorded in `fixed_samples.json`. Source order
on the command line does not affect the fixed result because sources are sorted
by their content-derived identifier.

This command uses the three current 250/750/1500 sampling outputs, selects the
default 1000 transitions from each, tests the documented five lambdas, and uses
128-row distance blocks:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_baseline.py `
  --source results/llm_samples/td3d-ep250-20260927T105459Z-83c75c4c `
  --source results/llm_samples/td3d-ep750-20260927T113004Z-4386d2df `
  --source results/llm_samples/td3d-ep1500-20260927T120552Z-2a10fddc
```

Sampling and numeric settings can be overridden without embedding checkpoint
numbers or source counts in the program:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_baseline.py `
  --source <SAMPLE_DIR_A> --source <SAMPLE_DIR_B> `
  --samples-per-source 750 --seed 17 --batch-size 64 `
  --lambdas 0 0.0001 0.0002 `
  --output-dir C:\absolute\new\baseline-output
```

An explicit `--output-dir` must not exist. Without it, every invocation creates
a collision-safe directory below `results/llm_baselines/` (or
`--output-root`). A new run contains:

- `fixed_samples.npz`: fixed original states, current auxiliary snapshots,
  reward components, masks, and trace identifiers;
- `fixed_samples.json`: source/file hashes, ordered row trace, sampling and
  stratum distributions, compatibility contract, bundle hash, and fixed pair
  contract;
- `baseline_report.json`: reward formula/units, numeric settings, empirical
  Lipschitz estimates, maximum-ratio pair diagnostics, distance/reward
  distributions, and zero/near-zero-distance diagnostics.

For every lambda, the float64 reward is reconstructed as
`B_timely_Mbit - lambda_Mbit_per_J * E_movement_J - P_C9 - P_C10 - P_COM`.
The sampled checkpoint's stored lambda is trace-only. Distances are direct
float64 Euclidean differences over the original current state. All unique
`i < j` pairs are used; only pairs with original-state distance greater than
`1e-8` enter the primary maximum. This is explicitly a finite-sample empirical
estimate, not a global Lipschitz constant.

Reloading verifies the saved NPZ SHA-256 and canonical array-content hash and
does not resample. It also reuses the saved lambda list, distance threshold, and
reward tolerance. Optional repeated `--source` arguments additionally verify
that every original source still has exactly the saved hashes. `--batch-size`
may change because it does not change pair identity or numerical definitions:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_llm_baseline.py `
  --fixed-sample results/llm_baselines/<RUN_NAME> --batch-size 64
```

The downstream current-only LM Studio design and approved-artifact contract is
documented in `docs/llm_design_contract.md`.
