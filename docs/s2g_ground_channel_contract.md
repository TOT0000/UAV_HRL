# Ground S2G channel contract

`S2G` means one ground SR team uploading directly to the ground station. It is
independent of the aerial S2U/U2U/U2G range and channel contracts.

- Inclusive service range: three-dimensional distance `d <= 100 m`.
- Effective distance: `d_eff = max(d, 1 m)`.
- Path loss: `PL_dB = 38.46 + 40 log10(d_eff / 1 m)`.
- Transmit power: 23 dBm; noise PSD: -169 dBm/Hz.
- Small-scale power gain: `h ~ Exp(mean=1)` (Rayleigh amplitude), sampled on a
  dedicated named RNG stream every 5 ms and held for that block.
- One 0.25 s routing slot contains exactly 50 blocks. Equal FDMA uses the
  actually allocated portion of the common 10 MHz pool. If the active set
  changes inside a block, capacity is recalculated with the same `h`; the
  channel is not redrawn.

The path-loss exponent four and Rayleigh microwave model follow *System-Level
Security Solution for Hybrid D2D Communication in Heterogeneous D2D-Underlaid
Cellular Network*, IEEE Transactions on Wireless Communications (2024),
DOI `10.1109/TWC.2024.3423351`. The 2 GHz one-metre reference loss of
38.46 dB, the 100 m service cutoff, and the shared equal-FDMA scheduler are
project-specific scaling and system choices.

SRs start on a stable-ID ordered 80 m quarter arc around the GS. A direct-range
crossing changes only the physical first-hop route. It never adds/removes the
COM task, clears its UAV binding, or queues a matching event.

## Reward-step records

Training and ordinary evaluation can stream reconstructable movement reward
rows without retaining them in memory:

```powershell
python run_experiment.py td3_dinkelbach --episodes 1 --episode-seconds 1 --smoke --record-reward-steps
```

The unique run/evaluation directory contains `reward_step_records.jsonl`.
Recording is off by default and observes already-computed values without using
any RNG or changing actions, replay, updates, or environment events.
