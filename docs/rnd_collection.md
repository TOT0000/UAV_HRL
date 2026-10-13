# TD3 + RND exploration collection

`td3_dinkelbach_rnd_collect` is an isolated controlled method. It keeps the
same environment, K-KM assignment, centralized TD3 action domain, Safe-DDQN
routing, task reward, Dinkelbach update, energy accounting, and formal metrics
as `td3_dinkelbach`. It does not invoke an LLM and does not change any LLM
prompt. Only the TD3 learner receives the RND bonus.

## RND definition and ordering

RND consumes the actual centralized TD3 state dimension reported by the
current movement interface. A float64, per-dimension running mean and variance
normalizes each observed state, followed by clipping to `[-5, 5]`. Both
networks use the same frozen normalization snapshot for a calculation. The
target MLP is `state_dim -> 128 -> 128 -> 64`; the predictor is
`state_dim -> 128 -> 128 -> 128 -> 64`. Hidden activations are ReLU and the
outputs are linear. The target is fixed. The predictor uses Adam with a
default learning rate of `1e-4`. Network initialization uses a forked,
explicit torch seed and therefore does not advance baseline model RNG state.

For the true post-movement `next_state`, including the terminal physical
state, the raw error is the mean squared error over the 64 output dimensions.
Chronological observations update

`q_t = gamma_rnd * q_(t-1) + e_t`

with `gamma_rnd=0.99`. The normalized bonus for that observation uses the
historical q variance before q_t is inserted. Until two q observations exist,
the documented finite initialization variance is `1.0`; this prevents an
uninitialized near-zero denominator. The q filter continues across ordinary
episode boundaries, but starts fresh for a new run and is restored on resume.
It is only a normalization statistic: TD3 gamma, done handling, target
bootstrap, Dinkelbach lambda, EE, and routing remain unchanged.

The production warmup supplies observation and q statistics, but does not
train the predictor. A RND run rejects fewer than two warmup observations.
After warmup, every newly observed state enters the predictor queue at most
once. On each movement transition the order is:

1. calculate the online diagnostic with the pre-update predictor;
2. add the original reward components and true next state to replay;
3. sample replay and reconstruct the original reward at the current lambda;
4. recompute the sampled next-state bonus with the current RND networks and a
   frozen normalization snapshot under `no_grad`;
5. optimize TD3 with `r_original + eta * r_rnd` (`eta=0.5` by default);
6. consume a complete once-only predictor batch.

Shuffled replay never updates observation statistics, the q accumulator, or q
statistics. The replay's original components are never overwritten. Routing
does not receive RND. The default `eta=0.5` is an adaptation starting point,
not a claim that this TD3/MLP implementation reproduces the original PPO/CNN
RND experiment or guarantees broader or better samples.

## Training and exact resume

The default command trains 1500 episodes and creates model/full checkpoints at
100, 250, 500, 750, 1000, and 1500:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_experiment.py td3_dinkelbach_rnd_collect
```

Configuration remains explicit:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_experiment.py td3_dinkelbach_rnd_collect `
  --episodes 1500 --seed 20260817 `
  --checkpoint-episodes 100 250 500 750 1000 1500 `
  --rnd-eta 0.5 --rnd-gamma 0.99 --rnd-learning-rate 0.0001 `
  --rnd-predictor-batch-size 64
```

An existing train manifest can be selected with `--training-manifest <path>`;
its split, episode count, and manifest seed must match `--episodes` and
`--seed`. Sampling accepts its independent shared scenario path through
`run_rnd_sampling.py --manifest <path>`.

Resume selects the newest valid full checkpoint using the existing atomic
reconciliation flow:

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_experiment.py resume `
  'results\td3_dinkelbach_rnd_collect\<run-id>'
```

Full checkpoints add the target/predictor, predictor optimizer, float64
observation and q moments, q accumulator, update counters, configuration, and
the unconsumed predictor batch. Model-only checkpoints contain inference and
frozen-diagnostic RND state but intentionally omit optimizer/pending-batch
resume state. Missing, wrong-shape, or configuration-incompatible RND payloads
fail explicitly. Non-RND checkpoint payloads and loaders are unchanged.

Episode metrics record original and combined reward plus mean/max raw error,
normalized bonus, weighted bonus, predictor loss, and actor/critic/predictor
update counts in `rnd_training_episode_metrics.jsonl`. A checkpoint taken during warmup therefore reports zero learner
updates and is not presented as a learned exploration policy.

## Multi-checkpoint collection and restart

The collector generates one balanced 2--8 RoI test manifest and reuses it for
every requested checkpoint. It applies the existing evaluation policy with
zero extra action noise. Actor, critics, routing networks, RND networks,
normalization statistics, q accumulator, lambda, and optimizers remain frozen.

```powershell
& 'C:\Users\user\anaconda3\envs\LLM_HRL\python.exe' run_rnd_sampling.py `
  --run-directory 'results\td3_dinkelbach_rnd_collect\<run-id>' `
  --checkpoint-episodes 100 250 500 750 1000 1500 `
  --episodes 100 --sampling-seed 20260927
```

Pass `--manifest <path>` to use an existing test/validation manifest, and
`--output-root <short-path>` when a shorter Windows path is useful. Re-running
the same command is the resume operation: each completed checkpoint directory
is skipped only after its manifest identity, episode/transition counts, NPZ
shape, and SHA-256 hashes pass validation. Incomplete or invalid checkpoint
work is rebuilt in its deterministic batch directory.

Each checkpoint output contains the backward-compatible enriched
`joint_replay.npz`, `rnd_diagnostics.npz`, `episodes.json`, and `metadata.json`.
The RND sidecar aligns row-for-row with replay and stores raw error, normalized
bonus, weighted bonus, original reward, diagnostic mixed reward, done,
truncation, episode, step, and scenario index. Replay retains current/next
state, action, original reward components, current/next auxiliary snapshots,
masks, and stable trace IDs. Sampling bonuses are explicitly labeled
`checkpoint_frozen_diagnostic_not_applied_to_actions`; they are not confused
with online training-time bonuses.

The defaults represent 100 complete 60-step episodes per checkpoint (600
episodes, normally about 36,000 transitions). The collector records the actual
transition count and does not pad a shortened environment trajectory.
