# Maskable PPO with GAE: Implementation Plan

## Goal

Build a first trainable Balatro policy using PPO, generalized advantage estimation (GAE), and legal-action masks. The milestone is a reproducible agent that completes full runs in Jackdaw and can be evaluated on held-out seeds.

## Current project and Jackdaw interface

`RandomAgent.py` uses Jackdaw's low-level interface:

- `BalatroEnvironment(adapter_factory=DirectAdapter)` runs the in-process Rust game engine.
- `reset()` returns `(obs, mask, info)`.
- `step(action)` returns `(obs, terminated, truncated, mask, info)`.
- `obs.global_context` is a 235-element vector; `obs.entities` contains variable-length card, Joker, consumable, shop, and pack arrays.
- Actions are `FactoredAction` values, not simple integer IDs. `ActionMask` has separate action-type, card, and entity masks plus card selection limits and forced-card information.

The Jackdaw RL guide documents `BalatroGymnasiumEnv`, a Gymnasium wrapper specifically usable with SB3-Contrib `MaskablePPO`. It flattens legal factored actions to `Discrete(500)`, provides a boolean action mask, pads entity arrays into a Dict observation, and accepts reward functions. Use this supported route for the first baseline rather than recreating action enumeration and masks.

## Proposed approach

1. **Environment integration:** Use `BalatroGymnasiumEnv` with `DirectAdapter`, initially with one environment. Keep a small `env_factory.py` for settings, seeds, and reward choice. Add vectorization later using Jackdaw's documented `BalatroVectorEnv`/SB3 adapter when a single-env baseline works.
2. **Policy and observations:** Start with SB3-Contrib `MaskablePPO` over the Gymnasium Dict observation. The guide documents 235 global features and padded arrays for hand cards, Jokers, consumables, shop items, pack cards, and entity counts. Confirm local shapes before training. Encode each candidate action's type and targets in a fixed per-slot matrix, then use a shared context/candidate scorer so the output depends on action meaning as well as game state.
3. **Action masking:** The wrapper enumerates valid `FactoredAction` candidates into a flat action table; `action_masks()` returns a 500-element bool array with `True` for legal actions. Mask-aware prediction must be used during both training and evaluation. Monitor mask cardinality and refused actions. The wrapper can sample card subsets when combinatorial candidates exceed its budget, so record this limitation when interpreting results.
4. **Reward:** The low-level environment does not return reward. The Gymnasium wrapper takes a reward function; Jackdaw documents sparse win/loss, blind-progress, and `ShapedReward`. Start with `blind_progress_reward` for denser feedback, then compare against sparse reward. Keep any shaped weights explicit in config and report win rate, ante reached, and blinds beaten separately from training return.
5. **PPO and GAE:** Let SB3-Contrib manage clipped PPO updates, GAE, masked categorical sampling, and rollout storage. Keep `terminated` and `truncated` semantics intact, especially if vectorizing, so time-limit truncations are handled correctly.
6. **Artifacts:** Save model checkpoints, configuration, reward settings, seed information, library versions, and Jackdaw `ACTION_VERSION`. Use separate training, tuning, and held-out test seed sets.

## Implementation phases

### Phase 0: Verify the local environment contract

- Confirm installed Jackdaw exposes `BalatroGymnasiumEnv`, the documented reward functions, and the expected action version.
- Pin the verified Jackdaw, Gymnasium, and NumPy baseline dependencies. Pin Stable-Baselines3 and SB3-Contrib when introducing the PPO training phase, against the Python version used by the project.
- Run seeded episodes through the existing random agent and Gymnasium wrapper; inspect observation shapes, mask counts, reward, termination/truncation, refusal metrics, and timing.
- Confirm the wrapper's seed behavior and how it works with the local SB3 training API.

**Complete when:** transitions, masks, seeds, and episode endings are understood and reproducible.

**Status:** Jackdaw 0.2.3, Gymnasium 1.4.0, and action version 2 are installed. The direct `RandomAgent` API and Gymnasium wrapper smoke check passed with valid observations/masks; two additional seeded Gymnasium episodes terminated with no refusals. The environment initially could not import because its local NumPy package was incomplete; installing NumPy 2.5.4 into `.venv` resolved the import. See `requirements-phase1.txt` for the verified pins. SB3-Contrib dependencies are deferred until the PPO implementation phase.

### Phase 1: Baseline and reward choice

- Add a thin environment factory and configuration, without duplicating Jackdaw's action adapter.
- Collect random-policy metrics on a fixed seed set: win rate, mean/median ante, blinds beaten, episode length, return, and refused-action count.
- Use blind-progress reward as the recommended first PPO training signal. Run a sparse win/loss baseline comparison where feasible and log reward choice and any shaping weights.

**Complete when:** baseline results are repeatable and training reward is clearly distinguished from evaluation metrics.

**Status:** A 10-episode uniform-random legal-action baseline ran with blind-progress reward and seeds 10000-10009. It had 0/10 wins, mean ante 1.0, mean 0 blinds beaten, mean episode length 8.2 steps, mean return -1.0, 0 truncations, and 0 refused actions. A repeated run produced identical per-episode results. A same-seed sparse-reward comparison had identical episode outcomes and return. Per-episode outputs are in ignored `runs/` directories and can be regenerated with the README command.

### Phase 2: MaskablePPO baseline

- Use Jackdaw's padded observation and action mapping, with a semantic feature row for every candidate action.
- Verify `action_masks()` shape and use SB3-Contrib's mask-aware prediction API in evaluation and inference.
- Track legal-action counts and inspect states with large candidate sets, where the wrapper may subsample actions.
- Add a custom feature extractor only if measured results justify it; keep the default policy for comparison.

**Complete when:** PPO collects real transitions without NaNs, masked actions, or refused actions.

**Status:** `train.py` and `evaluate.py` implement mask-aware PPO training, checkpointing, resume for compatible architectures, held-out evaluation, optional action traces, and HTML replays. Training uses explicit shaped reward weights; evaluation reports blind outcomes separately from training return. The observation includes Chips / Mult / X Mult tags, action history, and semantic candidate-action rows. The actor uses a shared candidate scorer. Action masks prevent back-to-back hand sorts and immediate reversals of the same Joker swap. Older runs showed zero wins and recurring sort/Joker-swap loops; the newer constraints and candidate scorer are intended to address those failure modes. Observation or policy changes require a fresh model; reward-only changes can resume a same-architecture checkpoint.

**Follow-up implementation:** The action-history wrapper masks Rank/Suit Sort immediately after either sort, Joker swaps that revisit an order already seen within the current blind, and all Joker swaps once the configurable per-blind limit (default five) is reached. A candidate-action observation provides a semantic feature row per action slot, and `CandidateActionActorCriticPolicy` uses a shared state/candidate scorer instead of fixed slot logits. Evaluation defaults to seeds 30000-30099, records Joker inventories/acquisitions and per-action-type counts, and reports win rate independently from training return. These observation/policy changes require a fresh model; older checkpoints remain evaluable with `--no-candidate-actions`.

### Phase 3: Training and checkpoint workflow

- Configure rollout length, batch size, PPO epochs, `gamma`, GAE `lambda`, clip range, entropy coefficient, and value coefficient.
- Log policy/value loss, entropy, approximate KL, clip fraction, explained variance, reward, mask cardinality, refusals, and completed-episode metrics.
- Add checkpoint/resume and periodic evaluation on seeds excluded from training.
- If vectorizing, verify autoreset and terminal-observation behavior so truncation bootstrapping remains correct.

**Complete when:** a short run completes cleanly and a saved model can resume consistently.

### Phase 4: Evaluation and tuning

- Compare PPO to the random baseline on the same held-out seeds; report per-seed results and uncertainty for win rate.
- Tune a small number of parameters systematically: learning rate, rollout length, batch size, PPO epochs, `gamma`, GAE `lambda`, clip range, entropy coefficient, and value coefficient.
- Evaluate multiple training seeds and reserve a final test set from tuning.
- Inspect action frequencies and failure cases for reward, truncation, or mask bugs.

**Complete when:** results are reproducible and conclusions are based on independent win/ante outcomes, not only shaped reward.

## Initial hyperparameter starting points

Confirm the wrapper and episode horizon first, and record the resolved library settings.

| Setting | Starting point |
|---|---:|
| Discount `gamma` | 0.99 |
| GAE `lambda` | 0.95 |
| PPO clip range | 0.2 |
| Learning rate | 3e-4, optional annealing |
| PPO epochs per rollout | 4 |
| Minibatch size | 64, subject to rollout size |
| Entropy coefficient | 0.01, tune based on valid-action entropy |
| Value coefficient | 0.5 |
| Max gradient norm | 0.5 |

These are starting values, not known optima.

## Reward and episode semantics

Jackdaw's low-level environment has no reward return; `BalatroGymnasiumEnv` accepts a reward function, and low-level code can use `Transition.from_step(...)`. Start with blind-progress reward and compare against sparse win/loss reward. Treat `ShapedReward` as a separate configurable experiment. Always report win rate, ante reached, and blinds beaten independently of training return. Distinguish game terminal states from collector truncations. The guide notes that discounted terminal-loss penalties can interact with per-step costs and `gamma` to make delaying a loss attractive; inspect this if using shaped reward.

## Reproducibility and reporting

- Store configs, code revision, library versions, Jackdaw version, and action version with each run.
- Save `jackdaw.env.ACTION_VERSION`; retrain or explicitly validate if it changes.
- Seed Python, NumPy, PyTorch, and the environment where supported. Keep train/tuning/test seed sets disjoint.
- Keep observation copies when storing transitions: with `copy=False`, arrays are views that later steps overwrite.
- Report win rate, mean/median ante, blinds beaten, return, episode length, refused-action count, episode count, and variability.
- Use identical evaluation conditions for random and PPO agents, with masks enabled for both.

## Suggested folder layout

```text
Policy based maskable ppo/
  IMPLEMENTATION_PLAN.md
  config.yaml
  env_factory.py
  train.py
  evaluate.py
  callbacks.py
  README.md
  checkpoints/       # generated, ignored by git
  runs/              # generated logs, ignored by git
```

Validate the documented Gymnasium wrapper against the installed Jackdaw version before implementing model-specific code. The guide covers structured observations, factored actions flattened to 500 candidates, masks, rewards, vectorization, and action versioning. Keep the baseline close to Jackdaw's SB3 example; add custom encoders or reward logic only when measurements motivate them.
