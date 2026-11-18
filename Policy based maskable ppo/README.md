# Policy Based Maskable PPO

This folder contains the Jackdaw environment integration, a reproducible random baseline, and a MaskablePPO training/evaluation path. PPO uses Jackdaw's padded Dict observations, explicit Joker Chips / Mult / X Mult effect tags, action-history features, semantic features for each of the 500 candidate action slots, and Jackdaw's legal-action mask. A shared context/candidate scorer computes each action logit from the current game context and that candidate's description.

## Setup

Use the repository's `.venv` and install the training dependencies from `requirements-ppo.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r "Policy based maskable ppo\requirements-ppo.txt"
```

`requirements-phase1.txt` contains the Jackdaw, Gymnasium, and NumPy versions verified for the random baseline.

## Random baseline

```powershell
.\.venv\Scripts\python.exe "Policy based maskable ppo\baseline.py"
```

Defaults are in `config.json`. The baseline samples uniformly from currently legal candidates. Use `--reward sparse` for a sparse-reward comparison, or override episode count and seed start with `--episodes` and `--seed-start`.

## Train and evaluate PPO

```powershell
.\.venv\Scripts\python.exe "Policy based maskable ppo\train.py" --timesteps 100000 --run-name candidate_action_ppo_100k
.\.venv\Scripts\python.exe "Policy based maskable ppo\evaluate.py" "Policy based maskable ppo\runs\candidate_action_ppo_100k\maskable_ppo_final.zip" --record-actions
```

Training uses Jackdaw's `ShapedReward` with an explicit loss penalty and a stronger blind-cleared reward. It adds a configurable `+0.1` bonus when a buy or pack-pick action actually adds a Joker while the Joker area had an open slot. It also adds a small one-time bonus when a run first acquires a Joker with a tagged score effect: X Mult `+0.04`, additive Mult `+0.025`, Chips `+0.01`. A Joker with multiple tags receives the highest applicable effect bonus. The effect bonus applies once per Joker key per run, so repeatedly buying/selling the same Joker cannot repeatedly collect it. Hand reorder actions (Swap Hand Left/Right) receive a configurable `-0.5` penalty each. Consecutive Rank Sort / Suit Sort actions receive a configurable `-1.0` penalty, and the action mask blocks either sort immediately after a sort. Joker swaps are capped at five per blind by default; the mask also blocks any swap that would revisit a Joker ordering already seen during that blind, preventing cycles across multiple pairs. Selling a Joker or consumable while its area had an open slot receives a configurable `-0.5` penalty. Skipping a pack while either the Joker or consumable area has room also receives a configurable `-0.5` penalty, encouraging the agent to inspect/take pack contents before skipping. Using an owned consumable or buying-and-using one immediately receives a configurable `+0.05` bonus. These hints discourage unproductive loops and consumable hoarding while keeping scoring and beating blinds as the main objective. The tagger uses Jackdaw's center catalog config/effect metadata plus a small override list, so effects with opaque catalog metadata may not be fully classified.

Evaluation defaults to Jackdaw's blind-progress reward and reports win rate, ante, and blinds beaten separately from training return; it does not include training-only shaping bonuses. Its default seed range starts at 30000, separate from the repeatedly inspected 20000-20099 range. Action traces include Joker inventory changes and identify acquisitions caused by buy or pack-pick actions. Treat the 30000 range as a final holdout: use a separate range for repeated tuning and reserve fresh seeds if results guide more changes.

To compare the random baseline on the same initial holdout range, run `baseline.py --episodes 100 --seed-start 30000`; for repeated tuning, pass a different seed range to both evaluation commands.

Training writes periodic checkpoints and TensorBoard logs under the run directory. The semantic candidate-action observation and shared scorer change the model input and policy architecture, so start a fresh model rather than resuming an older checkpoint. For example:

```powershell
.\.venv\Scripts\python.exe "Policy based maskable ppo\train.py" --timesteps 100000 --run-name candidate_action_ppo_100k
.\.venv\Scripts\python.exe "Policy based maskable ppo\evaluate.py" "Policy based maskable ppo\runs\candidate_action_ppo_100k\maskable_ppo_final.zip" --record-actions
```

On resume, `--timesteps` is the additional training budget. Keep the held-out evaluation set for final reporting rather than repeatedly selecting checkpoints against it.

## Inspect actions

Pass `--record-actions` to evaluation to write `actions.jsonl`. Each line records the seed, step, selected action slot and decoded action/targets, action type, number of currently legal candidates, reward, and Joker inventory keys before and after the action. The summary also counts action types and lists Joker acquisitions attributed to buy or pack-pick actions. The action slot indexes Jackdaw's current factored-action table; the candidate feature matrix provides its state-specific meaning to the policy.

For a visual replay, pass `--record-replay`. By default this writes a standalone `replay_seed_<seed>.html` for the first evaluated seed. Use `--replay-seed` to choose another seed in the evaluation range. Open that HTML file in a browser; it has step controls and shows a simplified card table, game stats, and readable actions. This is a replay of the headless Jackdaw simulation, not captured footage from the Balatro game.

## Checkpoint compatibility

The candidate-action feature matrix changes the observation shape, and the shared candidate scorer changes the policy architecture. Older checkpoints cannot be resumed with this model; train a fresh run. To evaluate a pre-candidate-action checkpoint, pass `--no-candidate-actions` (and any older-feature flags required by that checkpoint). New checkpoints record the candidate-action architecture and observation shape in `run.json`.
