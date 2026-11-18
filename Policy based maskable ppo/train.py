"""Train a MaskablePPO policy with Jackdaw's legal-action masks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_factory import DEFAULT_SHAPED_REWARD, environment_metadata, make_env
from action_history import PreviousActionObservation
from candidate_actions import CandidateActionObservation, CANDIDATE_ACTION_FEATURES
from candidate_policy import CandidateActionActorCriticPolicy, GameStateFeaturesExtractor
from joker_features import JokerEffectObservation

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_RUNS = HERE / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path, help="Resume from a MaskablePPO .zip checkpoint")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    env_cfg, train_cfg = config["environment"], config["training"]
    total = args.timesteps or int(train_cfg["total_timesteps"])
    seed = args.seed if args.seed is not None else int(train_cfg["seed"])
    if total < 1:
        parser.error("--timesteps must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = args.run_name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = args.output_dir or (DEFAULT_RUNS / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    def make_training_env():
        raw_env = make_env(
            reward_name=env_cfg["reward"],
            reward_kwargs=env_cfg.get("reward_kwargs", DEFAULT_SHAPED_REWARD),
            joker_preference=env_cfg.get("joker_preference"),
            max_steps=int(env_cfg["max_steps"]),
            seed_prefix=f"PPO_{seed}",
            max_free_moves=int(env_cfg["max_free_moves"]),
            on_refusal=env_cfg["on_refusal"],
        )
        wrapped = PreviousActionObservation(
            JokerEffectObservation(raw_env),
            max_joker_swaps_per_blind=int(env_cfg.get("max_joker_swaps_per_blind", 5)),
        )
        return Monitor(CandidateActionObservation(wrapped))

    env = DummyVecEnv([make_training_env])
    try:
        if args.resume:
            model = MaskablePPO.load(args.resume, env=env, device=train_cfg.get("device", "auto"))
            model.set_random_seed(seed)
        else:
            model = MaskablePPO(
                CandidateActionActorCriticPolicy, env,
                learning_rate=float(train_cfg["learning_rate"]),
                n_steps=int(train_cfg["n_steps"]),
                batch_size=int(train_cfg["batch_size"]),
                n_epochs=int(train_cfg["n_epochs"]),
                gamma=float(train_cfg["gamma"]),
                gae_lambda=float(train_cfg["gae_lambda"]),
                clip_range=float(train_cfg["clip_range"]),
                ent_coef=float(train_cfg["ent_coef"]),
                vf_coef=float(train_cfg["vf_coef"]),
                max_grad_norm=float(train_cfg["max_grad_norm"]),
                policy_kwargs={
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                    "features_extractor_class": GameStateFeaturesExtractor,
                    "candidate_hidden_dim": 64,
                },
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=seed,
                device=train_cfg.get("device", "auto"),
                verbose=1,
            )
        model.tensorboard_log = str(run_dir / "tensorboard")
        checkpoint = CheckpointCallback(
            save_freq=max(int(train_cfg["checkpoint_freq"]), 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="maskable_ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model.learn(total_timesteps=total, callback=checkpoint, reset_num_timesteps=args.resume is None,
                    tb_log_name=run_name, progress_bar=False)
        model.save(run_dir / "maskable_ppo_final")
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "seed": seed,
            "requested_additional_timesteps": total,
            "total_timesteps": model.num_timesteps,
            "resumed_from": str(args.resume) if args.resume else None,
            "joker_effect_features": True,
            "previous_action_sort_feature": True,
            "candidate_action_features": True,
            "candidate_action_feature_dim": CANDIDATE_ACTION_FEATURES,
            "candidate_action_policy": "shared_context_candidate_scorer",
            "joker_swap_reversal_mask": True,
            "joker_swap_order_cycle_mask": True,
            "max_joker_swaps_per_blind": int(env_cfg.get("max_joker_swaps_per_blind", 5)),
            "joker_preference_reward": env_cfg.get("joker_preference", {"enabled": False}),
            "environment": environment_metadata(env.envs[0]),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        }
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved final checkpoint to {run_dir / 'maskable_ppo_final.zip'}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
