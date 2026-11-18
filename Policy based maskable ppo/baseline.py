"""Run a reproducible uniformly random legal-action baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from env_factory import environment_metadata, make_env


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "runs" / "random_baseline"


def _episode(env: Any, seed: int, action_rng: np.random.Generator) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    mask = env.action_masks()
    if not env.observation_space.contains(observation):
        raise RuntimeError("Initial observation is outside the declared observation space")

    total_reward = 0.0
    steps = 0
    legal_count_sum = 0
    terminated = truncated = False
    while not (terminated or truncated):
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size == 0:
            raise RuntimeError(f"No legal actions at seed={seed}, step={steps}")
        legal_count_sum += int(legal_actions.size)

        action = int(action_rng.choice(legal_actions))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break
        if not env.observation_space.contains(observation):
            raise RuntimeError(f"Invalid observation at seed={seed}, step={steps}")
        mask = env.action_masks()

    return {
        "seed": seed,
        "won": bool(info.get("balatro/won", False)),
        "ante_reached": int(info.get("balatro/ante_reached", 1)),
        "blinds_beaten": int(info.get("balatro/rounds_beaten", 0)),
        "steps": steps,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "refused": bool(info.get("balatro/refused", False)),
        "mean_legal_actions": legal_count_sum / max(steps, 1),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in rows)

    def median(key: str) -> float:
        return float(statistics.median(float(row[key]) for row in rows))

    return {
        "episodes": len(rows),
        "win_rate": mean("won"),
        "mean_ante_reached": mean("ante_reached"),
        "median_ante_reached": median("ante_reached"),
        "mean_blinds_beaten": mean("blinds_beaten"),
        "mean_return": mean("return"),
        "mean_episode_steps": mean("steps"),
        "mean_legal_actions": mean("mean_legal_actions"),
        "terminated_episodes": sum(bool(row["terminated"]) for row in rows),
        "truncated_episodes": sum(bool(row["truncated"]) for row in rows),
        "refused_actions": sum(bool(row["refused"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--reward", choices=("blind_progress", "sparse"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    env_cfg = config["environment"]
    baseline_cfg = config["baseline"]
    episodes = args.episodes or int(baseline_cfg["episodes"])
    seed_start = args.seed_start if args.seed_start is not None else int(baseline_cfg["seed_start"])
    reward = args.reward or env_cfg["reward"]
    if episodes < 1:
        parser.error("--episodes must be positive")

    # Keep Python-side randomness controlled too; reset(seed=...) controls Jackdaw's
    # engine and legal-action candidate sampler for each episode.
    random.seed(seed_start)
    action_rng = np.random.default_rng(seed_start)
    env = make_env(
        reward_name=reward,
        max_steps=int(env_cfg["max_steps"]),
        seed_prefix=f"RANDOM_{seed_start}",
        max_free_moves=int(env_cfg["max_free_moves"]),
        on_refusal=env_cfg["on_refusal"],
    )

    rows = []
    try:
        for index in range(episodes):
            seed = seed_start + index
            row = _episode(env, seed, action_rng)
            rows.append(row)
            print(
                f"episode={index + 1}/{episodes} seed={seed} "
                f"ante={row['ante_reached']} won={row['won']} "
                f"steps={row['steps']} return={row['return']:.3f} "
                f"truncated={row['truncated']} refused={row['refused']}"
            )
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "uniform_random_over_legal_actions",
        "reward": reward,
        "seed_start": seed_start,
        "config": config,
        "environment": environment_metadata(env),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jackdaw", "gymnasium", "numpy")
        },
        "python": sys.version,
        "numpy": np.__version__,
        "results": _summary(rows),
        "episode_results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["results"], indent=2))
    print(f"Saved results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
