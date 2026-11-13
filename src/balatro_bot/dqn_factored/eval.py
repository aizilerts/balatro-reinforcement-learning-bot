
from __future__ import annotations

import time

import numpy as np

from .agent import FactoredDQNAgent
from .train import _disable_types, make_env


def evaluate(
    agent: FactoredDQNAgent,
    n_episodes: int = 20,
    seed_prefix: str = "EVAL_FACTORED",
    max_steps_per_episode: int = 10_000,
) -> dict:
    env = make_env(seed_prefix, max_steps_per_episode)
    rng = np.random.default_rng(0)  
    antes, wins, lengths, durations = [], [], [], []
    for _ in range(n_episodes):
        start = time.time()
        obs, mask, info = env.reset()
        mask = _disable_types(mask)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs.global_context, obs.entities, mask, epsilon=0.0, rng=rng)
            obs, terminated, truncated, mask, info = env.step(action)
            mask = _disable_types(mask)

        antes.append(env.episode_ante)
        wins.append(bool(env.episode_won))
        lengths.append(env.episode_length)
        durations.append(time.time() - start)

    return {
        "n_episodes": n_episodes,
        "mean_ante": float(np.mean(antes)),
        "win_rate": float(np.mean(wins)),
        "mean_length": float(np.mean(lengths)),
        "mean_duration_s": float(np.mean(durations)),
        "antes": antes,
        "wins": wins,
    }
