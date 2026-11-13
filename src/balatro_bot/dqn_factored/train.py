"""Training loop for DQN on the raw BalatroEnvironment.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from jackdaw.env import BalatroEnvironment, DirectAdapter, Transition, blind_progress_reward

Reward = Callable[[Transition], float]

from .agent import FactoredDQNAgent
from .buffer import FactoredReplayBuffer
from .buffer import Transition as BufferTransition



SKIP_BLIND = 3
DISABLED_ACTION_TYPES = (SKIP_BLIND,)
# disabling skip blind action for now
# the greedy policy led to a skip then stalling loop instead of learning the 
# game


def _disable_types(mask, types=DISABLED_ACTION_TYPES):
    mask.type_mask = mask.type_mask.copy()
    mask.type_mask[list(types)] = False
    return mask


def make_env(seed_prefix: str, max_steps: int) -> BalatroEnvironment:
    return BalatroEnvironment(adapter_factory=DirectAdapter, seed_prefix=seed_prefix, max_steps=max_steps)


def train(
    total_steps: int = 20_000,
    buffer_capacity: int = 20_000,
    batch_size: int = 32,
    learning_starts: int = 500,
    train_every: int = 4,
    target_sync_every: int = 1_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 15_000,
    gamma: float = 0.99,
    lr: float = 1e-4,
    max_steps_per_episode: int = 10_000,
    seed_prefix: str = "TRAIN_FACTORED",
    seed: int = 0,
    log_every_episodes: int = 20,
    reward_fn: Reward = blind_progress_reward,
) -> FactoredDQNAgent:
    rng = np.random.default_rng(seed)
    env = make_env(seed_prefix, max_steps_per_episode)
    agent = FactoredDQNAgent(lr=lr, gamma=gamma)
    buffer = FactoredReplayBuffer(buffer_capacity, seed=seed)

    obs, mask, info = env.reset()
    mask = _disable_types(mask)
    episode_return = 0.0
    episode_count = 0
    blinds_beaten_total = 0
    start_time = time.time()

    for step in range(1, total_steps + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - step / epsilon_decay_steps)
        action = agent.act(obs.global_context, obs.entities, mask, epsilon, rng)

        next_obs, terminated, truncated, next_mask, info = env.step(action)
        next_mask = _disable_types(next_mask)
        transition = Transition.from_step(action, terminated, truncated, info)
        reward = reward_fn(transition)
        if transition.blind_beaten:
            blinds_beaten_total += 1

        buffer.add(
            BufferTransition(
                global_vec=obs.global_context,
                entities=obs.entities,
                mask=mask,
                action=action,
                reward=reward,
                next_global_vec=next_obs.global_context,
                next_entities=next_obs.entities,
                next_mask=next_mask,
                done=terminated,
            )
        )
        episode_return += reward

        if step >= learning_starts and step % train_every == 0 and len(buffer) >= batch_size:
            agent.train_step(buffer.sample(batch_size))
        if step % target_sync_every == 0:
            agent.sync_target()

        if terminated or truncated:
            episode_count += 1
            if episode_count % log_every_episodes == 0 or episode_count == 1:
                elapsed = time.time() - start_time
                print(
                    f"[episode {episode_count}] step={step} return={episode_return:.2f} "
                    f"len={env.episode_length} ante={env.episode_ante} won={env.episode_won} "
                    f"blinds_beaten_total={blinds_beaten_total} epsilon={epsilon:.3f} elapsed={elapsed:.0f}s"
                )
            obs, mask, info = env.reset()
            mask = _disable_types(mask)
            episode_return = 0.0
        else:
            obs, mask = next_obs, next_mask

        if step % max(1, total_steps // 10) == 0:
            print(f"-- step {step}/{total_steps} --")

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the factored (action-conditioned) DQN agent")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(total_steps=args.total_steps, seed=args.seed)
