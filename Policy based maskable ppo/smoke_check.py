"""Quickly exercise both Jackdaw's direct and Gymnasium APIs."""

from __future__ import annotations

import argparse

import numpy as np

from jackdaw.env import BalatroEnvironment, DirectAdapter, RandomAgent

from env_factory import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()

    direct = BalatroEnvironment(
        adapter_factory=DirectAdapter,
        max_steps=args.max_steps,
        on_refusal="raise",
    )
    agent = RandomAgent()
    obs, mask, info = direct.reset(seed=f"SMOKE_{args.seed}")
    agent.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(
            obs={"global": obs.global_context, **obs.entities},
            action_mask=mask,
            info=info,
        )
        obs, terminated, truncated, mask, info = direct.step(action)
    print(
        "direct API: "
        f"steps={direct.episode_length} ante={direct.episode_ante} "
        f"won={direct.episode_won} terminated={terminated} truncated={truncated}"
    )

    gym_env = make_env(max_steps=args.max_steps)
    try:
        observation, info = gym_env.reset(seed=args.seed)
        mask = gym_env.action_masks()
        assert gym_env.observation_space.contains(observation)
        assert mask.shape == (500,) and mask.dtype == np.bool_ and mask.any()
        legal_action = int(np.flatnonzero(mask)[0])
        observation, reward, terminated, truncated, info = gym_env.step(legal_action)
        assert np.isfinite(reward)
        if not (terminated or truncated):
            assert gym_env.observation_space.contains(observation)
            assert gym_env.action_masks().any()
        print(
            "Gymnasium API: "
            f"mask_shape={mask.shape} legal_actions={int(mask.sum())} "
            f"reward={reward:.3f} terminated={terminated} truncated={truncated}"
        )
    finally:
        gym_env.close()


if __name__ == "__main__":
    main()
