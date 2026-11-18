"""Construction helpers for the Jackdaw Gymnasium environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jackdaw.env import (
    ACTION_VERSION,
    BalatroGymnasiumEnv,
    DirectAdapter,
    ShapedReward,
    blind_progress_reward,
    sparse_reward,
)

from joker_features import JokerPreferenceReward


REWARD_FUNCTIONS: dict[str, Callable[..., float]] = {
    "blind_progress": blind_progress_reward,
    "sparse": sparse_reward,
}

DEFAULT_SHAPED_REWARD = {
    "step_cost": 0.001,
    "shop_step_cost": 0.002,
    "blind_beaten": 1.0,
    "boss_beaten": 0.1,
    "hands_left": 0.01,
    "score_progress": 0.02,
    "win": 1.0,
    "loss": 1.0,
}


def make_env(
    *,
    reward_name: str = "blind_progress",
    max_steps: int = 10_000,
    seed_prefix: str = "TRAIN",
    max_free_moves: int = 10,
    on_refusal: str = "raise",
    reward_kwargs: dict[str, float] | None = None,
    joker_preference: dict[str, Any] | None = None,
) -> BalatroGymnasiumEnv:
    """Create a fresh environment with an explicitly selected reward function."""
    if reward_name == "shaped":
        reward_fn = ShapedReward(**(reward_kwargs or DEFAULT_SHAPED_REWARD))
        preference = dict(joker_preference or {})
        enabled = bool(preference.pop("enabled", False))
        if enabled:
            reward_fn = JokerPreferenceReward(reward_fn, **preference)
    else:
        try:
            reward_fn = REWARD_FUNCTIONS[reward_name]
        except KeyError as exc:
            choices = ", ".join(sorted((*REWARD_FUNCTIONS, "shaped")))
            raise ValueError(f"Unknown reward {reward_name!r}; choose from {choices}") from exc

    return BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        reward_fn=reward_fn,
        max_steps=max_steps,
        seed_prefix=seed_prefix,
        max_free_moves=max_free_moves,
        on_refusal=on_refusal,
        copy=True,
    )


def environment_metadata(env: Any) -> dict[str, Any]:
    """Return metadata worth saving beside a run or checkpoint."""
    return {
        "jackdaw_action_version": int(getattr(env, "action_version", ACTION_VERSION)),
        "action_space": repr(env.action_space),
        "observation_space": repr(env.observation_space),
    }
