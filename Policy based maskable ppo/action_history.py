"""Observation features that expose short action history to the policy."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from jackdaw.env import balatro_game_spec


_SORT_ACTION_TYPES = {
    index
    for index, action in enumerate(balatro_game_spec().action_types)
    if action.name in {"sort_hand_rank", "sort_hand_suit"}
}
_JOKER_SWAP_ACTIONS = {
    index: action.name
    for index, action in enumerate(balatro_game_spec().action_types)
    if action.name in {"swap_jokers_left", "swap_jokers_right"}
}
_SELECT_BLIND_ACTION_TYPES = {
    index
    for index, action in enumerate(balatro_game_spec().action_types)
    if action.name == "select_blind"
}
_HISTORY_FEATURES = 1


class PreviousActionObservation(gym.ObservationWrapper):
    """Track recent sorts and Joker swaps for features and legal masking.

    Both Rank Sort and Suit Sort set the flag. Any other action clears it, so
    the policy can distinguish a repeated sort from the first sort in a run.
    An immediate reversal of the same adjacent Joker pair is also masked.
    """

    def __init__(self, env: gym.Env, *, max_joker_swaps_per_blind: int = 5):
        super().__init__(env)
        if max_joker_swaps_per_blind < 0:
            raise ValueError("max_joker_swaps_per_blind cannot be negative")
        original = env.observation_space
        assert isinstance(original, spaces.Dict)
        spaces_by_key = dict(original.spaces)
        global_space = spaces_by_key["global"]
        assert isinstance(global_space, spaces.Box)
        low = np.concatenate((global_space.low, np.zeros(_HISTORY_FEATURES, dtype=np.float32)))
        high = np.concatenate((global_space.high, np.ones(_HISTORY_FEATURES, dtype=np.float32)))
        spaces_by_key["global"] = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Dict(spaces_by_key)
        self._last_action_was_sort = False
        self._last_joker_swap_pair: tuple[int, int] | None = None
        self.max_joker_swaps_per_blind = int(max_joker_swaps_per_blind)
        self._joker_swaps_this_blind = 0
        self._joker_orders_seen: set[tuple[str, ...]] = set()

    def _current_joker_order(self) -> tuple[str, ...]:
        game = getattr(self.unwrapped, "game", None)
        signatures = []
        for card in (getattr(game, "jokers", ()) or ()):
            edition = getattr(card, "edition", None)
            if isinstance(edition, dict):
                edition = tuple(sorted((str(key), repr(value)) for key, value in edition.items()))
            signatures.append(
                f"{card.key}|edition={edition!r}|seal={getattr(card, 'seal', None)!r}"
            )
        return tuple(signatures)

    @staticmethod
    def _joker_swap_pair(action: Any) -> tuple[int, int] | None:
        """Return the adjacent Joker slots affected by a swap action."""
        action_type = int(getattr(action, "action_type", -1))
        name = _JOKER_SWAP_ACTIONS.get(action_type)
        target = getattr(action, "entity_target", None)
        if name is None or target is None:
            return None
        target = int(target)
        pair = (target - 1, target) if name == "swap_jokers_left" else (target, target + 1)
        return pair if pair[0] >= 0 else None

    def _append_history(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = dict(observation)
        flag = np.array([self._last_action_was_sort], dtype=np.float32)
        result["global"] = np.concatenate((observation["global"], flag))
        return result

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self._last_action_was_sort = False
        self._last_joker_swap_pair = None
        self._joker_swaps_this_blind = 0
        observation, info = self.env.reset(**kwargs)
        self._joker_orders_seen = {self._current_joker_order()}
        return self._append_history(observation), info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_type = -1
        joker_swap_pair = None
        previous_joker_order = self._current_joker_order()
        try:
            factored_action = self.unwrapped._action_table[int(action)]
            action_type = int(factored_action.action_type)
            joker_swap_pair = self._joker_swap_pair(factored_action)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_joker_order = self._current_joker_order()
        self._last_action_was_sort = action_type in _SORT_ACTION_TYPES
        self._last_joker_swap_pair = joker_swap_pair
        if action_type in _SELECT_BLIND_ACTION_TYPES:
            self._joker_swaps_this_blind = 0
            self._joker_orders_seen = {current_joker_order}
        elif action_type in _JOKER_SWAP_ACTIONS:
            self._joker_swaps_this_blind += 1
            self._joker_orders_seen.add(current_joker_order)
        elif current_joker_order != previous_joker_order:
            # New or removed Jokers change the possible orderings. Keep the
            # per-blind swap count, but start order-cycle detection afresh.
            self._joker_orders_seen = {current_joker_order}
        return self._append_history(observation), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Expose Jackdaw's mask and block redundant consecutive actions."""
        mask = np.asarray(self.env.action_masks(), dtype=bool).copy()  # type: ignore[attr-defined]
        if self._last_action_was_sort:
            # The agent may sort once to arrange its hand, but another sort
            # before any other action is redundant and has caused long loops.
            # Mask both sort variants while preserving all other legal moves.
            try:
                for action_id, action in enumerate(self.unwrapped._action_table):
                    if action_id < mask.size and int(action.action_type) in _SORT_ACTION_TYPES:
                        mask[action_id] = False
            except (AttributeError, TypeError, ValueError):
                pass
        if self._last_joker_swap_pair is not None:
            # A left/right swap of the same adjacent pair immediately undoes
            # the previous action. Other adjacent pairs remain available so
            # the agent can still build a longer Joker ordering.
            try:
                for action_id, action in enumerate(self.unwrapped._action_table):
                    if (
                        action_id < mask.size
                        and self._joker_swap_pair(action) == self._last_joker_swap_pair
                    ):
                        mask[action_id] = False
            except (AttributeError, TypeError, ValueError):
                pass
        if self._joker_swaps_this_blind >= self.max_joker_swaps_per_blind:
            try:
                for action_id, action in enumerate(self.unwrapped._action_table):
                    if action_id < mask.size and int(action.action_type) in _JOKER_SWAP_ACTIONS:
                        mask[action_id] = False
            except (AttributeError, TypeError, ValueError):
                pass
        else:
            # Reject any swap that would revisit a Joker arrangement already
            # seen since this blind was selected. This also prevents longer
            # cycles that alternate between different adjacent pairs.
            current_order = self._current_joker_order()
            try:
                for action_id, action in enumerate(self.unwrapped._action_table):
                    if action_id >= mask.size or int(action.action_type) not in _JOKER_SWAP_ACTIONS:
                        continue
                    pair = self._joker_swap_pair(action)
                    if pair is None or pair[1] >= len(current_order):
                        continue
                    next_order = list(current_order)
                    next_order[pair[0]], next_order[pair[1]] = next_order[pair[1]], next_order[pair[0]]
                    if tuple(next_order) in self._joker_orders_seen:
                        mask[action_id] = False
            except (AttributeError, TypeError, ValueError):
                pass
        return mask
