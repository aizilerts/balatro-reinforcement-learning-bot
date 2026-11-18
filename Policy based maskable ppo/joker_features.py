"""Explicit Balatro Joker effect tags for observations and shaped rewards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from jackdaw.engine.data import load
from jackdaw.env.observation import NUM_CENTER_KEYS, center_key_id


# Feature order is Chips, additive Mult, multiplicative X Mult. These are tags,
# not a claim that every Joker is useful in every build or every situation.
JOKER_EFFECT_FEATURES = ("chips", "mult", "x_mult")
_SCORE_KEYS = {
    "chips": {"chips", "chip_mod", "t_chips"},
    "mult": {"mult", "t_mult", "s_mult"},
    "x_mult": {"xmult", "x_mult"},
}

# Some center records use an opaque `extra` field for a well-known scoring
# effect. Most categories are inferred from config keys / the effect label;
# these overrides cover common entries whose catalog metadata is underspecified.
_CATEGORY_OVERRIDES: dict[str, set[str]] = {
    "x_mult": {
        "j_ancient", "j_baron", "j_blackboard", "j_caino", "j_campfire",
        "j_drivers_license", "j_flower_pot", "j_hit_the_road", "j_madness",
        "j_photograph", "j_stencil", "j_steel_joker",
        "j_triboulet", "j_throwback",
    },
    "mult": {
        "j_even_steven", "j_fibonacci", "j_fortune_teller", "j_gros_michel",
        "j_raised_fist", "j_ride_the_bus", "j_shoot_the_moon", "j_supernova",
    },
    "chips": {
        "j_arrowhead", "j_blue_joker", "j_bull", "j_hiker", "j_ice_cream",
        "j_odd_todd", "j_scary_face", "j_stone",
    },
}


def _keys_recursive(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key).lower() for key in value} | set().union(
            *(_keys_recursive(child) for child in value.values()), set()
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_keys_recursive(child) for child in value), set())
    return set()


def _effect_tags(center: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    key = str(center.get("key", ""))
    effect = str(center.get("effect") or "").lower()
    config_keys = _keys_recursive(center.get("config", {}))
    labels: set[str] = set()
    for label, aliases in _SCORE_KEYS.items():
        if config_keys & aliases:
            labels.add(label)
    if "mult" in effect:
        labels.add("x_mult" if "x" in effect and any(char.isdigit() for char in effect) else "mult")
    if "chip" in effect:
        labels.add("chips")
    for label, keys in _CATEGORY_OVERRIDES.items():
        if key in keys:
            labels.add(label)
    return tuple(label in labels for label in JOKER_EFFECT_FEATURES)  # type: ignore[return-value]


def _catalog_tags() -> dict[int, tuple[bool, bool, bool]]:
    centers = load("centers")
    return {
        center_key_id(key): _effect_tags(center)
        for key, center in centers.items()
        if center.get("set") == "Joker"
    }


_TAGS_BY_CENTER_ID = _catalog_tags()
_CENTERS = load("centers")


class JokerEffectObservation(gym.ObservationWrapper):
    """Append explicit Chips / Mult / X Mult flags to owned and shop Joker rows."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        original = env.observation_space
        assert isinstance(original, spaces.Dict)
        spaces_by_key = dict(original.spaces)
        for key in ("joker", "shop_item"):
            space = spaces_by_key[key]
            assert isinstance(space, spaces.Box)
            shape = (*space.shape[:-1], space.shape[-1] + len(JOKER_EFFECT_FEATURES))
            spaces_by_key[key] = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
            )
        self.observation_space = spaces.Dict(spaces_by_key)

    @staticmethod
    def _append_tags(rows: np.ndarray) -> np.ndarray:
        tags = np.zeros((*rows.shape[:-1], len(JOKER_EFFECT_FEATURES)), dtype=np.float32)
        for index, row in enumerate(rows):
            center_id = int(round(float(row[0]) * NUM_CENTER_KEYS))
            category = _TAGS_BY_CENTER_ID.get(center_id)
            if category is not None:
                tags[index] = category
        return np.concatenate((rows.astype(np.float32, copy=False), tags), axis=-1)

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = dict(observation)
        result["joker"] = self._append_tags(observation["joker"])
        result["shop_item"] = self._append_tags(observation["shop_item"])
        return result

    def action_masks(self) -> np.ndarray:
        """Expose Jackdaw's mask through the observation wrapper for MaskablePPO."""
        return self.env.action_masks()  # type: ignore[attr-defined]


class JokerPreferenceReward:
    """Preserve Jackdaw's shaped reward and add a small per-run acquisition bonus."""

    def __init__(
        self,
        base_reward: Any,
        *,
        x_mult_bonus: float = 0.04,
        mult_bonus: float = 0.025,
        chips_bonus: float = 0.01,
        hand_swap_penalty: float = 0.5,
        repeated_sort_penalty: float = 1.0,
        sell_joker_below_capacity_penalty: float = 0.5,
        sell_consumable_below_capacity_penalty: float = 0.5,
        use_consumable_bonus: float = 0.05,
        skip_pack_below_capacity_penalty: float = 0.5,
        joker_acquisition_bonus_when_room: float = 0.1,
    ) -> None:
        self.base_reward = base_reward
        self.bonuses = {
            "x_mult": float(x_mult_bonus),
            "mult": float(mult_bonus),
            "chips": float(chips_bonus),
        }
        self.hand_swap_penalty = float(hand_swap_penalty)
        self._seen_jokers: set[str] = set()
        self.repeated_sort_penalty = float(repeated_sort_penalty)
        self._last_action_type: int | None = None
        self.sell_joker_below_capacity_penalty = float(sell_joker_below_capacity_penalty)
        self.sell_consumable_below_capacity_penalty = float(sell_consumable_below_capacity_penalty)
        self.use_consumable_bonus = float(use_consumable_bonus)
        self.skip_pack_below_capacity_penalty = float(skip_pack_below_capacity_penalty)
        self.joker_acquisition_bonus_when_room = float(joker_acquisition_bonus_when_room)
        self._last_joker_count: int | None = None

    def reset(self) -> None:
        self._seen_jokers.clear()
        self._last_action_type = None
        self._last_joker_count = None
        reset = getattr(self.base_reward, "reset", None)
        if callable(reset):
            reset()

    def __call__(self, transition: Any) -> float:
        reward = float(self.base_reward(transition))
        action_type = int(getattr(transition.action, "action_type", -1))
        # Jackdaw action types 17 and 18 swap adjacent cards in the hand.
        # The order is usually irrelevant during play, but these actions can
        # loop and delay scoring; penalize each swap as requested.
        if action_type in (17, 18):
            reward -= self.hand_swap_penalty
        # Rank Sort (19) and Suit Sort (20) are both sort actions. Penalize a
        # sort when the immediately previous action was also a sort, including
        # switching from rank sort to suit sort or vice versa.
        if action_type in (19, 20) and self._last_action_type in (19, 20):
            reward -= self.repeated_sort_penalty
        self._last_action_type = action_type
        game = getattr(transition, "game", None)
        if game is None:
            return reward

        joker_count = len(game.jokers)
        # Buying a Joker (8) or selecting one from a pack (14) earns a small
        # acquisition bonus only when it actually increases the Joker count
        # and the area had an open slot before the action. This avoids
        # rewarding unrelated shop purchases or full-area replacements.
        if (
            action_type in (8, 14)
            and self._last_joker_count is not None
            and joker_count > self._last_joker_count
            and self._last_joker_count < int(transition.prev.joker_slots)
        ):
            reward += self.joker_acquisition_bonus_when_room
        self._last_joker_count = joker_count

        # On sell actions, the post-action area count is one below its count
        # before the sale. Penalize only sales made while that area had a free
        # slot; selling from a full area can be a reasonable replacement.
        if action_type == 9:
            count_before_sale = len(game.jokers) + 1
            if count_before_sale < int(transition.prev.joker_slots):
                reward -= self.sell_joker_below_capacity_penalty
        elif action_type == 10:
            count_before_sale = len(game.consumables) + 1
            if count_before_sale < int(transition.prev.consumable_slots):
                reward -= self.sell_consumable_below_capacity_penalty

        # Packs can contain Jokers or consumables. Avoid skipping one while
        # either inventory still has room, especially before the Joker area
        # is full. The live game reflects the same inventory on this action.
        elif action_type == 7:
            joker_room = len(game.jokers) < int(transition.prev.joker_slots)
            consumable_room = len(game.consumables) < int(transition.prev.consumable_slots)
            if joker_room or consumable_room:
                reward -= self.skip_pack_below_capacity_penalty

        # Using an owned consumable (11) or buying and using one in a single
        # shop action (22) gets a small bonus to reduce hoarding.
        if action_type in (11, 22):
            reward += self.use_consumable_bonus

        for card in game.jokers:
            key = str(card.key)
            if key in self._seen_jokers:
                continue
            self._seen_jokers.add(key)
            center = _CENTERS.get(key)
            if center is None:
                continue
            tags = dict(zip(JOKER_EFFECT_FEATURES, _effect_tags(center)))
            category = next((name for name in ("x_mult", "mult", "chips") if tags[name]), None)
            if category is not None:
                reward += self.bonuses[category]
        return reward
