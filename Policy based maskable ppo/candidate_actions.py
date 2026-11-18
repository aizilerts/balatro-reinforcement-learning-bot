"""Encode Jackdaw's changing factored-action table in the observation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from jackdaw.env import balatro_game_spec


NUM_ACTION_TYPES = len(balatro_game_spec().action_types)
MAX_ENTITY_TARGETS = 10
MAX_HAND_CARDS = 8
CANDIDATE_ACTION_FEATURES = NUM_ACTION_TYPES + 1 + MAX_ENTITY_TARGETS + MAX_HAND_CARDS + 1 + 1


class CandidateActionObservation(gym.ObservationWrapper):
    """Add a fixed row of semantic features for every action slot.

    Rows are aligned with the environment's 500 output slots. Each row stores
    action type, entity target, selected hand-card mask, selection size, and
    current legality. This lets a policy score the meaning of a candidate
    instead of learning from the slot number alone.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("CandidateActionObservation requires a Dict observation")
        self.num_actions = int(env.action_space.n)
        spaces_by_key = dict(env.observation_space.spaces)
        spaces_by_key["candidate_action"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_actions, CANDIDATE_ACTION_FEATURES),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(spaces_by_key)

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        features = np.zeros(
            (self.num_actions, CANDIDATE_ACTION_FEATURES), dtype=np.float32
        )
        try:
            actions = self.unwrapped._action_table
            legal = np.asarray(self.env.action_masks(), dtype=bool)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            actions, legal = (), np.zeros(self.num_actions, dtype=bool)

        entity_present_offset = NUM_ACTION_TYPES
        entity_index_offset = entity_present_offset + 1
        card_mask_offset = entity_index_offset + MAX_ENTITY_TARGETS
        card_count_offset = card_mask_offset + MAX_HAND_CARDS
        legal_offset = card_count_offset + 1

        for action_id, action in enumerate(actions[: self.num_actions]):
            action_type = int(action.action_type)
            if 0 <= action_type < NUM_ACTION_TYPES:
                features[action_id, action_type] = 1.0
            target = action.entity_target
            if target is not None:
                features[action_id, entity_present_offset] = 1.0
                target = int(target)
                if 0 <= target < MAX_ENTITY_TARGETS:
                    features[action_id, entity_index_offset + target] = 1.0
            card_targets = action.card_target or ()
            for card_index in card_targets:
                card_index = int(card_index)
                if 0 <= card_index < MAX_HAND_CARDS:
                    features[action_id, card_mask_offset + card_index] = 1.0
            features[action_id, card_count_offset] = min(len(card_targets), MAX_HAND_CARDS) / MAX_HAND_CARDS
            if action_id < legal.size and legal[action_id]:
                features[action_id, legal_offset] = 1.0

        result = dict(observation)
        result["candidate_action"] = features
        return result

    def action_masks(self) -> np.ndarray:
        """Expose the wrapped legal-action mask to MaskablePPO."""
        return self.env.action_masks()  # type: ignore[attr-defined]
