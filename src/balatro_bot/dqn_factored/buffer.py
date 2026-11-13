"""Replay buffer for the agent. Stores raw observations and the mask object for both sides of each transition.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    global_vec: Any
    entities: dict[str, Any]
    mask: Any
    action: Any
    reward: float
    next_global_vec: Any
    next_entities: dict[str, Any]
    next_mask: Any
    done: bool  # win/loss or continuining


class FactoredReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = capacity # is capacity needed? how large r we training
        # tried 100k steps
        self._data: list[Transition] = []
        self._ptr = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, transition: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._ptr] = transition
            self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(self._data, batch_size)
