"""Factored double-DQN agent: masked e-greedy + a Bellman update.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from jackdaw.env import FactoredAction

from .network import CARD_SELECT_TYPES, ENTITY_KIND_BY_TYPE, FactoredQNetwork, decide, q_of_action


def build_action(action_type: int, target) -> FactoredAction:
    # Converts action type back into FactoredAction for the Balatro env to process
    if action_type in CARD_SELECT_TYPES: 
        return FactoredAction(action_type=action_type, card_target=tuple(target), entity_target=None)
    if action_type in ENTITY_KIND_BY_TYPE:
        return FactoredAction(action_type=action_type, card_target=None, entity_target=int(target))
    return FactoredAction(action_type=action_type, card_target=None, entity_target=None)


def sample_random_legal_action(entities: dict[str, np.ndarray], mask, rng: np.random.Generator) -> FactoredAction:
    # e-greedy random branch, samples uniformly among legal action choices
    legal_types = np.nonzero(mask.type_mask)[0]
    t = int(rng.choice(legal_types))

    if t in CARD_SELECT_TYPES:
        legal = [i for i in range(len(mask.card_mask)) if mask.card_mask[i]]
        forced = mask.forced_card
        if forced is not None and forced not in legal:
            legal.append(forced)
        base = [forced] if forced is not None else []
        rest = [i for i in legal if i != forced]
        rng.shuffle(rest)
        lo = max(mask.min_card_select, len(base))
        hi = min(mask.max_card_select, len(base) + len(rest))
        k = int(rng.integers(lo, hi + 1)) if hi >= lo else len(base)
        subset = tuple(sorted(base + rest[: k - len(base)]))
        return build_action(t, subset)

    if t in ENTITY_KIND_BY_TYPE:
        entity_mask = mask.entity_masks[t]
        legal_entities = np.nonzero(entity_mask)[0]
        return build_action(t, int(rng.choice(legal_entities)))

    return build_action(t, None)


class FactoredDQNAgent:
    def __init__(self, lr: float = 1e-4, gamma: float = 0.99, hidden: int = 128):
        self.gamma = gamma
        self.online = FactoredQNetwork(hidden=hidden) # entire network
        self.target = copy.deepcopy(self.online) #deep copy for computing targets
        self.target.eval() #not yet needed(?) no training-mode only layers so far
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

    def sync_target(self) -> None:
        # sync online network weights into target
        self.target.load_state_dict(self.online.state_dict())

    def act(
        self,
        global_vec: np.ndarray,
        entities: dict[str, np.ndarray],
        mask,
        epsilon: float,
        rng: np.random.Generator,
    ) -> FactoredAction:
        # e-greedy policy
        if rng.random() < epsilon: # exploration
            return sample_random_legal_action(entities, mask, rng)

        with torch.no_grad(): # exploitation
            type_bias, entity_values = self.online.encode(global_vec, entities)
            q_per_type, reconstruction = decide(type_bias, entity_values, mask)
        best_t = int(np.argmax(q_per_type))
        return build_action(best_t, reconstruction.get(best_t))

    def train_step(self, batch: list) -> float:
        # gradient update over sampled batch
        losses = []
        for t in batch:
            type_bias, entity_values = self.online.encode(t.global_vec, t.entities)
            q_sa = q_of_action(type_bias, entity_values, t.action)

            if t.done:
                target_value = 0.0
            else:
                with torch.no_grad():
                    #Double DQN, use online net for action selection, then
                    # use frozen target net for evluating selected action
                    type_bias2_on, entity_values2_on = self.online.encode(t.next_global_vec, t.next_entities)
                    q_per_type2, reconstruction2 = decide(type_bias2_on, entity_values2_on, t.next_mask)
                    legal2 = np.nonzero(t.next_mask.type_mask)[0]
                    if len(legal2) == 0:
                        target_value = 0.0
                    else:
                        best_t2 = int(np.argmax(q_per_type2))
                        action2 = build_action(best_t2, reconstruction2.get(best_t2))
                        type_bias2_tgt, entity_values2_tgt = self.target.encode(t.next_global_vec, t.next_entities)
                        target_value = float(q_of_action(type_bias2_tgt, entity_values2_tgt, action2).item())

            target = t.reward + self.gamma * target_value # bellman target one-step
            losses.append((q_sa - target) ** 2) #squared error

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item()) # returns scalar loss (can use for logging/charts)
