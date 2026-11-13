"""Action-conditioned Q-network for the raw Balatro action space.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_ACTION_TYPES = 23
D_GLOBAL = 235 
ENTITY_DIMS: dict[str, int] = { # Feature width of each entity
    "hand_card": 15,
    "joker": 15,
    "consumable": 7,
    "shop_item": 9,
    "pack_card": 15,
}

# Which entity array an action type's entity_target indexes into.
ENTITY_KIND_BY_TYPE: dict[int, str] = {
    8: "shop_item",  # BuyCard
    12: "shop_item",  # RedeemVoucher
    13: "shop_item",  # OpenBooster
    22: "shop_item",  # BuyAndUse
    9: "joker",  # SellJoker
    15: "joker",  # SwapJokersLeft
    16: "joker",  # SwapJokersRight
    10: "consumable",  # SellConsumable
    11: "consumable",  # UseConsumable
    14: "pack_card",  # PickPackCard
    17: "hand_card",  # SwapHandLeft
    18: "hand_card",  # SwapHandRight
}
CARD_SELECT_TYPES = (0, 1)  # PlayHand, Discard
NO_TARGET_TYPES = (2, 3, 4, 5, 6, 7, 19, 20, 21) # No targets: SelectBlind, SkipBlind, etc.

NEG_INF = float(torch.finfo(torch.float32).min)


class FactoredQNetwork(nn.Module):
    def __init__(self, hidden: int = 128, value_hidden: int = 64):
        super().__init__()
        self.global_encoder = nn.Sequential( # 2 layer MLP: 235 -> hidden -> hidden
            # creates a single context vector h
            nn.Linear(D_GLOBAL, hidden),  
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.type_bias_head = nn.Linear(hidden, N_ACTION_TYPES) # linear layer: hidden -> 23
        # gives action types a scalar bias conditioned on h
        self.value_heads = nn.ModuleDict(# Smaller 2 layer MLP per entity type: 
            # (hidden + entity_dim) -> value_hidden -> 1
            # Takes shared context h concatenated with one entity row and ouputs a scalar
            # for how good that card is given the current state
            {
                kind: nn.Sequential(
                    nn.Linear(hidden + dim, value_hidden),
                    nn.ReLU(),
                    nn.Linear(value_hidden, 1),
                )
                for kind, dim in ENTITY_DIMS.items()
            }
        )

    def encode(
        self, global_vec: np.ndarray, entities: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """global_vec: (235,) float32
        entities[kind]: (N_kind, D_kind) float32

        Forward pass, returns (type_bias (23,), entity_values[kind] -> (N_kind,)).
        """
        g = torch.from_numpy(global_vec).unsqueeze(0)
        h = self.global_encoder(g)  # get shape (1, hidden)
        type_bias = self.type_bias_head(h).squeeze(0)  # (23,)

        entity_values: dict[str, torch.Tensor] = {}
        for kind, arr in entities.items():
            n = arr.shape[0]
            if n == 0:
                entity_values[kind] = torch.zeros(0) # if none of this entity, record empty tensor
                continue
            arr_t = torch.from_numpy(arr)  # (n, dim)
            h_rep = h.expand(n, -1)
            entity_values[kind] = self.value_heads[kind](torch.cat([h_rep, arr_t], dim=1)).squeeze(-1) # maps entity kind to 1D tensor of scalar scores for each instance
        return type_bias, entity_values


def best_card_subset(
    card_values: torch.Tensor,
    card_mask: np.ndarray,
    min_select: int,
    max_select: int,
    forced_card: int | None,
) -> tuple[float, tuple[int, ...]]:
    """Scoring best legal subset of hand cards, sized within [min_select, max_select].
    """
    legal = [i for i in range(len(card_mask)) if card_mask[i]]
    if forced_card is not None and forced_card not in legal: # append forced cards if not already in legal
        legal.append(forced_card)
    values = card_values.detach().numpy()

    rest = [i for i in legal if i != forced_card] # non-forced legal cards
    rest.sort(key=lambda i: values[i], reverse=True) # sort by desc network value

    best_sum = NEG_INF
    best_subset: tuple[int, ...] = ()
    base = [forced_card] if forced_card is not None else []
    lo = max(min_select, len(base))
    hi = min(max_select, len(base) + len(rest))
    for k in range(lo, hi + 1): # try all legal subset sizes
        n_rest = k - len(base)
        subset = base + rest[:n_rest]
        total = sum(values[i] for i in subset)
        if total > best_sum:
            best_sum = total
            best_subset = tuple(sorted(subset))
    return best_sum, best_subset


def decide(
    type_bias: torch.Tensor,
    entity_values: dict[str, torch.Tensor],
    mask,
) -> tuple[np.ndarray, dict[int, tuple[int, ...] | int]]:
    """Q-value per action type (NEG_INF where illegal) and how to reconstruct
    each type's chosen target."""
    q = np.full(N_ACTION_TYPES, NEG_INF, dtype=np.float32)
    reconstruction: dict[int, tuple[int, ...] | int] = {}
    type_bias_np = type_bias.detach().numpy()

    for t in range(N_ACTION_TYPES):
        if not mask.type_mask[t]:
            continue
        if t in CARD_SELECT_TYPES:
            best_sum, subset = best_card_subset( 
                entity_values["hand_card"], mask.card_mask, mask.min_card_select, mask.max_card_select, mask.forced_card
            )
            q[t] = type_bias_np[t] + best_sum
            reconstruction[t] = subset
        elif t in NO_TARGET_TYPES:
            q[t] = type_bias_np[t]
        else:
            kind = ENTITY_KIND_BY_TYPE[t]
            entity_mask = mask.entity_masks.get(t)
            if entity_mask is None or not entity_mask.any():
                continue
            legal = np.nonzero(entity_mask)[0]
            vals = entity_values[kind].detach().numpy()[legal]
            best_pos = int(np.argmax(vals))
            best_entity = int(legal[best_pos])
            q[t] = type_bias_np[t] + vals[best_pos]
            reconstruction[t] = best_entity
    return q, reconstruction


def q_of_action(
    type_bias: torch.Tensor, entity_values: dict[str, torch.Tensor], action
) -> torch.Tensor:
    """Differentiable Q(s, a) for a specific FactoredAction that's already
    been taken."""
    t = action.action_type
    q = type_bias[t]
    if t in CARD_SELECT_TYPES:
        idx = torch.tensor(list(action.card_target), dtype=torch.long)
        q = q + entity_values["hand_card"][idx].sum()
    elif t in ENTITY_KIND_BY_TYPE:
        kind = ENTITY_KIND_BY_TYPE[t]
        q = q + entity_values[kind][action.entity_target]
    return q
