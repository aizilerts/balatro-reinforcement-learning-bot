"""
Greedy Algorithm

1. Playing a Hand: Maximize immediate score using Jackdaw's exact Rust scoring
   engine (`game.preview_play`). Discard non-scoring cards when immediate play
   falls short of the blind and dig for higher scoring combinations.
2. Joker Sorting: Optimal left-to-right order (chips -> flat mult -> xMult).
3. Shop & Packs (Deterministic Heuristic):
   - Ignores Tarots, Spectral cards, and playing card modifiers (don't provide any scoring benefit).
   - Ignores Arcana, Spectral, and Standard booster packs (these provide tarots, spectral cards, and playing card modifiers).
   - Prioritizes Planets by mult increase (Neptune down to Pluto).
   - Prioritizes Jokers strictly: highest xMult -> highest Mult -> highest flat Chips.
   - Evaluates liquidation upgrades when at full Joker capacity (5/5).
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    Reroll,
    SelectBlind,
    SellCard,
    SkipPack,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
)
from jackdaw.engine.data import load
from jackdaw.env import ActionMask, ActionType, Agent, FactoredAction

logger = logging.getLogger(__name__)

# Precomputed 218 combinations of 1 to 5 cards from an 8-card hand
ALL_218_COMBOS: tuple[tuple[int, ...], ...] = tuple(
    combo for k in range(1, 6) for combo in combinations(range(8), k)
)

_RANK_CHIPS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "Jack": 10, "Queen": 10, "King": 10, "Ace": 11,
}

# Planet priority: highest mult first (Neptune down to Pluto)
_PLANET_HAND_MAP: dict[str, str] = {
    "c_pluto": "High Card",
    "c_mercury": "Pair",
    "c_uranus": "Two Pair",
    "c_venus": "Three of a Kind",
    "c_saturn": "Straight",
    "c_jupiter": "Flush",
    "c_earth": "Full House",
    "c_mars": "Four of a Kind",
    "c_neptune": "Straight Flush",
}

_PLANET_PRIORITY: dict[str, int] = {
    "c_neptune": 9,  # Straight Flush (+4 Mult)
    "c_mars": 8,     # Four of a Kind (+3 Mult)
    "c_earth": 7,    # Full House (+2 Mult)
    "c_jupiter": 6,  # Flush (+2 Mult)
    "c_saturn": 5,   # Straight (+3 Mult)
    "c_venus": 4,    # Three of a Kind (+2 Mult)
    "c_uranus": 3,   # Two Pair (+1 Mult)
    "c_mercury": 2,  # Pair (+1 Mult)
    "c_pluto": 1,    # High Card (+1 Mult)
}

# Explicit baseline values for known Jokers: (xmult, mult, chips)
_KNOWN_JOKER_STATS: dict[str, tuple[float, float, float]] = {
    # Tier 3: xMult Jokers
    "j_family": (4.0, 0.0, 0.0),
    "j_loyalty_card": (4.0, 0.0, 0.0),
    "j_cavendish": (3.0, 0.0, 0.0),
    "j_card_sharp": (3.0, 0.0, 0.0),
    "j_blackboard": (3.0, 0.0, 0.0),
    "j_order": (3.0, 0.0, 0.0),
    "j_trio": (3.0, 0.0, 0.0),
    "j_acrobat": (3.0, 0.0, 0.0),
    "j_drivers_license": (3.0, 0.0, 0.0),
    "j_flower_pot": (3.0, 0.0, 0.0),
    "j_duo": (2.0, 0.0, 0.0),
    "j_tribe": (2.0, 0.0, 0.0),
    "j_ramen": (2.0, 0.0, 0.0),
    "j_photograph": (2.0, 0.0, 0.0),
    "j_idol": (2.0, 0.0, 0.0),
    "j_triboulet": (2.0, 0.0, 0.0),
    "j_seeing_double": (2.0, 0.0, 0.0),
    "j_caino": (2.0, 0.0, 0.0),
    "j_stencil": (2.0, 0.0, 0.0),
    "j_ancient": (1.5, 0.0, 0.0),
    "j_baron": (1.5, 0.0, 0.0),
    "j_baseball": (1.5, 0.0, 0.0),
    "j_bloodstone": (1.5, 0.0, 0.0),
    "j_yorick": (1.5, 0.0, 0.0),
    "j_glass": (1.5, 0.0, 0.0),
    "j_madness": (1.5, 0.0, 0.0),
    "j_campfire": (1.5, 0.0, 0.0),
    "j_throwback": (1.5, 0.0, 0.0),
    "j_steel_joker": (1.4, 0.0, 0.0),
    "j_constellation": (1.3, 0.0, 0.0),
    "j_hologram": (1.3, 0.0, 0.0),
    "j_lucky_cat": (1.3, 0.0, 0.0),
    "j_vampire": (1.3, 0.0, 0.0),
    "j_obelisk": (1.2, 0.0, 0.0),
    "j_hit_the_road": (1.2, 0.0, 0.0),

    # Tier 2: Flat Mult Jokers
    "j_popcorn": (1.0, 20.0, 0.0),
    "j_half": (1.0, 20.0, 0.0),
    "j_gros_michel": (1.0, 15.0, 0.0),
    "j_mystic_summit": (1.0, 15.0, 0.0),
    "j_shoot_the_moon": (1.0, 13.0, 0.0),
    "j_crazy": (1.0, 12.0, 0.0),
    "j_zany": (1.0, 12.0, 0.0),
    "j_misprint": (1.0, 11.5, 0.0),
    "j_mad": (1.0, 10.0, 0.0),
    "j_droll": (1.0, 10.0, 0.0),
    "j_fibonacci": (1.0, 8.0, 0.0),
    "j_jolly": (1.0, 8.0, 0.0),
    "j_onyx_agate": (1.0, 7.0, 0.0),
    "j_smiley": (1.0, 5.0, 0.0),
    "j_even_steven": (1.0, 4.0, 0.0),
    "j_joker": (1.0, 4.0, 0.0),
    "j_erosion": (1.0, 4.0, 0.0),
    "j_scholar": (1.0, 4.0, 20.0),
    "j_walkie_talkie": (1.0, 4.0, 10.0),
    "j_abstract": (1.0, 9.0, 0.0),
    "j_gluttenous_joker": (1.0, 3.0, 0.0),
    "j_greedy_joker": (1.0, 3.0, 0.0),
    "j_lusty_joker": (1.0, 3.0, 0.0),
    "j_wrathful_joker": (1.0, 3.0, 0.0),
    "j_red_card": (1.0, 3.0, 0.0),
    "j_bootstraps": (1.0, 2.0, 0.0),
    "j_trousers": (1.0, 2.0, 0.0),
    "j_ride_the_bus": (1.0, 2.0, 0.0),
    "j_green_joker": (1.0, 2.0, 0.0),
    "j_fortune_teller": (1.0, 2.0, 0.0),
    "j_flash": (1.0, 2.0, 0.0),
    "j_swashbuckler": (1.0, 1.0, 0.0),
    "j_supernova": (1.0, 1.0, 0.0),
    "j_ceremonial": (1.0, 1.0, 0.0),

    # Tier 1: Flat Chip Jokers
    "j_stuntman": (1.0, 0.0, 250.0),
    "j_ice_cream": (1.0, 0.0, 100.0),
    "j_devious": (1.0, 0.0, 100.0),
    "j_wily": (1.0, 0.0, 100.0),
    "j_clever": (1.0, 0.0, 80.0),
    "j_crafty": (1.0, 0.0, 80.0),
    "j_arrowhead": (1.0, 0.0, 50.0),
    "j_sly": (1.0, 0.0, 50.0),
    "j_odd_todd": (1.0, 0.0, 31.0),
    "j_banner": (1.0, 0.0, 30.0),
    "j_scary_face": (1.0, 0.0, 30.0),
    "j_stone": (1.0, 0.0, 25.0),
    "j_blue_joker": (1.0, 0.0, 30.0),
    "j_bull": (1.0, 0.0, 20.0),
    "j_runner": (1.0, 0.0, 15.0),
    "j_castle": (1.0, 0.0, 15.0),
    "j_wee": (1.0, 0.0, 8.0),
    "j_hiker": (1.0, 0.0, 5.0),
    "j_square": (1.0, 0.0, 4.0),
}

_CENTERS: dict[str, Any] = load("centers")


def _get_joker_stats(card_or_key: Any, edition: str | None = None) -> tuple[float, float, float]:
    """Return (xmult, mult, chips) for a joker. Lexicographically comparable:
    (highest xmult, then highest mult, then highest flat chips).
    """
    key = card_or_key if isinstance(card_or_key, str) else getattr(card_or_key, "key", "")
    card_edition = edition if edition is not None else getattr(card_or_key, "edition", None)

    if key in _KNOWN_JOKER_STATS:
        xm, m, c = _KNOWN_JOKER_STATS[key]
    else:
        # Fallback dynamic parsing from centers table
        center = _CENTERS.get(key, {})
        cfg = center.get("config", {}) if isinstance(center.get("config"), dict) else {}
        extra = cfg.get("extra") if isinstance(cfg, dict) else None

        xm, m, c = 1.0, 0.0, 0.0
        if "Xmult" in cfg: xm = max(xm, float(cfg["Xmult"]))
        if "x_mult" in cfg: xm = max(xm, float(cfg["x_mult"]))
        if "mult" in cfg: m = max(m, float(cfg["mult"]))
        if "t_mult" in cfg: m = max(m, float(cfg["t_mult"]))
        if "chips" in cfg: c = max(c, float(cfg["chips"]))
        if "t_chips" in cfg: c = max(c, float(cfg["t_chips"]))

        if isinstance(extra, dict):
            if "Xmult" in extra: xm = max(xm, float(extra["Xmult"]))
            if "x_mult" in extra: xm = max(xm, float(extra["x_mult"]))
            if "mult" in extra: m = max(m, float(extra["mult"]))
            if "s_mult" in extra: m = max(m, float(extra["s_mult"]))
            if "chips" in extra: c = max(c, float(extra["chips"]))

    # Edition bonuses
    if card_edition == "polychrome":
        xm *= 1.5
    elif card_edition == "holo":
        m += 10.0
    elif card_edition == "foil":
        c += 50.0

    return (xm, m, c)


def _get_joker_tier(card: Any) -> int:
    """Left-to-right positioning tier:
    Tier 0: Utility / flat chips
    Tier 1: Flat mult
    Tier 2: xMult
    """
    xm, m, c = _get_joker_stats(card)
    if xm > 1.0:
        return 2
    if m > 0.0:
        return 1
    return 0


def _find_best_current_hand(
    game: Any, action_mask: ActionMask | None = None
) -> tuple[tuple[int, ...], float, str]:
    """Find the highest-scoring combination from the cards currently in hand."""
    forced = action_mask.forced_card if action_mask else None
    n_cards = len(game.hand)

    best_score = -1.0
    best_combo: tuple[int, ...] = (0,) if n_cards > 0 else ()
    best_type = "High Card"

    if forced is None and n_cards == 8:
        for combo in ALL_218_COMBOS:
            score = game.preview_play(combo)
            if score.total > best_score:
                best_score = float(score.total)
                best_combo = combo
                best_type = score.hand_type
        return best_combo, best_score, best_type

    # Fallback for non-8 hand size or forced card
    for k in range(1, min(6, n_cards + 1)):
        for combo in combinations(range(n_cards), k):
            if forced is not None and forced not in combo:
                continue
            score = game.preview_play(combo)
            if score.total > best_score:
                best_score = float(score.total)
                best_combo = combo
                best_type = score.hand_type

    return best_combo, best_score, best_type


def _select_cards_to_discard(
    game: Any, keep_set: set[int], action_mask: ActionMask | None = None
) -> tuple[int, ...]:
    """Choose up to 5 cards to discard, prioritizing un-kept, lowest-rank cards."""
    n = len(game.hand)
    forced = action_mask.forced_card if action_mask else None

    candidates = [i for i in range(n) if i not in keep_set]
    if not candidates:
        candidates = list(range(n))

    candidates.sort(key=lambda i: _RANK_CHIPS.get(game.hand[i].rank, 0))
    discard = candidates[:5]

    if forced is not None and forced not in discard:
        discard = [forced] + [i for i in discard if i != forced][:4]

    return tuple(sorted(discard))


class GreedyAgent(Agent):
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._in_shop = False

    def reset(self) -> None:
        self._in_shop = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)

    def act(
        self,
        obs: dict[str, Any],
        action_mask: ActionMask,
        info: dict[str, Any] | None = None,
    ) -> FactoredAction:
        """Select the greedy optimal action for the current phase."""
        game = info.get("game") if info else None
        if game is None:
            return self._fallback_action(action_mask)

        phase = game.phase
        if phase == GamePhase.SELECTING_HAND:
            return self._handle_selecting_hand(game, action_mask)
        elif phase == GamePhase.SHOP:
            return self._handle_shop(game, action_mask)
        elif phase == GamePhase.ROUND_EVAL:
            return self._handle_round_eval(game, action_mask)
        elif phase == GamePhase.BLIND_SELECT:
            return self._handle_blind_select(game, action_mask)
        elif phase == GamePhase.PACK_OPENING:
            return self._handle_pack_opening(game, action_mask)

        return self._fallback_action(action_mask)

    def _fallback_action(self, action_mask: ActionMask) -> FactoredAction:
        for at in ActionType:
            if action_mask.type_mask[at]:
                legal = action_mask.entity_masks.get(at)
                target = None
                if legal:
                    for i, is_leg in enumerate(legal):
                        if is_leg:
                            target = i
                            break
                return FactoredAction(action_type=at, entity_target=target)
        return FactoredAction(action_type=ActionType.SelectBlind)

   
    # Blind Selection: Always play blind
    def _handle_blind_select(self, game: Any, action_mask: ActionMask) -> FactoredAction:
        if action_mask.type_mask[ActionType.SelectBlind]:
            self._log(f"[BlindSelect] Entering {game.blind.key} (Target: {game.blind.chips:,.0f} chips)")
            return FactoredAction(action_type=ActionType.SelectBlind)
        return self._fallback_action(action_mask)


    # Hand Selection
    def _handle_selecting_hand(self, game: Any, action_mask: ActionMask) -> FactoredAction:
        # 1. Use any held planet cards immediately. Always uses consumables for immediate value.
        planet_act = self._try_use_planet_in_hand(game, action_mask)
        if planet_act is not None:
            return planet_act

        # 2. Keep Jokers sorted into optimal tiers (chips -> mult -> xmult)
        sort_act = self._optimize_joker_positions(game, action_mask)
        if sort_act is not None:
            return sort_act

        # 3. Find the best scoring combo currently in hand
        best_combo, best_score, best_type = _find_best_current_hand(game, action_mask)
        remaining_needed = max(0.0, game.blind.chips - game.chips)

        self._log(
            f"[SelectingHand] Best play: {best_type} {best_combo} -> {best_score:,.0f} pts "
            f"(Target remaining: {remaining_needed:,.0f}, Discards: {game.discards_left})"
        )

        # 4. Immediate win condition: play if it clears the blind or no discards left
        if best_score >= remaining_needed or game.discards_left == 0:
            self._log(f"  ▶ RULE: Clears blind ({best_score:,.0f} >= {remaining_needed:,.0f}) or 0 discards -> PLAY")
            return FactoredAction(action_type=ActionType.PlayHand, card_target=best_combo)

        # 5. Discard logic: if hand score is weak relative to target, discard non-essential cards
        if action_mask.type_mask[ActionType.Discard] and game.discards_left > 0:
            if best_score >= 0.5 * remaining_needed:
                self._log(f"  ▶ RULE: Score is substantial ({best_score:,.0f} >= 50% of target) -> PLAY")
                return FactoredAction(action_type=ActionType.PlayHand, card_target=best_combo)

            discard_indices = _select_cards_to_discard(game, set(best_combo), action_mask)
            if discard_indices:
                self._log(f"  ▶ RULE: Low score ({best_score:,.0f}) -> DISCARD cards {discard_indices}")
                return FactoredAction(action_type=ActionType.Discard, card_target=discard_indices)

        return FactoredAction(action_type=ActionType.PlayHand, card_target=best_combo)

    def _try_use_planet_in_hand(self, game: Any, action_mask: ActionMask) -> FactoredAction | None:
        """Use held Planet cards immediately for level-up."""
        if not action_mask.type_mask[ActionType.UseConsumable]:
            return None

        legal = action_mask.entity_masks.get(ActionType.UseConsumable)
        for i, consumable in enumerate(game.consumables):
            if legal is not None and i < len(legal) and not legal[i]:
                continue
            if consumable.key in _PLANET_HAND_MAP:
                self._log(f"  [Auto-Use] Using Planet {consumable.key} for immediate level-up")
                return FactoredAction(action_type=ActionType.UseConsumable, entity_target=i)
        return None

    def _optimize_joker_positions(self, game: Any, action_mask: ActionMask) -> FactoredAction | None:
        """Keep Jokers ordered left-to-right by tier (ascending)."""
        jokers = game.jokers
        if len(jokers) < 2:
            return None

        tiers = [_get_joker_tier(j) for j in jokers]
        for i in range(len(tiers) - 1):
            if tiers[i] > tiers[i + 1]:
                if action_mask.type_mask[ActionType.SwapJokersRight]:
                    legal = action_mask.entity_masks.get(ActionType.SwapJokersRight)
                    if legal is None or (i < len(legal) and legal[i]):
                        self._log(f"  [Joker Sort] Swapping {jokers[i].key} right past {jokers[i+1].key}")
                        return FactoredAction(action_type=ActionType.SwapJokersRight, entity_target=i)

                if action_mask.type_mask[ActionType.SwapJokersLeft]:
                    legal = action_mask.entity_masks.get(ActionType.SwapJokersLeft)
                    if legal is None or (i + 1 < len(legal) and legal[i + 1]):
                        self._log(f"  [Joker Sort] Swapping {jokers[i+1].key} left past {jokers[i].key}")
                        return FactoredAction(action_type=ActionType.SwapJokersLeft, entity_target=i + 1)
        return None

    
    # Cashout and see how much money we have after the round.
    def _handle_round_eval(self, game: Any, action_mask: ActionMask) -> FactoredAction:
        self._in_shop = False
        if action_mask.type_mask[ActionType.CashOut]:
            self._log(f"[RoundEval] CashOut (Bank before round reward: ${game.dollars})")
            return FactoredAction(action_type=ActionType.CashOut)
        return self._fallback_action(action_mask)

    # Shop Phase
    def _handle_shop(self, game: Any, action_mask: ActionMask) -> FactoredAction:
        if not self._in_shop:
            self._log(f"[Shop] Entering shop (Total Bank: ${game.dollars})")
            self._in_shop = True

        # 1. Sort Jokers first
        sort_act = self._optimize_joker_positions(game, action_mask)
        if sort_act is not None:
            return sort_act

        # 2. Use any held planet cards immediately in shop
        planet_act = self._try_use_planet_in_hand(game, action_mask)
        if planet_act is not None:
            return planet_act

        # 3. Evaluate deterministic purchases (Planets & Jokers)
        buy_act = self._evaluate_shop_purchases(game, action_mask)
        if buy_act is not None:
            return buy_act

        # 4. Open booster packs (Buffoon and Celestial packs only)
        booster_act = self._try_open_booster(game, action_mask)
        if booster_act is not None:
            return booster_act

        # 5. Reroll if shop items exhausted and player can afford reroll + minimum card cost ($4)
        if action_mask.type_mask[ActionType.Reroll]:
            reroll_cost = getattr(game, "reroll_cost", 5)
            if game.dollars >= reroll_cost + 4:
                self._log(f"[Shop] Rerolling shop (Cost: ${reroll_cost}, Dollars: ${game.dollars})")
                return FactoredAction(action_type=ActionType.Reroll)

        # 6. Exit shop to next round
        if action_mask.type_mask[ActionType.NextRound]:
            self._in_shop = False
            self._log(f"[Shop] Exiting shop to next round (Bank: ${game.dollars})")
            return FactoredAction(action_type=ActionType.NextRound)

        return self._fallback_action(action_mask)

    def _evaluate_shop_purchases(self, game: Any, action_mask: ActionMask) -> FactoredAction | None:
        """Deterministic evaluation of shop items:
        - Completely ignores Tarots, Spectrals, and playing card modifiers.
        - Prioritizes Planets by mult increase (Neptune -> Pluto).
        - Prioritizes Jokers by highest xMult -> highest Mult -> highest flat Chips.
        """
        if not (action_mask.type_mask[ActionType.BuyCard] or action_mask.type_mask[ActionType.BuyAndUse]):
            return None

        legal_buy = action_mask.entity_masks.get(ActionType.BuyCard)
        legal_use = action_mask.entity_masks.get(ActionType.BuyAndUse)

        # Collect affordable, qualified candidates
        best_cand: dict[str, Any] | None = None
        best_priority: tuple[int, ...] = (-1,)

        for i, card in enumerate(game.shop_jokers):
            can_buy = legal_buy is not None and i < len(legal_buy) and legal_buy[i]
            can_use = legal_use is not None and i < len(legal_use) and legal_use[i]
            if not (can_buy or can_use):
                continue

            cost = card.cost
            key = card.key

            # Ignore Tarots, Spectral cards, and playing card modifiers
            if key.startswith("c_") and key not in _PLANET_PRIORITY:
                continue
            if key.startswith("m_") or key == "c_base":
                continue

            # Case A: Planet card
            if key in _PLANET_PRIORITY:
                if game.dollars >= cost and (len(game.consumables) < 2 or can_use):
                    p_rank = _PLANET_PRIORITY[key]
                    # Planet priority tuple: Tier 2 (value: 2, planet_rank)
                    priority = (2, p_rank, 0, 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_cand = {
                            "type": "buy_card",
                            "idx": i,
                            "use": can_use,
                            "key": key,
                            "desc": f"Planet {key} (Rank {p_rank})",
                        }

            # Case B: Joker
            elif key.startswith("j_"):
                stats = _get_joker_stats(card)
                xm, m, c = stats

                # Only buy if it provides tangible scoring benefits
                if xm <= 1.0 and m <= 0.0 and c <= 0.0:
                    continue

                # If open joker slots available (< 5)
                if len(game.jokers) < 5:
                    if game.dollars >= cost:
                        # Joker priority tuple: Tier 3 for xMult, Tier 2 for Mult, Tier 1 for Chips
                        tier = 3 if xm > 1.0 else (2 if m > 0.0 else 1)
                        priority = (tier, int(xm * 100), int(m), int(c))
                        if priority > best_priority:
                            best_priority = priority
                            best_cand = {
                                "type": "buy_card",
                                "idx": i,
                                "use": False,
                                "key": key,
                                "desc": f"Joker {key} (xMult={xm:.1f}, Mult={m:.0f}, Chips={c:.0f})",
                            }

                # If 5/5 slots: evaluate liquidation upgrade
                elif len(game.jokers) >= 5:
                    # Find weakest held joker
                    held_stats = [_get_joker_stats(j) for j in game.jokers]
                    min_stats = min(held_stats)
                    weakest_idx = held_stats.index(min_stats)
                    weakest_joker = game.jokers[weakest_idx]

                    # Strictly better candidate
                    if stats > min_stats and game.dollars + weakest_joker.sell_cost >= cost:
                        self._log(
                            f"  ▶ LIQUIDATION UPGRADE: Sell {weakest_joker.key} {min_stats} "
                            f"for ${weakest_joker.sell_cost} to buy {key} {stats}"
                        )
                        return FactoredAction(action_type=ActionType.SellJoker, entity_target=weakest_idx)

        if best_cand is not None:
            idx = best_cand["idx"]
            if best_cand["use"]:
                self._log(f"  ▶ BUY & USE {best_cand['desc']}")
                return FactoredAction(action_type=ActionType.BuyAndUse, entity_target=idx)
            else:
                self._log(f"  ▶ BUY CARD {best_cand['desc']}")
                return FactoredAction(action_type=ActionType.BuyCard, entity_target=idx)

        return None

    def _try_open_booster(self, game: Any, action_mask: ActionMask) -> FactoredAction | None:
        """Open affordable Buffoon or Celestial packs only. Completely ignores Arcana, Spectral, and Standard."""
        if not action_mask.type_mask[ActionType.OpenBooster]:
            return None

        legal = action_mask.entity_masks.get(ActionType.OpenBooster)
        best_idx = None
        best_priority = -1

        for i, booster in enumerate(game.shop_boosters):
            if legal is not None and i < len(legal) and not legal[i]:
                continue

            key = booster.key
            # Ignore Arcana, Spectral, and Standard packs
            if "arcana" in key or "spectral" in key or "standard" in key:
                continue

            if game.dollars >= booster.cost:
                # Prioritize Buffoon packs if we need jokers (< 5), else Celestial
                if "buffoon" in key:
                    prio = 20 if len(game.jokers) < 5 else 5
                elif "celestial" in key:
                    prio = 10
                else:
                    prio = 1

                if prio > best_priority:
                    best_priority = prio
                    best_idx = i

        if best_idx is not None:
            booster = game.shop_boosters[best_idx]
            self._log(f"  ▶ OPEN BOOSTER #{best_idx} {booster.key} (Cost: ${booster.cost})")
            return FactoredAction(action_type=ActionType.OpenBooster, entity_target=best_idx)

        return None

    
    # Pack Opening 
    def _handle_pack_opening(self, game: Any, action_mask: ActionMask) -> FactoredAction:
        """Deterministic card picking inside booster packs:
        - In Buffoon packs: highest xMult -> Mult -> Chips.
        - In Celestial packs: Neptune down to Pluto.
        - Skip or ignore all others.
        """
        if not action_mask.type_mask[ActionType.PickPackCard]:
            if action_mask.type_mask[ActionType.SkipPack]:
                return FactoredAction(action_type=ActionType.SkipPack)
            return self._fallback_action(action_mask)

        legal = action_mask.entity_masks.get(ActionType.PickPackCard)
        pack_cards = getattr(game, "pack_cards", ())

        best_idx = None
        best_priority: tuple[int, ...] = (-1,)

        for i, card in enumerate(pack_cards):
            if legal is not None and i < len(legal) and not legal[i]:
                continue

            key = card.key

            # In Celestial pack (Planets)
            if key in _PLANET_PRIORITY:
                p_rank = _PLANET_PRIORITY[key]
                priority = (2, p_rank, 0, 0)
                if priority > best_priority:
                    best_priority = priority
                    best_idx = i

            # In Buffoon pack (Jokers)
            elif key.startswith("j_"):
                xm, m, c = _get_joker_stats(card)
                tier = 3 if xm > 1.0 else (2 if m > 0.0 else 1)
                priority = (tier, int(xm * 100), int(m), int(c))
                if priority > best_priority:
                    best_priority = priority
                    best_idx = i

        if best_idx is not None:
            self._log(f"  ▶ PICK PACK CARD #{best_idx} {pack_cards[best_idx].key}")
            return FactoredAction(action_type=ActionType.PickPackCard, entity_target=best_idx)

        if action_mask.type_mask[ActionType.SkipPack]:
            self._log("  ▶ SKIP PACK (No qualified Planets or Jokers)")
            return FactoredAction(action_type=ActionType.SkipPack)

        return FactoredAction(action_type=ActionType.PickPackCard, entity_target=0)
