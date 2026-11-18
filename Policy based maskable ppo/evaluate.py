"""Evaluate a checkpoint on a reproducible seed range and optionally record traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from env_factory import DEFAULT_SHAPED_REWARD, make_env
from action_history import PreviousActionObservation
from candidate_actions import CandidateActionObservation
from candidate_policy import CandidateActionActorCriticPolicy, GameStateFeaturesExtractor
from joker_features import JokerEffectObservation
from jackdaw.env import balatro_game_spec

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"


def _base_env(env: Any) -> Any:
    return env.unwrapped


def _joker_keys(env: Any) -> list[str]:
    game = _base_env(env).game
    return [str(card.key) for card in (getattr(game, "jokers", ()) or ())]


def _action_type_name(env: Any, action_id: int) -> str:
    try:
        action = _base_env(env)._action_table[action_id]
        return balatro_game_spec().action_types[int(action.action_type)].name
    except (AttributeError, IndexError, TypeError, ValueError):
        return "unknown"


def _card_label(card: Any) -> str:
    key = str(getattr(card, "front", None) or getattr(card, "key", "card"))
    if len(key) > 2 and key[1] == "_" and key[0] in "SHCD":
        suit = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}[key[0]]
        rank = {"T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}.get(key[2:], key[2:])
        return f"{rank}{suit}"
    return key.removeprefix("j_").removeprefix("c_").replace("_", " ").title()


def _card_rows(cards: Any) -> list[dict[str, Any]]:
    result = []
    for card in cards or ():
        ability = getattr(card, "ability", {}) or {}
        name = ability.get("name") if hasattr(ability, "get") else None
        result.append({"label": str(name or _card_label(card)), "key": str(card.key)})
    return result


def _state_frame(env: Any) -> dict[str, Any]:
    game = _base_env(env).game
    blind = game.blind
    last_score = game.last_score
    return {
        "phase": getattr(game.phase, "name", str(game.phase)).replace("_", " ").title(),
        "ante": int(game.ante),
        "round": int(game.round),
        "money": int(game.dollars),
        "chips": float(game.chips),
        "blind": str(blind.key).removeprefix("bl_").replace("_", " ").title() if blind else "None",
        "blind_target": float(blind.chips) if blind else 0,
        "hands_left": int(game.hands_left),
        "discards_left": int(game.discards_left),
        "hand": _card_rows(game.hand),
        "jokers": _card_rows(game.jokers),
        "consumables": _card_rows(game.consumables),
        "shop": _card_rows((*game.shop_jokers, *game.shop_vouchers, *game.shop_boosters)),
        "last_score": float(last_score.total) if last_score else 0,
        "won": bool(game.won),
    }


def _action_label(env: Any, action_id: int) -> str:
    base = _base_env(env)
    try:
        action = base._action_table[action_id]
        action_name = balatro_game_spec().action_types[int(action.action_type)].name.replace("_", " ").title()
        details = []
        if action.card_target:
            cards = base.game.hand
            details.append("cards " + ", ".join(
                _card_label(cards[index]) for index in action.card_target if index < len(cards)
            ))
        if action.entity_target is not None:
            details.append(f"target slot {int(action.entity_target)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"{action_name}{suffix} · action slot {action_id}"
    except (AttributeError, IndexError, TypeError, ValueError):
        return f"Action slot {action_id}"


def _write_replay_html(path: Path, seed: int, frames: list[dict[str, Any]]) -> None:
    payload = json.dumps(frames, ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Balatro agent replay — seed SEED</title>
<style>
:root{color-scheme:dark;--felt:#12634c;--gold:#f3c969;--panel:#162622;--muted:#a9c1b8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#26795e,#0d2924 75%);color:#f3f6f1;font:16px/1.45 system-ui,Segoe UI,sans-serif;min-height:100vh}
main{max-width:1100px;margin:auto;padding:28px 22px 48px}h1{margin:0 0 4px;font-size:clamp(24px,4vw,38px)}.sub{color:var(--muted);margin-bottom:20px}
.top,.zone,.controls{background:#10221fee;border:1px solid #ffffff24;border-radius:16px;box-shadow:0 12px 35px #07161266;padding:16px;margin:12px 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:10px}.stat{background:#ffffff0b;padding:10px;border-radius:10px}.stat span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}.stat b{font-size:20px;color:var(--gold)}
.zone h2{font-size:14px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 12px}.cards{display:flex;flex-wrap:wrap;gap:9px;min-height:54px}.card{min-width:58px;min-height:76px;background:linear-gradient(145deg,#fff,#e5e8e1);color:#1c2520;border:2px solid #d5d3ca;border-radius:9px;padding:8px;display:flex;align-items:center;justify-content:center;text-align:center;font-weight:750;box-shadow:0 5px 12px #0005;text-transform:capitalize}.card.joker{background:linear-gradient(145deg,#f8d889,#d49b35);border-color:#ffe8ac}.card.shop{background:linear-gradient(145deg,#cbe7da,#7cb69b);border-color:#e4fff2}.empty{color:#91aaa0;font-style:italic;padding:12px 0}
.action{font-size:19px;font-weight:700;color:#ffe5a0}.meta{color:var(--muted);font-size:13px;margin-top:5px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}button{background:#eec75e;color:#212117;border:0;border-radius:9px;padding:9px 14px;font-weight:750;cursor:pointer}input[type=range]{flex:1;min-width:200px;accent-color:#eec75e}#counter{min-width:90px;text-align:right;color:var(--muted)}
</style><main><h1>Agent run replay</h1><div class="sub">Jackdaw simulation · seed SEED · simplified card-table view</div>
<section class="top"><div class="stats" id="stats"></div></section>
<section class="zone"><h2>Hand</h2><div class="cards" id="hand"></div></section>
<section class="zone"><h2>Jokers</h2><div class="cards" id="jokers"></div></section>
<section class="zone"><h2>Consumables</h2><div class="cards" id="consumables"></div></section>
<section class="zone"><h2>Shop</h2><div class="cards" id="shop"></div></section>
<section class="top"><div class="action" id="action"></div><div class="meta" id="meta"></div></section>
<section class="controls"><button id="prev">◀ Previous</button><button id="play">▶ Play</button><button id="next">Next ▶</button><input id="slider" type="range" min="0" max="0" value="0"><span id="counter"></span></section>
</main><script>
const frames=FRAMES;let at=0,timer=null;
const $=id=>document.getElementById(id);
function text(id,value){$(id).textContent=value??''}
function cardRow(id,items,cls=''){const root=$(id);root.replaceChildren();if(!items||!items.length){const e=document.createElement('span');e.className='empty';e.textContent='None';root.append(e);return}for(const item of items){const el=document.createElement('div');el.className='card '+cls;el.textContent=item.label||item.key||'Card';el.title=item.key||'';root.append(el)}}
function render(){const f=frames[at],s=f.state;const vals=[['Ante',s.ante],['Round',s.round],['Phase',s.phase],['Blind',s.blind],['Score',Math.round(s.chips)+' / '+Math.round(s.blind_target)],['Money','$'+s.money],['Hands',s.hands_left],['Discards',s.discards_left]];const root=$('stats');root.replaceChildren();for(const [k,v] of vals){const box=document.createElement('div');box.className='stat';const label=document.createElement('span'),value=document.createElement('b');label.textContent=k;value.textContent=v;box.append(label,value);root.append(box)}cardRow('hand',s.hand);cardRow('jokers',s.jokers,'joker');cardRow('consumables',s.consumables);cardRow('shop',s.shop,'shop');text('action',f.action||'Episode start');text('meta',`Step ${f.step} · reward ${Number(f.reward||0).toFixed(3)} · legal options ${f.legal_action_count??'—'}${s.won?' · RUN WON':''}`);$('slider').value=at;text('counter',`${at+1} / ${frames.length}`);$('prev').disabled=at===0;$('next').disabled=at===frames.length-1}
function stop(){if(timer){clearInterval(timer);timer=null}$('play').textContent='▶ Play'}
$('prev').onclick=()=>{stop();at=Math.max(0,at-1);render()};$('next').onclick=()=>{stop();at=Math.min(frames.length-1,at+1);render()};$('slider').oninput=e=>{stop();at=Number(e.target.value);render()};$('play').onclick=()=>{if(timer){stop();return}$('play').textContent='❚❚ Pause';timer=setInterval(()=>{if(at>=frames.length-1){stop();return}at++;render()},850)};$('slider').max=frames.length-1;render();
</script></html>'''
    page = template.replace("SEED", str(seed)).replace("FRAMES", payload)
    path.write_text(page, encoding="utf-8")


def episode(
    env: Any,
    model: MaskablePPO,
    seed: int,
    trace_file: Any = None,
    replay_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    if replay_frames is not None:
        replay_frames.append({
            "step": 0, "action": "Episode start", "reward": 0.0,
            "legal_action_count": int(np.count_nonzero(env.action_masks())),
            "state": _state_frame(env),
        })
    total_reward = 0.0
    steps = 0
    mask_sizes: list[int] = []
    action_type_counts: Counter[str] = Counter()
    joker_acquisition_events: list[dict[str, Any]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        legal_count = int(np.count_nonzero(mask))
        if legal_count == 0:
            raise RuntimeError(f"No legal actions at seed={seed}, step={steps}")
        mask_sizes.append(legal_count)
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action_id = int(np.asarray(action).item())
        action_type = _action_type_name(env, action_id)
        action_type_counts[action_type] += 1
        action_text = _action_label(env, action_id) if replay_frames is not None or trace_file is not None else ""
        jokers_before = _joker_keys(env)
        obs, reward, terminated, truncated, info = env.step(action_id)
        jokers_after = _joker_keys(env)
        acquired_jokers = list((Counter(jokers_after) - Counter(jokers_before)).elements())
        if acquired_jokers:
            joker_acquisition_events.append({
                "step": steps,
                "action_type": action_type,
                "joker_keys": acquired_jokers,
            })
        total_reward += float(reward)
        if trace_file is not None:
            trace_file.write(json.dumps({
                "seed": seed, "step": steps, "action_id": action_id,
                "action": action_text, "action_type": action_type,
                "legal_action_count": legal_count, "reward": float(reward),
                "joker_keys_before": jokers_before,
                "joker_keys_after": jokers_after,
                "joker_acquisitions": acquired_jokers,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "info": {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))},
            }) + "\n")
        steps += 1
        if replay_frames is not None:
            replay_frames.append({
                "step": steps, "action": action_text, "reward": float(reward),
                "legal_action_count": legal_count,
                "state": _state_frame(env),
            })
    return {
        "seed": seed,
        "won": bool(info.get("balatro/won", False)),
        "ante_reached": int(info.get("balatro/ante_reached", 1)),
        "blinds_beaten": int(info.get("balatro/rounds_beaten", 0)),
        "steps": steps,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "refused": bool(info.get("balatro/refused", False)),
        "mean_legal_actions": statistics.fmean(mask_sizes) if mask_sizes else 0.0,
        "action_type_counts": dict(action_type_counts),
        "joker_acquisition_events": joker_acquisition_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--reward", choices=("blind_progress", "sparse", "shaped"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record-actions", action="store_true")
    parser.add_argument("--record-replay", action="store_true", help="Write a standalone visual HTML replay")
    parser.add_argument("--replay-seed", type=int, help="Seed to replay (defaults to the first evaluated seed)")
    parser.add_argument(
        "--no-joker-features",
        action="store_true",
        help="Use the original observation shape for checkpoints trained before Joker features were added",
    )
    parser.add_argument(
        "--no-action-history",
        action="store_true",
        help="Use the original observation shape for checkpoints trained before last-action features were added",
    )
    parser.add_argument(
        "--no-candidate-actions",
        action="store_true",
        help="Evaluate an older checkpoint that was trained without candidate-action features",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    env_cfg, eval_cfg = config["environment"], config["evaluation"]
    episodes = args.episodes or int(eval_cfg["episodes"])
    seed_start = args.seed_start if args.seed_start is not None else int(eval_cfg["seed_start"])
    if episodes < 1:
        parser.error("--episodes must be positive")
    replay_seed = args.replay_seed
    if args.record_replay and replay_seed is None:
        replay_seed = seed_start
    if replay_seed is not None and not seed_start <= replay_seed < seed_start + episodes:
        parser.error("--replay-seed must be within the evaluated seed range")
    reward_name = args.reward or eval_cfg.get("reward", "blind_progress")
    out_dir = args.output_dir or (args.checkpoint.parent / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        reward_name=reward_name,
        reward_kwargs=env_cfg.get("reward_kwargs", DEFAULT_SHAPED_REWARD),
        max_steps=int(env_cfg["max_steps"]),
        seed_prefix=f"EVAL_{seed_start}",
        max_free_moves=int(env_cfg["max_free_moves"]),
        on_refusal=env_cfg["on_refusal"],
    )
    if not args.no_joker_features:
        env = JokerEffectObservation(env)
    if not args.no_action_history:
        env = PreviousActionObservation(
            env,
            max_joker_swaps_per_blind=int(env_cfg.get("max_joker_swaps_per_blind", 5)),
        )
    if not args.no_candidate_actions:
        env = CandidateActionObservation(env)
    model = MaskablePPO.load(args.checkpoint, device="auto")
    trace_path = out_dir / "actions.jsonl" if args.record_actions else None
    replay_path = out_dir / f"replay_seed_{replay_seed}.html" if replay_seed is not None else None
    replay_frames: list[dict[str, Any]] | None = None
    rows = []
    try:
        trace_context = trace_path.open("w", encoding="utf-8") if trace_path else nullcontext(None)
        with trace_context as trace:
            for seed in range(seed_start, seed_start + episodes):
                frames = [] if seed == replay_seed else None
                row = episode(env, model, seed, trace, frames)
                rows.append(row)
                if frames is not None:
                    replay_frames = frames
                print(f"seed={seed} ante={row['ante_reached']} blinds={row['blinds_beaten']} won={row['won']} steps={row['steps']}")
    finally:
        env.close()
    if replay_path is not None and replay_frames is not None:
        _write_replay_html(replay_path, replay_seed, replay_frames)
    numeric = lambda k: statistics.fmean(float(row[k]) for row in rows)
    all_action_counts: Counter[str] = Counter()
    for row in rows:
        all_action_counts.update(row["action_type_counts"])
    joker_events = [
        {"seed": row["seed"], **event}
        for row in rows
        for event in row["joker_acquisition_events"]
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "reward": reward_name,
        "seed_start": seed_start, "episodes": episodes,
        "joker_effect_features": not args.no_joker_features,
        "previous_action_sort_feature": not args.no_action_history,
        "candidate_action_features": not args.no_candidate_actions,
        "joker_swap_order_cycle_mask": True,
        "max_joker_swaps_per_blind": int(env_cfg.get("max_joker_swaps_per_blind", 5)),
        "results": {
            "win_rate": numeric("won"), "mean_ante_reached": numeric("ante_reached"),
            "median_ante_reached": statistics.median(row["ante_reached"] for row in rows),
            "mean_blinds_beaten": numeric("blinds_beaten"), "mean_return": numeric("return"),
            "mean_episode_steps": numeric("steps"), "mean_legal_actions": numeric("mean_legal_actions"),
            "refused_actions": sum(row["refused"] for row in rows),
            "joker_acquisitions": len(joker_events),
            "joker_buy_acquisitions": sum(event["action_type"] == "buy_card" for event in joker_events),
            "joker_pack_acquisitions": sum(event["action_type"] == "pick_pack_card" for event in joker_events),
        },
        "action_type_counts": dict(all_action_counts),
        "joker_acquisition_events": joker_events,
        "episode_results": rows,
        "action_trace": str(trace_path.resolve()) if trace_path else None,
        "visual_replay": str(replay_path.resolve()) if replay_path else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["results"], indent=2))
    print(f"Saved evaluation to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
