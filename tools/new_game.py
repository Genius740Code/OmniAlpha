#!/usr/bin/env python3
"""Scaffold a new game in seconds:  python3 tools/new_game.py <name> [--from <template>]

Templates:
  take-away  games/template          minimal logic-only demo (fastest to finish)
  board      templates/board_template   8x8 grid, 4-in-a-row (chess/othello/gomoku family)
  dice       templates/dice_template   race w/ chance + capture (monopoly/ludo family)

After scaffolding, the ONLY work left is rewriting the logic in games/<name>/game.hpp
(plus reference_game.hpp if you want differential testing, which you do).
The engine, tools, training pipeline and NN glue are all game-agnostic and pick up
your game automatically on the next cmake run.
"""
import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATES = {
    "take-away": os.path.join(ROOT, "games", "template"),
    "board": os.path.join(ROOT, "templates", "board_template"),
    "dice": os.path.join(ROOT, "templates", "dice_template"),
}


def rename_namespaces(text, name):
    return text.replace("game_template", "game_" + name)


def scaffold(name, src, force):
    dst = os.path.join(ROOT, "games", name)
    if os.path.exists(dst):
        if not force:
            sys.exit(f"error: games/{name} already exists (use --force to overwrite)")
        shutil.rmtree(dst)
    os.makedirs(dst)
    for fn in ("game.hpp", "reference_game.hpp", "config.yaml", "GAME_RULES.md", "README.md"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            with open(p) as f:
                out = rename_namespaces(f.read(), name)
            with open(os.path.join(dst, fn), "w") as f:
                f.write(out)
    return dst


TEMPLATE_BLURB = {
    "take-away": "take-stones logic game: you edit ~10 small functions",
    "board": "grid + win-line logic game (bitboard, mirror symmetry included)",
    "dice": "chance + capture race game (deck-based dice, engine-safe pure determinism)",
}

CHEATSHEET = """
GAME SCAFFOLDED at games/{name}/  ({src})
Template: {template} -- {blurb}

NOW DO (each step is one command; ~10-30 s per loop):
  1. Edit games/{name}/game.hpp   -- the ONLY logic file. Every function has a
     TODO comment: constants, legal_moves, apply, is_terminal, outcome,
     current_player, hash, encode (+optional symmetries).
  2. Edit games/{name}/reference_game.hpp  -- slow-but-obvious same rules
     (differential oracle; copy the STATE into simple lists and loop).
  3. Re-run cmake is NOT needed if games/{name} already existed in configure;
     otherwise: cmake -S . -B build && cmake --build build -j
     (build once per session, then just `cmake --build build -j`).
  4. ./build/validate_game --game {name}      # MUST print VALIDATION PASSED
     (every time you touch game.hpp; it random-playouts + differential-tests)
  5. ./build/benchmark_game --game {name}     # legal_moves/apply speed profile
     If legal_moves or apply are < 1e6/s, a bot trained in 1 h will be weak.
     For a jam target: legal_moves > 5e6/s AND apply+hash > 5e6/s.
  6. Smoke-test a bot RIGHT NOW (no training needed):
     ./build/play --game {name} --p0 human --p1 mcts --sims 400 --tc 15
     and the arena baseline: ./build/evaluate --game {name} --games 20
  7. Train: PYTHONPATH=python python3 -m gai.orchestrator \\
        --config configs/competition_60m.yaml --game {name} --minutes 50
     Play against the improving net WHILE TRAINING (needs libtorch build, build-lt):
     ./build-lt/play --game {name} --p0 human --p1 mcts --model models/latest.ts \\
        --reload --tc 2+15     # --reload picks up every ~5-min checkpoint live

TRAPDOORS (validator prints these first):
  * moves must be 0 <= m < kActionCount and enumerate-able in ascending legal_moves
  * kMaxGameLength too small -> random playouts exceed it -> raise it
  * outcome must be zero-sum (+1 winner / -1 loser / 0 draw)
  * encode() is from the VIEW OF THE PLAYER TO MOVE
  * forget to flip current_player in apply() and MCTS plays both sides
Docs: docs/ADD_ANY_GAME.md (recipes + what trains fast), engine/core/api.hpp (contract).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="lowercase snake_case game id (namespace = game_<name>)")
    ap.add_argument("--from", dest="src", default="take-away", choices=sorted(TEMPLATES),
                    help="template family (default: take-away)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing scaffold")
    a = ap.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", a.name):
        sys.exit(f"error: game name '{a.name}' must be [a-z][a-z0-9_]{{1,30}} -- "
                 f"it becomes namespace game_{a.name} (used verbatim by generated code)")
    src = TEMPLATES[a.src]
    if not os.path.isdir(src):
        sys.exit(f"error: template {a.src} missing at {src}")
    dst = scaffold(a.name, src, a.force)
    print(CHEATSHEET.format(name=a.name, src=a.src, template=a.src, blurb=TEMPLATE_BLURB[a.src]))


if __name__ == "__main__":
    main()
