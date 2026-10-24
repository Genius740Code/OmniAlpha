#!/usr/bin/env bash
# One command from "game announced" to "training is running".
#   tools/jam.sh mygame 15        # add+validate+benchmark+A/B-race, start a 15-min pilot
#   tools/jam.sh mygame 50        # the real 50-min jam training run
# Uses configs/connect4_jam.yaml flags (measured: lanes 96 / batch 48 etc.) for any game.
set -euo pipefail
GAME="${1:?usage: tools/jam.sh <game> [--from board|dice|take-away] [minutes]}"
MINS="${2:-50}"
FROM="take-away"
if [[ -n "${JAM_FROM:-}" ]]; then FROM="$JAM_FROM"; fi
CMAKE="${CMAKE:-$HOME/.local/bin/cmake}"
command -v "$CMAKE" >/dev/null 2>&1 || CMAKE=cmake

if [ ! -d "games/$GAME" ]; then
  python3 tools/new_game.py "$GAME" --from "$FROM"
  echo "== now EDIT games/$GAME/game.hpp (the TODOs) and reference_game.hpp, then re-run this command =="
  exit 0
fi
[ -f "games/$GAME/game.hpp" ]  # un-scaffolded dir = not ready for training

echo "== build =="
"$CMAKE" -S . -B build >/dev/null && "$CMAKE" --build build -j 8 2>/dev/null

echo "== validate =="
./build/validate_game --game "$GAME" | grep -q VALIDATION_PASSED \
  || ./build/validate_game --game "$GAME" | grep -q "VALIDATION PASSED" \
  || { echo "VALIDATION FAILED - fix game.hpp before training"; exit 1; }

echo "== benchmark (need legal_moves/apply > 5e6/s for the jam) =="
./build/benchmark_game --game "$GAME" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('legal/s %.2g apply/s %.2g encode/s %.2g'%(d['legal_moves_per_s'],d['apply_per_s'],d['encode_per_s']))"

echo "== bot smoke: rollout-MCTS plays (no NN needed) =="
./build/evaluate --game "$GAME" --games 10 --sims-a 400 --sims-b 100 --tc-ms 30 | python3 -c "import json,sys; d=json.load(sys.stdin); print('arena ok:', d.get('wins_a'),'/',d.get('games',d.get('n','-')))"

echo "== dry-run 2 iterations of the training timeline =="
PYTHONPATH=python python3 -m gai.orchestrator --config configs/connect4_jam.yaml --game "$GAME" --minutes "$MINS" --dry-run || true

echo "== START TRAINING (checkpoints every ~5 min; play live with: =="
echo "==   ./build-lt/play --game $GAME --p0 human --p1 mcts --model models/latest.ts --reload --tc 2+15 --solve-probe 1  ) =="
PYTHONPATH=python python3 -m gai.orchestrator --config configs/connect4_jam.yaml --game "$GAME" --minutes "$MINS" --out models
