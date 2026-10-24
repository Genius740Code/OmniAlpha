# Adding a game (for a coding agent)
> Superseded: see **docs/ADD_ANY_GAME.md** for the current fast path (scaffolder tool,
> 3 templates, family recipes, training-oriented design). The steps below remain valid.
1. `cp -r games/template games/<name>`; rename namespace `game_template` -> `game_<name>` in `game.hpp` and `reference_game.hpp`.
   (Faster: `python3 tools/new_game.py <name>` does this for you.)
2. Write `GAME_RULES.md` (rules, move numbering, outcomes, symmetries).
3. Edit `game.hpp`: `State`, constants, `legal_moves`, `apply`, `is_terminal`, `outcome`, `current_player`, `hash`, `encode`, optional `undo` and symmetries.
   Contract and conventions are documented at the top of `engine/core/api.hpp`. Do NOT touch `engine/`.
4. Edit `reference_game.hpp`: a slow, different-by-design implementation of the same rules (differential-tested).
5. `cmake -S . -B build && cmake --build build -j && ./build/validate_game --game <name>` until `VALIDATION PASSED`.
   Also run once with sanitizers (see HANDOFF.md).
6. `./build/benchmark_game --game <name>`; if `apply`/`legal_moves` are slow, optimize (bitboards, incremental updates) and re-run the validator.
7. Train (once the NN pipeline is complete): `python -m gai.orchestrator --config configs/competition_60m.yaml --game <name>`.
Common validator failures: legal moves not in `[0,kActionCount)`, `kMaxMoves` too small, `outcome` not zero-sum, `encode` not from the mover's view, forgetting to switch player, games longer than `kMaxGameLength`.
