# Dice race template — rules
- 2 players race from square 0 to square 30; exact landing required; first to land on 30 wins (no draws).
- Each turn the engine computes a die roll r in 1..6 from a public hash-chain (deterministic, but
  history-dependent so rolls differ across games — the "dice deck" trick, see game.hpp header note).
- The mover takes k = 1..r steps (move id == step count == policy index). Move id 0 means "pass" and
  is legal ONLY when every step would overshoot 30 (waiting turn; the dice still advance).
- Landing on a bonus square {7,14,21,28} slides +2; landing on the opponent's square sends them back 3.
- Player 0 moves first.

## Replace for YOUR chance game (monopoly / ludo / cards)
1. Chance (dice, card draws, spinners) -> derive from the `chain` hash (add another `mix64` round
   with a different constant per event type). Never use global RNG state inside apply().
2. Multiple decision phases per turn (roll then buy) -> extend the move id space instead, e.g.
   move id = phase-action pair, unfolding phase by phase in `apply` (see docs/ADD_ANY_GAME.md §4).
3. Play money / purchases -> track per player in State (keep it small & trivially copyable).
Verify with `./build/validate_game --game <name>` after every edit; the differential reference
(reference_game.hpp) is what catches subtle rule/side-effect bugs — keep it updated.
