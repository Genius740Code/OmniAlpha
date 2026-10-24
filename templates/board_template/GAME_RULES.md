# Board game template — rules
- 2 players alternate placing one stone on an empty 8x8 cell (64-bit bitboard; cell 0 = top-left).
- Move id = cell index `y*8 + x` == policy index (0..63).
- First player to own 4 stones in a row (horizontal / vertical / diagonal) wins; full board without a line = draw.
- Player 0 ('X') moves first.

## Replace for YOUR game
Adjust W/H/K constants + these hooks: `legal_moves` (your mover-set), `apply` (+ any capture/flip
side effects with their win re-check), `to_string`, and keep the move numbering == policy index
primitive in GAME_RULES (players and the NN both rely on it). See docs/ADD_ANY_GAME.md.
