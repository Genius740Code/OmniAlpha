# meridian-engine

The rules of Meridian ([RULES.md](../RULES.md)) in Rust: legal moves, playing them, and scoring.
Every rule is decided with integers, so nothing is ever rounded. The site runs this engine, and so
can a bot.

## Using it

```rust
use std::error::Error;

use meridian_engine::{Game, Move, notation};

/// The legal move that gains the mover the most area, in the position a bot request describes:
/// a start position in the site's notation and the IDs of the moves played from it.
fn most_area(start: &str, moves: &[usize]) -> Result<Option<Move>, Box<dyn Error>> {
    let mut game = Game::from_position(notation::parse_setup(start)?);
    for &id in moves {
        game.play(Move::from_index(id).ok_or("not a move ID")?)?;
    }
    let position = game.position();
    let mover = position.to_move();
    // Copies are cheap: a position has no heap data.
    let area_after = |mv: Move| {
        let mut next = position.clone();
        next.apply_unchecked(mv);
        next.area(mover)
    };
    Ok(game.legal_moves().iter().max_by_key(|&mv| area_after(mv)))
}
```

| Item | What it is |
|---|---|
| `Game` | A position that keeps its legal moves and result up to date. `play` refuses an illegal move and says why. |
| `Position` | The state of a game. `to_move`, `actions_left_in_turn`, `edges`, `nodes`, `area`, `score`, `outcome`. |
| `Position::legal_moves` | The legal moves, as a `LegalMoves` set. Empty once the game is over. |
| `Position::apply`, `apply_unchecked` | Play a move. `apply` checks it first; `apply_unchecked` is for moves from `legal_moves`. Both return a `MoveOutcome`: the kind of move, the edge cut, and the areas scored if a turn ended. |
| `Position::check_move` | Whether a move is legal, and if not, which rule it breaks. |
| `Move` | An edge from a source point in one of 48 directions. `index` and `from_index` convert to and from the site's move IDs. |
| `Point`, `Edge`, `Direction` | Board points, edges and edge directions. |
| `Area`, `Score` | Exact values. `to_f64` converts them for an evaluation. |
| `notation` | The site's text for points, moves, positions and games: `parse_setup`, `setup_text`, `Line`. |

## How it works

**Points.** The board is the 19 × 19 lattice with `x` and `y` from −9 to 9. A `Point` is numbered
`(y + 9) * 19 + (x + 9)`, from 0 to 360.

**Directions and moves.** An edge joins two points at most 3 apart in `x` and in `y`, so there are 48
possible edge vectors, the `Direction`s. A `Move` is a source point and a direction, and its ID on the
site is `direction * 361 + source`, below 17,328. Some IDs leave the board; they are never legal.

**The window.** The 48 directions are the cells of the 7 × 7 box around a point, minus its centre. A
`DirSet` holds one bit per cell of that box. A `Bitboard`, a set of points with one `u32` per row and a
3-point empty margin, reads the same box around any point without bounds checks. So "which neighbours
are occupied" and "which directions are allowed" are both 49-bit masks that combine with plain bitwise
operations.

**Edges.** An edge is stored once, from its lower end (its left end when it is level), as the 16-bit
slot `origin * 24 + direction`, counting only the 24 directions that point up, or right when level. A
player's `EdgeSet` is a sorted list of slots. The nodes are exactly the ends of the edges.

**Positions.** A `Position` stores each player's edges, the number of moves played, the edges placed
in the latest turns, and both scores. The number of moves decides whose turn it is. The edges are kept
sorted, so two positions with the same edges, recent edges, move count and scores compare equal, whatever
order the edges were placed in. The edges from the opponent's last turn are *shielded*: the player to move
may not touch them (rule 6). A position also keeps each player's nodes and area, which follow from the edges.

**Move generation** (`movegen`, `tables`). For each of the mover's nodes, the legal moves are the
directions that end on the board, minus every direction a rule strikes out:

| Struck out | Rule | Found from |
|---|---|---|
| passing through one of the mover's nodes | 4 | the window of the mover's nodes, then a table |
| a new node on one of the mover's edges | 4 | the window of points inside the mover's edges |
| repeating one of the mover's edges | 4 | the directions of the source's own edges |
| touching two or more opponent edges | 5 | the contact table, counted per direction |
| touching a shielded edge | 6 | the contact table |

The tables are built once from the exact geometric tests. The contact table says how an edge at a
given offset from the source meets each of the 48 candidates, so no candidate is ever tested
geometrically. `Position::check_move` does test one move geometrically, rule by rule, and shares no
code with move generation, so each checks the other.

**Area** (`area`). A player's edges are split wherever they cross or touch, which gives a planar graph.
The walk around the outside of each connected part of the graph goes around exactly the faces that part
encloses, so the shoelace formula over the walk gives their area. An edge that encloses nothing is
walked once each way and cancels out, and a part inside another part's walk adds nothing. Edges come in
only 32 angles, so the edges around a vertex are a 32-bit mask and the next edge around it is a bit
scan. After a move, an area is recomputed only if it can have changed.

**Exact numbers** (`units`). Two edges can cross between lattice points, so an area's corners can have
fractional coordinates. The fraction always has a denominator dividing 360,360 (`CROSSING_LCM`), and
the shoelace formula halves it, so every area and score is a whole number of 1/720,720ths. `Area` and
`Score` store that whole number.

**Notation** (`notation`). A point is a column letter `A`–`S` and a row number `1`–`19`: `D10` is
(−6, 0). A move is its two points, `D10-D13`. A position is written in brackets,
`[A:12 B:A10-D10-D7,D10=D13 R:P10-S10 S:5,3.5]`: moves played, Blue's and Red's edges as paths, and
both scores. `=` marks an edge from the latest turns. Empty fields are left out, so the starting
position is `[B:A10-D10 R:P10-S10]`.

**Invariants.** After every legal move:

1. Both ends of every edge are nodes of the edge's owner, and every node has at least one edge.
2. Blue and Red edges never share a point, so no node lies on an opponent edge.
3. A player's own edges meet only at shared ends or where they cross. None runs along another, and no
   node lies inside one of its owner's edges.
4. Every edge a player placed in their last turn still exists at the start of their next one, so no
   player can be wiped out.

