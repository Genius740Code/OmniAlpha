"""Corpus loading, the 4-fold board symmetry, and batch collation.

The packed cache (built by extract.py + the Rust dumper) holds one 442-byte
record per decision position:

    0..414   binary planes 0..8, bit-packed (46 bytes per plane, bit i%8 of byte i/8)
    414..434 scalar planes 9..13, f32 LE (scores /100, areas /100, actions /120)
    434..436 played move id, u16 LE
    436..440 final margin (blue - red), f32 LE
    440      ply (moves already played)
    441      flags (game kind | termination << 3)

The board is a 19x19 lattice, point index (y+9)*19+(x+9), and the game is
invariant under the Klein four-group {identity, horizontal mirror, 180-degree
rotation, vertical mirror}. The rotation and vertical mirror map the standard
start (Blue left, Red right) onto the color-swapped start, so they pair with a
blue/red swap: planes 0<->1, 2<->3 and 8 invert, the scalar pairs swap, the
value target negates, and the played move id is remapped. The horizontal
mirror fixes the start and needs no swap. That gives 4 exact augmentations per
position.
"""

import hashlib
import json
import os

import numpy as np

REC = 442
BOARD, POINTS, IN_PLANES = 19, 361, 14
NUM_MOVES = 48 * POINTS
VAL_PCT = 6  # must match extract.py's hash split
CLOSE = 300.0  # |final margin| <= CLOSE counts as a close game


# ---------------------------------------------------------------- symmetries

def _dir_index(dx: int, dy: int) -> int:
    """Direction index, window order, skipping the centre (geometry.rs)."""
    bit = (dy + 3) * 7 + (dx + 3)
    return bit - (1 if bit > 24 else 0)


def _transform(flip_x: bool, flip_y: bool):
    """Move-id permutation for the board transform (x,y) -> (±x, ±y)."""
    src = np.arange(POINTS, dtype=np.int64)
    col = src % BOARD
    row = src // BOARD
    col = (BOARD - 1 - col) if flip_x else col
    row = (BOARD - 1 - row) if flip_y else row
    point_map = row * BOARD + col

    dirs = []
    for d in range(48):
        wb = d + (1 if d >= 24 else 0)
        dx, dy = wb % 7 - 3, wb // 7 - 3
        dirs.append(_dir_index(-dx if flip_x else dx, -dy if flip_y else dy))
    dir_map = np.array(dirs, dtype=np.int64)

    remap = np.empty(NUM_MOVES, dtype=np.int64)
    for d, nd in enumerate(dir_map):
        remap[d * POINTS:(d + 1) * POINTS] = nd * POINTS + point_map
    return remap


# identity=0, horizontal mirror=1, 180-degree rotation=2, vertical mirror=3
MOVE_REMAP = [np.arange(NUM_MOVES), _transform(False, True),
              _transform(True, True), _transform(True, False)]
# Which transforms need the blue/red swap (those that map the standard start
# onto the color-swapped start).
SWAP = [False, False, True, True]


def _selftest() -> None:
    for t in (1, 2, 3):
        assert (MOVE_REMAP[t][MOVE_REMAP[t]] == np.arange(NUM_MOVES)).all(), "not an involution"
        assert len(set(MOVE_REMAP[t].tolist())) == NUM_MOVES, "not a permutation"
    # The start position: Blue (-9,0)-(-6,0), Red (6,0)-(9,0), fixed by the
    # horizontal mirror and mapped to itself (colors swapped) by the other two.
    start = np.zeros((9, POINTS), np.float32)
    for (x, y) in [(-9, 0), (-6, 0)]:
        start[0, (y + 9) * BOARD + (x + 9)] = 1
    for (x, y) in [(6, 0), (9, 0)]:
        start[1, (y + 9) * BOARD + (x + 9)] = 1
    board = start.reshape(9, BOARD, BOARD)
    mirrored = board[:, ::-1]  # horizontal mirror: y -> -y flips rows
    assert (mirrored == board).all(), "H must fix the start"
    rot = board[:, ::-1, ::-1]
    assert (rot[0] == board[1]).all() and (rot[1] == board[0]).all(), "R180 swaps colors"
    # A known move: Blue's opening edge from (-9,0) to (-6,0) has id 26*361 + 171.
    mv = 26 * POINTS + 171
    dx, dy = -6 - -9, 0
    assert _dir_index(dx, dy) == 26 and 171 == (0 + 9) * BOARD + (-9 + 9)
    # Under R180 it becomes S10 -> P10: source (9,0) = index 189, direction (-3,0) = 21.
    assert MOVE_REMAP[2][mv] == 21 * POINTS + 189, MOVE_REMAP[2][mv]


_selftest()


# ---------------------------------------------------------------- corpus

def sha16_of_line(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()[:16]


def load_split(out_dir: str):
    """Reads the packed cache; returns (packed records, per-position is_val).

    packed: (N, 442) uint8 — every decision position of every cached game.
    is_val: (N,) bool — a position is validation iff its game is.
    """
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    cache = os.path.join(out_dir, "cache")
    recs, is_val = [], []
    for sha, meta in sorted(manifest.items()):
        b = np.fromfile(os.path.join(cache, f"{sha}.bin"), dtype=np.uint8)
        n = b.size // REC
        recs.append(b.reshape(n, REC))
        is_val.append(np.full(n, int(sha, 16) % 100 < VAL_PCT))
    return np.concatenate(recs), np.concatenate(is_val)


def column(recs: np.ndarray, which: str):
    """Reads one packed column for every row, without unpacking the planes."""
    if which == "margin":
        return recs[:, 436:440].copy().view("<f4").reshape(-1).astype(np.float32)
    if which == "move":
        return recs[:, 434:436].copy().view("<u2").reshape(-1).astype(np.int64)
    if which == "ply":
        return recs[:, 440].astype(np.int64)
    raise KeyError(which)


def unpack(records: np.ndarray):
    """Splits packed records into per-position arrays (planes still uint8 0/1)."""
    n = records.shape[0]
    bits = np.unpackbits(records[:, :414].copy(), axis=1, bitorder="little")
    planes = bits.reshape(n, 9, 368)[:, :, :POINTS]
    scalars = records[:, 414:434].copy().view("<f4").reshape(n, 5)
    move = records[:, 434:436].copy().view("<u2").reshape(n).astype(np.int64)
    margin = records[:, 436:440].copy().view("<f4").reshape(n).astype(np.float32)
    ply = records[:, 440].astype(np.int64)
    return planes, scalars, move, margin, ply


def apply_symmetry(x, scalars, value, move, t):
    """Applies symmetry t (0..3) to a batch of positions (in-place views).

    x: (B, 14, 361) float32 C-contiguous; scalars: (B, 5); value: (B,);
    move: (B,). t may be a per-sample array of 0..3 (grouped here).
    """
    x = x.reshape(x.shape[0], IN_PLANES, BOARD, BOARD)
    for tf, sel in ((1, t == 1), (2, t == 2), (3, t == 3)):
        if not sel.any():
            continue
        x[sel] = x[sel][:, :, ::-1] if tf == 1 else (
            x[sel][:, :, ::-1, ::-1] if tf == 2 else x[sel][:, :, :, ::-1])
        # The move target follows the geometry for every transform.
        move[sel] = MOVE_REMAP[tf][move[sel]]
        if SWAP[tf]:
            sc = scalars[sel]
            sc = sc[:, [1, 0, 3, 2, 4]]
            scalars[sel] = sc
            # planes 0<->1, 2<->3 and to-move invert (4..7 hold identical marks)
            p = x[sel]  # a copy; fancy indexing does not view
            p[:, [0, 1]] = p[:, [1, 0]]
            p[:, [2, 3]] = p[:, [3, 2]]
            p[:, 8] = 1.0 - p[:, 8]
            x[sel] = p
            value[sel] = -value[sel]
    return x.reshape(x.shape[0], IN_PLANES, POINTS), scalars, value, move


def make_batch(packed, idx, scale, transforms=None):
    """Assembles one float32 training batch from packed records (augmented).

    idx: (B,) positions; transforms: optional (B,) symmetry choice, 0..3.
    Returns (x (B,14,19,19) f32, value target (B,), move target (B,)).
    """
    P = packed[idx]
    n = idx.size
    bits = np.unpackbits(P[:, :414].copy(), axis=1, bitorder="little")
    planes = bits.reshape(n, 9, 368)[:, :, :POINTS]
    scalars = P[:, 414:434].copy().view("<f4").reshape(n, 5)
    move = P[:, 434:436].copy().view("<u2").reshape(n).astype(np.int64)
    margin = P[:, 436:440].copy().view("<f4").reshape(n).astype(np.float32)

    x = np.zeros((n, IN_PLANES, POINTS), np.float32)
    x[:, :9] = planes
    val = np.tanh(margin.astype(np.float64) / scale).astype(np.float32)
    mv = move.copy()
    if transforms is not None:
        x, scalars, val, mv = apply_symmetry(x, scalars, val, mv, transforms)
    x[:, 9:, :] = scalars[:, :, None]
    return x.reshape(n, IN_PLANES, BOARD, BOARD), val, mv


def balance_weights(margin, close=CLOSE, mid=800.0, mix=(0.40, 0.30, 0.30)):
    """Per-position sampling weights: upweight close games to `mix`.

    Close games are ~22% of positions; the mix pushes them to 40% of every
    epoch (a ~3.2x relative boost), which is what fixes close-game accuracy.
    """
    am = np.abs(margin)
    w = np.full(margin.size, mix[2] / max((am > mid).mean(), 1e-9))
    w[(am <= mid) & (am > close)] = mix[1] / max(((am > close) & (am <= mid)).mean(), 1e-9)
    w[am <= close] = mix[0] / max((am <= close).mean(), 1e-9)
    return w
