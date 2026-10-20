//! Pure-Rust inference for the supervised Meridian value/policy CNN, and the
//! input encoding it was trained on. No `unsafe`, no allocations after the
//! first call, WASM-safe (`wasm32-wasip1`).
//!
//! # The input encoding (must match the data builder exactly)
//!
//! 14 planes of 19x19 = 361 points, point index `(y + 9) * 19 + (x + 9)`:
//!
//! | plane | contents |
//! |---|---|
//! | 0, 1 | Blue's and Red's node points |
//! | 2, 3 | Blue's and Red's edge points (endpoints and interior lattice points) |
//! | 4, 5 | Blue's and Red's fresh edges (placed this turn; shielded next turn) |
//! | 6, 7 | Blue's and Red's shielded edges (opponent placed them last turn) |
//! | 8 | all ones when Blue is to move, all zeros otherwise |
//! | 9..14 | constant planes: blue score/100, red score/100, blue area/100, red area/100, actions played/120 |
//!
//! Targets during training: value = `tanh((blue - red final score) / 1500)`,
//! policy = the move played, as a move ID (`direction * 361 + source index`),
//! which is exactly channel-major flattening of the policy head output.

use std::sync::OnceLock;

use meridian_engine::{Move, Player, Position};

use crate::weights;

pub const BOARD: usize = 19;
pub const POINTS: usize = BOARD * BOARD;
pub const IN_PLANES: usize = 14;
/// The value head output is `tanh(final score margin / this)`, Blue's lead positive.
pub const MARGIN_SCALE: f64 = 1500.0;
const SCORE_SCALE: f64 = 100.0;

/// Trunk channels of the CNN.
const CH: usize = 32;
/// Residual blocks in the CNN.
const BLOCKS: usize = 4;
/// Policy-head channels: one per move direction, so flatten == move ID.
const PCH: usize = 48;
/// Value-head channels.
const VCH: usize = 16;

/// One evaluation of a position.
pub struct Eval {
    /// `tanh((blue - red final score) / 1500)`: positive favours Blue.
    pub value: f64,
    /// The best `k` legal moves, best first, with log-probabilities (log softmax).
    pub moves: Vec<(Move, f64)>,
}

/// The value of `position` on the trained scale, Blue's perspective.
///
/// Positive favours Blue. `atanh(value) * 1500` is roughly the final point
/// margin the net expects; the bounded `value` itself is what comparisons and
/// search can use directly. Skips the policy head.
pub fn eval_value(position: &Position) -> f64 {
    let mut input = [0f32; IN_PLANES * POINTS];
    input_planes(position, &mut input);
    forward(&input, false).0
}

/// The full evaluation: value plus the top-8 legal moves with log-probabilities.
pub fn eval(position: &Position) -> Eval {
    let (value, logits) = eval_logits(position);
    let logits = logits.expect("policy was requested");
    let moves = top_moves(position, &logits, 8);
    Eval { value, moves }
}

/// The value and the raw policy logits (17,328 move IDs, before legality masking).
pub fn eval_logits(position: &Position) -> (f64, Option<[f32; meridian_engine::NUM_MOVES]>) {
    let mut input = [0f32; IN_PLANES * POINTS];
    input_planes(position, &mut input);
    forward(&input, true)
}

/// The top `k` legal moves by the policy head, best first, as log-probabilities.
pub fn top_moves(position: &Position, logits: &[f32; meridian_engine::NUM_MOVES], k: usize) -> Vec<(Move, f64)> {
    let legal = position.legal_moves();
    let mut masked = [f32::NEG_INFINITY; meridian_engine::NUM_MOVES];
    let mut max_logit = f32::NEG_INFINITY;
    for id in 0..meridian_engine::NUM_MOVES {
        if let Some(mv) = Move::from_index(id) {
            if legal.contains(mv) {
                masked[id] = logits[id];
                max_logit = max_logit.max(logits[id]);
            }
        }
    }
    let mut sum = 0f64;
    for &l in masked.iter() {
        if l > f32::NEG_INFINITY {
            sum += f64::from((l - max_logit).exp());
        }
    }
    let log_sum = sum.ln() + f64::from(max_logit);

    let mut moves = Vec::with_capacity(k);
    for _ in 0..k {
        let mut top: Option<(usize, f32)> = None;
        for (id, &l) in masked.iter().enumerate() {
            if l > f32::NEG_INFINITY && top.is_none_or(|(_, bl)| l > bl) {
                top = Some((id, l));
            }
        }
        match top {
            Some((id, l)) => {
                masked[id] = f32::NEG_INFINITY;
                moves.push((Move::from_index(id).expect("legal ids decode"), f64::from(l) - log_sum));
            }
            None => break,
        }
    }
    moves
}

/// Writes the 14 training planes of `position` into `out`.
pub fn input_planes(position: &Position, out: &mut [f32; IN_PLANES * POINTS]) {
    out.fill(0.0);
    let idx = |x: i8, y: i8| (y + 9) as usize * BOARD + (x + 9) as usize;
    let mut set = |plane: usize, x: i8, y: i8| out[plane * POINTS + idx(x, y)] = 1.0;

    for &(player, plane) in &[(Player::Blue, 0usize), (Player::Red, 1)] {
        for point in position.nodes(player).iter() {
            set(plane, point.x(), point.y());
        }
    }
    for &(player, plane) in &[(Player::Blue, 2usize), (Player::Red, 3)] {
        for edge in position.edges(player).iter() {
            set_edge(&mut set, plane, edge);
        }
    }
    // The engine reports fresh/shielded edges mover-relative (there is no
    // per-player split), so both players' planes get the same marks; plane 8
    // (the side to move) disambiguates ownership for the net.
    for plane in [4usize, 5] {
        for edge in position.fresh_edges() {
            set_edge(&mut set, plane, edge);
        }
    }
    for plane in [6usize, 7] {
        for edge in position.shielded_edges() {
            set_edge(&mut set, plane, edge);
        }
    }
    if position.to_move() == Player::Blue {
        for v in out[8 * POINTS..9 * POINTS].iter_mut() {
            *v = 1.0;
        }
    }
    // Constant planes carrying the scalars.
    let scalars = [
        position.score(Player::Blue).to_f64() / SCORE_SCALE,
        position.score(Player::Red).to_f64() / SCORE_SCALE,
        position.area(Player::Blue).to_f64() / SCORE_SCALE,
        position.area(Player::Red).to_f64() / SCORE_SCALE,
        f64::from(position.actions_played()) / 120.0,
    ];
    for (plane, s) in scalars.into_iter().enumerate() {
        let v = s as f32;
        for value in out[(9 + plane) * POINTS..(10 + plane) * POINTS].iter_mut() {
            *value = v;
        }
    }
}

fn set_edge(set: &mut impl FnMut(usize, i8, i8), plane: usize, edge: meridian_engine::Edge) {
    set(plane, edge.origin().x(), edge.origin().y());
    set(plane, edge.far().x(), edge.far().y());
    for point in edge.interior_points() {
        set(plane, point.x(), point.y());
    }
}

// The network, decoded once from `weights` and kept for the process's lifetime.

struct Block {
    c1_w: Vec<f32>,
    c1_b: Vec<f32>,
    c2_w: Vec<f32>,
    c2_b: Vec<f32>,
}

struct Net {
    stem_w: Vec<f32>,
    stem_b: Vec<f32>,
    blocks: Vec<Block>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    v_fc_w: Vec<f32>,
    v_fc_b: Vec<f32>,
    p1_w: Vec<f32>,
    p1_b: Vec<f32>,
    p2_w: Vec<f32>,
    p2_b: Vec<f32>,
}

impl Net {
    fn decode() -> Net {
        let mut at = 0usize;
        let mut floats: Vec<f32> = weights::BYTES
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(floats.len(), weights::LAYOUT.iter().sum::<usize>(), "weights truncated");
        let mut take = |n: usize| {
            let slice = floats[at..at + n].to_vec();
            at += n;
            slice
        };
        let stem_w = take(CH * IN_PLANES * 25);
        let stem_b = take(CH);
        let mut blocks = Vec::with_capacity(BLOCKS);
        for _ in 0..BLOCKS {
            blocks.push(Block {
                c1_w: take(CH * CH * 9),
                c1_b: take(CH),
                c2_w: take(CH * CH * 9),
                c2_b: take(CH),
            });
        }
        let net = Net {
            stem_w,
            stem_b,
            blocks,
            v_w: take(VCH * CH),
            v_b: take(VCH),
            v_fc_w: take(VCH),
            v_fc_b: take(1),
            p1_w: take(PCH * CH * 9),
            p1_b: take(PCH),
            p2_w: take(PCH * PCH),
            p2_b: take(PCH),
        };
        assert_eq!(at, floats.len(), "weights have trailing data");
        floats.clear();
        net
    }
}

static NET: OnceLock<Net> = OnceLock::new();

fn net() -> &'static Net {
    NET.get_or_init(Net::decode)
}

// Reused scratch buffers, so steady-state inference allocates nothing.
thread_local! {
    static TRUNK: std::cell::RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    static POLICY: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// One forward pass. Returns the value and, when `policy` is set, the move logits.
fn forward(input: &[f32; IN_PLANES * POINTS], policy: bool) -> (f64, Option<[f32; meridian_engine::NUM_MOVES]>) {
    let net = net();
    let mut logits_out = None;
    let value = TRUNK.with_borrow_mut(|(t0, t1, t2)| {
        t0.resize(CH * POINTS, 0.0);
        t1.resize(CH * POINTS, 0.0);
        t2.resize(CH * POINTS, 0.0);
        // Stem: conv 5x5 + ReLU.
        conv(input, IN_PLANES, CH, 5, 2, &net.stem_w, &net.stem_b, t0);
        relu(t0);
        // Residual blocks: t0 = relu(t0 + conv2(relu(conv1(t0)))).
        for block in &net.blocks {
            conv(t0, CH, CH, 3, 1, &block.c1_w, &block.c1_b, t1);
            relu(t1);
            conv(t1, CH, CH, 3, 1, &block.c2_w, &block.c2_b, t2);
            for i in 0..CH * POINTS {
                t0[i] = (t0[i] + t2[i]).max(0.0);
            }
        }
        // Value head: 1x1 conv, global average pool, linear, tanh.
        conv(t0, CH, VCH, 1, 0, &net.v_w, &net.v_b, &mut t1[..VCH * POINTS]);
        relu(&mut t1[..VCH * POINTS]);
        let mut pooled = [0f32; VCH];
        for c in 0..VCH {
            let row = &t1[c * POINTS..(c + 1) * POINTS];
            pooled[c] = row.iter().sum::<f32>() / POINTS as f32;
        }
        let mut acc = net.v_fc_b[0];
        for c in 0..VCH {
            acc += net.v_fc_w[c] * pooled[c];
        }
        let value = f64::from(acc.tanh());
        // Policy head: 3x3 conv, ReLU, 1x1 conv. Channel c is direction c, so the
        // flattened output is indexed by move ID.
        if policy {
            POLICY.with_borrow_mut(|(p1, p2)| {
                p1.resize(PCH * POINTS, 0.0);
                p2.resize(PCH * POINTS, 0.0);
                conv(t0, CH, PCH, 3, 1, &net.p1_w, &net.p1_b, p1);
                relu(p1);
                conv(p1, PCH, PCH, 1, 0, &net.p2_w, &net.p2_b, p2);
                let mut out = [0f32; meridian_engine::NUM_MOVES];
                out.copy_from_slice(p2);
                logits_out = Some(out);
            });
        }
        value
    });
    (value, logits_out)
}

fn relu(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

fn conv(input: &[f32], cin: usize, cout: usize, k: usize, pad: usize, w: &[f32], b: &[f32], out: &mut [f32]) {
    for o in 0..cout {
        let orow = &mut out[o * POINTS..(o + 1) * POINTS];
        for slot in orow.iter_mut() {
            *slot = b[o];
        }
        for c in 0..cin {
            for kh in 0..k {
                let h_lo = (pad as isize - kh as isize).max(0) as usize;
                let h_hi = (BOARD + pad - kh).min(BOARD);
                for h in h_lo..h_hi {
                    let ih = h + kh - pad;
                    for kw in 0..k {
                        let w_lo = (pad as isize - kw as isize).max(0) as usize;
                        let w_hi = (BOARD + pad - kw).min(BOARD);
                        let ws = w[(o * cin + c) * k * k + kh * k + kw];
                        if ws == 0.0 {
                            continue;
                        }
                        let out_base = h * BOARD + w_lo;
                        let in_base = c * POINTS + ih * BOARD + w_lo + kw - pad;
                        let window = &mut orow[out_base..out_base + (w_hi - w_lo)];
                        let source = &input[in_base..in_base + (w_hi - w_lo)];
                        for (dst, src) in window.iter_mut().zip(source) {
                            *dst += ws * src;
                        }
                    }
                }
            }
        }
    }
}
