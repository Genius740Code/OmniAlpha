//! Benchmark and verification harness for the Meridian value/policy net.
//!
//! Modes (native; the default mode also works under `wasm32-wasip1`, where it
//! has no files and uses only the embedded bench games):
//!
//!   (default)        Time the forward pass on real mid-game positions from
//!                     the embedded bench games (close, recent site games).
//!   --check          End-to-end verification: replay validation games, rebuild
//!                     the input planes with the engine (must byte-match the
//!                     dumper's stored inputs) and compare the Rust value head
//!                     with torch's reference outputs on the same positions.
//!   --val PATH [N]   Metrics over the validation games: the net against the
//!                     real final margins and against two heuristic
//!                     baselines (current score, Scout's evaluation), plus
//!                     policy quality. PATH is val_games.json; N is a stride
//!                     (default 10) to keep the NN pass affordable.

use std::time::Instant;

use meridian_engine::{Game, Move, Player, Position};
use serde_json::Value;

use nn::{MARGIN_SCALE, eval_value, input_planes, top_moves};

const BENCH_GAMES: &str = include_str!("../../fixtures/bench_games.json");
const DEFAULT_VAL_GAMES: &str = "/tmp/opencode/nn-data/out3/val_games.json";
const DEFAULT_DENSE: &str = "/tmp/opencode/nn-data/out/val_dense_head.bin";
const DEFAULT_EXPECTED: &str = "/tmp/opencode/nn-data/out/expected_dense.bin";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dense_path = DEFAULT_DENSE.to_string();
    let mut expected_path = DEFAULT_EXPECTED.to_string();
    let mut val_path = DEFAULT_VAL_GAMES.to_string();
    let mut stride = 10usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--check" => return check(&dense_path, &expected_path),
            "--val" => {
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    val_path = args[i + 1].clone();
                    i += 1;
                }
                if i + 1 < args.len() && args[i + 1].parse::<usize>().is_ok() {
                    stride = args[i + 1].parse().unwrap();
                }
                return val_metrics(&val_path, stride);
            }
            "--dense" => {
                if i + 1 < args.len() {
                    dense_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--expected" => {
                if i + 1 < args.len() {
                    expected_path = args[i + 1].clone();
                    i += 1;
                }
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    timing();
}

/// One game from the datasets: `moves` from the standard start, plus metadata.
struct BenchGame {
    moves: Vec<Move>,
    opening_len: usize,
    margin: f64,
}

fn parse_games(json: &str) -> Vec<BenchGame> {
    let values: Vec<Value> = serde_json::from_str(json).expect("game list json");
    values
        .into_iter()
        .filter_map(|g| {
            let moves = g["moves"]
                .as_array()?
                .iter()
                .map(|id| id.as_u64().and_then(|i| usize::try_from(i).ok()).and_then(Move::from_index))
                .collect::<Option<Vec<_>>>()?;
            Some(BenchGame {
                moves,
                opening_len: g["opening_len"].as_u64().unwrap_or(0) as usize,
                margin: g["margin"].as_f64().unwrap_or(0.0),
            })
        })
        .collect()
}

/// Replays one game, calling `at` for every position a real player moved from.
fn replay(game: &BenchGame, mut at: impl FnMut(&Position, usize)) {
    let mut g = Game::from_position(Position::new());
    for (i, &mv) in game.moves.iter().enumerate() {
        if g.is_over() {
            return;
        }
        let position = g.position();
        if i >= game.opening_len && !position.is_finished() {
            at(position, i);
        }
        if g.play(mv).is_err() {
            return;
        }
    }
}

/// Positions used for timing: every 5th decision position of the bench games.
fn bench_positions() -> Vec<Position> {
    let mut positions = Vec::new();
    for game in parse_games(BENCH_GAMES) {
        replay(&game, |position, i| {
            if i % 5 == 0 {
                positions.push(position.clone());
            }
        });
    }
    positions
}

fn timing() {
    let positions = bench_positions();
    println!("nnbench: {} real mid-game positions (bench games)", positions.len());
    let mut input = [0f32; nn::IN_PLANES * nn::POINTS];

    // Warmup (also triggers the one-time weight decode).
    for position in positions.iter().take(3) {
        std::hint::black_box(eval_value(position));
    }

    let mut encode_us = Vec::new();
    let mut value_us = Vec::new();
    let mut full_us = Vec::new();
    for position in &positions {
        let t = Instant::now();
        input_planes(position, &mut input);
        std::hint::black_box(input);
        encode_us.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        std::hint::black_box(eval_value(position));
        value_us.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        std::hint::black_box(nn::eval(position));
        full_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    for (name, v) in [("encode", &mut encode_us), ("value-only", &mut value_us), ("full", &mut full_us)] {
        v.sort_by(|a, b| a.total_cmp(b));
        let at = |p: f64| v[((v.len() as f64 - 1.0) * p) as usize];
        let (median, p90, max) = (at(0.5), at(0.9), at(1.0));
        println!("{name:>10}: median {median:7.0} us   p90 {p90:7.0} us   max {max:7.0} us");
    }
    println!(
        "single-threaded. At omni's 4096-node budget a value-only leaf eval would cost \
         {:.2} s/move; 100 root/candidate evals cost {:.1} ms/move.",
        value_us[value_us.len() / 2] * 4096.0 / 1e6,
        value_us[value_us.len() / 2] * 100.0 / 1e3
    );
}

/// End-to-end verification against the training pipeline's own outputs:
/// 1. the Rust input builder must byte-match the dumper's stored inputs, and
/// 2. the Rust value head must match torch's outputs on the same positions.
fn check(dense_path: &str, expected_path: &str) {
    let dense: Vec<f32> = read_f32(dense_path);
    let expected: Vec<f32> = read_f32(expected_path);
    let per = nn::IN_PLANES * nn::POINTS + 1;
    assert_eq!(dense.len() % per, 0, "dense file is whole samples");
    let n = dense.len() / per;
    assert_eq!(n, expected.len(), "dense samples and torch values must correspond");
    println!("check: {n} samples from {dense_path} against torch outputs in {expected_path}");

    // The same first `n` validation decision positions the dumper wrote out.
    let games_json = std::fs::read_to_string(DEFAULT_VAL_GAMES).expect("val_games.json (native only)");
    let games = parse_games(&games_json);
    let mut positions: Vec<Position> = Vec::with_capacity(n);
    'outer: for game in &games {
        replay(game, |position, _| {
            if positions.len() < n {
                positions.push(position.clone());
            }
        });
        if positions.len() >= n {
            break 'outer;
        }
    }
    assert_eq!(positions.len(), n, "not enough validation games to rebuild {n} positions");

    let mut input = [0f32; nn::IN_PLANES * nn::POINTS];
    let (mut input_bad, mut max_value_err, mut sign_ok) = (0usize, 0f32, 0usize);
    for (i, position) in positions.iter().enumerate() {
        input_planes(position, &mut input);
        let stored = &dense[i * per..i * per + nn::IN_PLANES * nn::POINTS];
        if stored.iter().zip(input.iter()).any(|(a, b)| a != b) {
            input_bad += 1;
            if input_bad == 1 {
                let first = stored.iter().zip(input.iter()).position(|(a, b)| a != b).expect("a diff");
                eprintln!("  first input diff at sample {i}, element {first}: {} vs {}", stored[first], input[first]);
            }
        }
        let value = eval_value(position);
        max_value_err = max_value_err.max((value as f32 - expected[i]).abs());
        if (value >= 0.0) == (expected[i] >= 0.0) {
            sign_ok += 1;
        }
    }
    println!("  input planes: {}/{} byte-exact", n - input_bad, n);
    println!("  value head: max |rust - torch| = {max_value_err:.6}, sign agreement {sign_ok}/{n}");
    if input_bad == 0 && max_value_err < 5e-3 {
        println!("PASS");
    } else {
        println!("FAIL (inputs exact, value tolerance 5e-3)");
        std::process::exit(1);
    }
}

/// The value Scout's evaluation would give, in points (a port of omni's
/// `evaluate_with` with Scout's default parameters: horizon 12, room 0.4).
fn scout_eval(position: &Position) -> f64 {
    let events_left = f64::from(position.scoring_events_left());
    let worth = |player: Player| {
        position.score(player).to_f64()
            + position.area(player).to_f64() * events_left.min(12.0)
            + room(position, player) * 0.4 * events_left.min(1.0)
    };
    worth(Player::Blue) - worth(Player::Red)
}

/// The area of the convex hull of a player's nodes (omni's `room`).
fn room(position: &Position, player: Player) -> f64 {
    let mut points: Vec<meridian_engine::Point> = position.nodes(player).iter().collect();
    points.sort_by_key(|p| (p.x(), p.y()));
    if points.len() < 3 {
        return 0.0;
    }
    let mut hull = half_hull(points.iter().copied());
    hull.extend(half_hull(points.iter().rev().copied()));
    let first = hull[0];
    hull.windows(2).map(|pair| cross(first, pair[0], pair[1]) as f64).sum::<f64>().abs() * 0.5
}

fn half_hull(points: impl Iterator<Item = meridian_engine::Point>) -> Vec<meridian_engine::Point> {
    let mut chain: Vec<meridian_engine::Point> = Vec::new();
    for point in points {
        while chain.len() >= 2 && cross(chain[chain.len() - 2], chain[chain.len() - 1], point) <= 0 {
            chain.pop();
        }
        chain.push(point);
    }
    chain.pop();
    chain
}

fn cross(
    a: meridian_engine::Point,
    b: meridian_engine::Point,
    c: meridian_engine::Point,
) -> i32 {
    i32::from(b.x() - a.x()) * i32::from(c.y() - a.y()) - i32::from(b.y() - a.y()) * i32::from(c.x() - a.x())
}

#[derive(Default)]
struct Stats {
    n: usize,
    sq_err: f64,
    sign_ok: usize,
    sx: f64,
    sy: f64,
    sxy: f64,
    sxx: f64,
    syy: f64,
}

impl Stats {
    fn add(&mut self, pred: f64, target: f64) {
        self.n += 1;
        self.sq_err += (pred - target) * (pred - target);
        if (pred >= 0.0) == (target >= 0.0) {
            self.sign_ok += 1;
        }
        self.sx += pred;
        self.sy += target;
        self.sxy += pred * target;
        self.sxx += pred * pred;
        self.syy += target * target;
    }
    fn mse(&self) -> f64 {
        self.sq_err / self.n as f64
    }
    fn sign(&self) -> f64 {
        self.sign_ok as f64 / self.n as f64
    }
    fn corr(&self) -> f64 {
        let n = self.n as f64;
        let num = n * self.sxy - self.sx * self.sy;
        let den = ((n * self.sxx - self.sx * self.sx) * (n * self.syy - self.sy * self.sy)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    fn report(&self, name: &str) {
        println!(
            "  {name:<22} mse {:.4}   sign-acc {:.4}   corr {:.4}   (n={})",
            self.mse(), self.sign(), self.corr(), self.n
        );
    }
}

/// Metrics over the validation games: the net vs the real final margins and vs
/// heuristic baselines, plus policy quality. All predictors are compared on
/// the same positions, in tanh(margin/1500) space.
fn val_metrics(path: &str, stride: usize) {
    let games_json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e} (native only)"));
    let games = parse_games(&games_json);
    println!("val: {} games (stride {stride} for the net)", games.len());

    let (mut nn, mut score, mut scout) = (Stats::default(), Stats::default(), Stats::default());
    let mut scout_all = Stats::default();
    let (mut n_seen, mut n_scored) = (0usize, 0usize);
    let (mut argmax_legal, mut top1_match, mut top8_hit, mut n_policy, mut sum_legal) =
        (0usize, 0usize, 0usize, 0usize, 0usize);

    for game in &games {
        let target = (game.margin / MARGIN_SCALE).tanh();
        replay(game, |position, i| {
            let score_pred = ((position.score(Player::Blue).to_f64()
                - position.score(Player::Red).to_f64())
                / MARGIN_SCALE)
                .tanh();
            let scout_pred = (scout_eval(position) / MARGIN_SCALE).tanh();
            scout_all.add(scout_pred, target);
            if n_seen % stride == 0 {
                let (value, logits) = nn::eval_logits(position);
                let logits = logits.expect("policy head requested");
                nn.add(value, target);
                score.add(score_pred, target);
                scout.add(scout_pred, target);
                n_scored += 1;

                // Policy quality on the same positions.
                let played = game.moves[i];
                sum_legal += position.legal_moves().len();
                n_policy += 1;
                let mut best = 0usize;
                for (id, &l) in logits.iter().enumerate() {
                    if logits[best] < l {
                        best = id;
                    }
                }
                let argmax = Move::from_index(best).expect("id in range");
                if position.legal_moves().contains(argmax) {
                    argmax_legal += 1;
                }
                let ranked = top_moves(position, &logits, 8);
                if ranked.first().is_some_and(|&(mv, _)| mv == played) {
                    top1_match += 1;
                }
                if ranked.iter().any(|&(mv, _)| mv == played) {
                    top8_hit += 1;
                }
            }
            n_seen += 1;
        });
    }
    println!("value predictors vs tanh(final margin / 1500):");
    nn.report("nn value");
    score.report("current score");
    scout.report("scout eval (stride)");
    println!(
        "  {:<22} mse {:.4}   sign-acc {:.4}   corr {:.4}   (n={})",
        "scout eval (all)",
        scout_all.mse(),
        scout_all.sign(),
        scout_all.corr(),
        scout_all.n
    );
    println!("policy (same positions, legality from the engine):");
    println!(
        "  argmax move legal: {}/{} ({:.1}%),  legal top-1 == played: {}/{} ({:.1}%),  played in top-8: {}/{} ({:.1}%)",
        argmax_legal, n_policy, 100.0 * argmax_legal as f64 / n_policy as f64,
        top1_match, n_policy, 100.0 * top1_match as f64 / n_policy as f64,
        top8_hit, n_policy, 100.0 * top8_hit as f64 / n_policy as f64,
    );
    println!("  mean legal moves per position: {:.0}", sum_legal as f64 / n_policy as f64);
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(bytes.len() % 4 == 0);
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
