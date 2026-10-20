//! Benchmark Omni's search on real positions, sampled from the site's games.jsonl.
//!
//! Usage: bench [--games PATH] [--positions N] [--budget B] [--micro]
//!   Samples one position per game at a spread of plies, runs the full search
//!   on each, and reports the time distribution (p50/p90/p99/max), plus the
//!   number of legal moves and a movegen/apply/eval micro-benchmark.
//!
//! Native timing is a lower bound for wasm: wasm is typically 1.5-3x slower.
//! The 5 s move limit is enforced, so the shipped search must stay far below.

use std::time::{Duration, Instant};

use meridian_engine::{Game, Move, notation};
use omni::search::{self, Limits, Params};

/// Reads sampled positions: (game id, ply, position) from real games.
fn sample_positions(path: &str, want: usize, plies: &[usize]) -> Vec<(usize, Game)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    let step = (lines.len() / want.max(1)).max(1);
    for (game_number, line) in lines.iter().enumerate().step_by(step) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if value["termination"].is_string() && value["termination"] != serde_json::Value::Null {
            continue; // Resigned/forfeited games can end anywhere; keep it simple.
        }
        let Some(start) = value["start"].as_str() else { continue };
        let Ok(start_position) = notation::parse_setup(start) else { continue };
        let mut game = Game::from_position(start_position);
        let moves = value["moves"].as_array().cloned().unwrap_or_default();
        let ids: Vec<usize> = moves.iter().filter_map(|id| id.as_u64().map(|v| v as usize)).collect();
        // Replay up to the largest requested ply that this game reaches.
        let &deepest = plies.last().unwrap();
        for &ply in plies {
            if ply >= ids.len() {
                continue;
            }
            while game.moves().len() < ply {
                let Some(mv) = Move::from_index(ids[game.moves().len()]) else { break };
                if game.play(mv).is_err() {
                    break;
                }
            }
            if game.moves().len() == ply && !game.is_over() {
                out.push((game_number, game.clone()));
            }
            if game.is_over() {
                break;
            }
        }
        if out.len() >= want {
            break;
        }
        let _ = deepest;
    }
    out
}

fn percentiles(mut times: Vec<Duration>) -> (Duration, Duration, Duration, Duration, Duration) {
    times.sort();
    if times.is_empty() {
        return (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    }
    let pick = |fraction: f64| times[((times.len() - 1) as f64 * fraction) as usize];
    (times[0], pick(0.5), pick(0.9), pick(0.99), times[times.len() - 1])
}

fn main() {
    let mut games_path = "/tmp/opencode/meridian-games/games.jsonl".to_string();
    let mut positions = 200usize;
    let mut budget = search::MOVE_BUDGET;
    let mut micro = false;
    let mut deadline_ms = 0u64; // 0 = node-budget mode; >0 = wall-clock mode
    let mut stress = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--games" => games_path = args.next().unwrap_or(games_path),
            "--positions" => positions = args.next().and_then(|v| v.parse().ok()).unwrap_or(positions),
            "--budget" => budget = args.next().and_then(|v| v.parse().ok()).unwrap_or(budget),
            "--micro" => micro = true,
            "--deadline-ms" => deadline_ms = args.next().and_then(|v| v.parse().ok()).unwrap_or(4300),
            "--stress" => stress = true,
            _ => eprintln!("unknown arg {arg}"),
        }
    }

    let plies = [8, 24, 48, 72, 96, 114];
    let sampled = sample_positions(&games_path, positions, &plies);
    println!("bench: {} positions sampled from {games_path}", sampled.len());

    if stress {
        // Adversarial deadline check: even tiny budgets must return a legal
        // move, and the deadline must hold to within the check interval.
        let mut worst_overshoot = Duration::ZERO;
        let mut illegal = 0;
        for (_, game) in sampled.iter().take(40) {
            let position = game.position();
            for ms in [100u64, 50, 20] {
                let limits = Limits {
                    deadline: Some(Instant::now() + Duration::from_millis(ms)),
                    node_budget: usize::MAX,
                };
                let start = Instant::now();
                let best = search::best_move_limited(position, &Params::default(), &limits);
                let elapsed = start.elapsed();
                match best {
                    Some(mv) if position.legal_moves().contains(mv) => {}
                    Some(_) => illegal += 1,
                    None if position.legal_moves().is_empty() => {}
                    None => illegal += 1,
                }
                worst_overshoot = worst_overshoot.max(elapsed.saturating_sub(Duration::from_millis(ms)));
            }
        }
        println!("stress: illegal replies: {illegal}, worst deadline overshoot: {worst_overshoot:?}");
        return;
    }

    if deadline_ms > 0 {
        // Realistic move-request timing: a wall-clock deadline, unlimited nodes.
        let mut times = Vec::new();
        let mut depths: Vec<usize> = Vec::new();
        let mut nodes: Vec<usize> = Vec::new();
        for (_, game) in sampled.iter().take(positions.min(40)) {
            let position = game.position();
            let limits = Limits {
                deadline: Some(Instant::now() + Duration::from_millis(deadline_ms)),
                node_budget: usize::MAX,
            };
            let start = Instant::now();
            let analysis = search::analyze_limited(position, &Params::default(), &limits);
            times.push(start.elapsed());
            depths.push(analysis.depth);
            nodes.push(analysis.nodes);
        }
        let (min, p50, p90, p99, max) = percentiles(times.clone());
        let mean = times.iter().sum::<Duration>() / times.len().max(1) as u32;
        depths.sort_unstable();
        nodes.sort_unstable();
        println!(
            "deep deadline={deadline_ms}ms: min={min:?} p50={p50:?} p90={p90:?} p99={p99:?} max={max:?} mean={mean:?}"
        );
        println!(
            "achieved depth: min={} p50={} max={}, nodes: p50={} max={}",
            depths.first().copied().unwrap_or(0),
            depths.get(depths.len() / 2).copied().unwrap_or(0),
            depths.last().copied().unwrap_or(0),
            nodes.get(nodes.len() / 2).copied().unwrap_or(0),
            nodes.last().copied().unwrap_or(0),
        );
        return;
    }

    // Node-budget mode: legacy 2-ply against the deep search, same budget.
    let params = Params { budget, ..Params::default() };
    let mut legacy_times = Vec::new();
    let mut deep_times = Vec::new();
    let mut legal_counts = Vec::new();
    let mut deep_depths: Vec<usize> = Vec::new();
    for (game_number, game) in &sampled {
        let position = game.position();
        let legal = position.legal_moves();
        legal_counts.push(legal.len());

        let start = Instant::now();
        search::analyze_2ply(position, &params);
        legacy_times.push(start.elapsed());

        let start = Instant::now();
        let deep = search::analyze_with(position, &params);
        deep_times.push(start.elapsed());
        deep_depths.push(deep.depth);
        let _ = game_number;
    }
    let (lmin, lp50, lp90, lp99, lmax) = percentiles(legacy_times.clone());
    let (dmin, dp50, dp90, dp99, dmax) = percentiles(deep_times.clone());
    println!("legacy budget={budget}: min={lmin:?} p50={lp50:?} p90={lp90:?} p99={lp99:?} max={lmax:?}");
    println!("deep   budget={budget}: min={dmin:?} p50={dp50:?} p90={dp90:?} p99={dp99:?} max={dmax:?}");
    legal_counts.sort_unstable();
    println!(
        "legal moves: min={} p50={} p90={} max={}",
        legal_counts.first().copied().unwrap_or(0),
        legal_counts.get(legal_counts.len() / 2).copied().unwrap_or(0),
        legal_counts.get(legal_counts.len() * 9 / 10).copied().unwrap_or(0),
        legal_counts.last().copied().unwrap_or(0),
    );
    deep_depths.sort_unstable();
    println!(
        "deep achieved depth: min={} p50={} p90={} max={}",
        deep_depths.first().copied().unwrap_or(0),
        deep_depths.get(deep_depths.len() / 2).copied().unwrap_or(0),
        deep_depths.get(deep_depths.len() * 9 / 10).copied().unwrap_or(0),
        deep_depths.last().copied().unwrap_or(0),
    );

    // Micro-benchmarks on the busiest sampled position (most legal moves) and a late one.
    if micro {
        let busiest = sampled.iter().max_by_key(|g| g.1.position().legal_moves().len());
        let latest = sampled
            .iter()
            .filter(|g| g.1.position().actions_played() >= 90)
            .max_by_key(|g| g.1.position().actions_played());
        for (label, game) in [("most-legal", busiest), ("late-game", latest)] {
            let Some((_, game)) = game else { continue };
            let position = game.position();
            let moves: Vec<Move> = position.legal_moves().iter().collect();
            let reps = 200usize;

            let start = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(position.legal_moves());
            }
            let movegen = start.elapsed() / reps as u32;

            let start = Instant::now();
            for i in 0..reps {
                let mut child = std::hint::black_box(position).clone();
                child.apply_unchecked(moves[i % moves.len()]);
            }
            let apply = start.elapsed() / reps as u32;

            let start = Instant::now();
            for i in 0..reps {
                let mut child = std::hint::black_box(position).clone();
                child.apply_unchecked(moves[i % moves.len()]);
                std::hint::black_box(search::evaluate_with(&child, &params));
            }
            let apply_eval = start.elapsed() / reps as u32;

            let start = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(search::evaluate_with(position, &params));
            }
            let eval = start.elapsed() / reps as u32;

            println!(
                "micro {label} (ply {}, {} legal): movegen={movegen:?} clone+apply={apply:?} apply+eval={apply_eval:?} eval={eval:?}",
                position.actions_played(),
                moves.len(),
            );
        }
    }
}
