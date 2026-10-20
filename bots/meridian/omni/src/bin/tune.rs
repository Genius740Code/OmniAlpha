//! Tune Omni's search parameters by playing engine-vs-engine matches.
//!
//! Usage: tune [--budget N] [--games N] [--openings K] [--seed S] [--only NAME]
//!   Plays the legacy 2-ply engine (Scout's shape, `analyze_2ply`) against each
//!   candidate, alternating colors, and prints W/L/D from the candidate's view.
//!   Each game starts from K random opening moves (seeded RNG) so that large
//!   --games values give independent games instead of replays.
//!   --only restricts to candidates whose name contains the filter.
//!
//! `--budget N` sets both sides' budget: positions for the legacy engine,
//! nodes for the deep search, so both sides spend about the same time.
//!
//! This is the "train" step: parameter search on the real rules engine.
//! Keep it small: full-budget games are slow; tune at reduced budget, then
//! verify the winner at full budget before shipping the .wasm.

use std::time::Instant;

use meridian_engine::{Game, Move, Outcome, Player};
use omni::search::{self, Params};

/// Minimal deterministic RNG (xorshift64*) — no extra crates.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        if x == 0 {
            x = 0x9e3779b97f4a7c15;
        }
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

/// Which engine a side plays.
enum Engine {
    /// Scout's exact 2-ply search over `width` candidates.
    Legacy(Params),
    /// The deep iterative-deepening search under a node budget.
    Deep(Params),
    /// The deep search under a wall-clock deadline, as the site plays it.
    Timed(Params, std::time::Duration),
}

impl Engine {
    fn best(&self, position: &meridian_engine::Position) -> Option<Move> {
        match self {
            Engine::Legacy(params) => search::analyze_2ply(position, params).candidates.first().map(|c| c.mv),
            Engine::Deep(params) => search::best_move_with(position, params),
            Engine::Timed(params, per_move) => {
                let limits = search::Limits {
                    deadline: Some(std::time::Instant::now() + *per_move),
                    node_budget: usize::MAX,
                };
                search::best_move_limited(position, params, &limits)
            }
        }
    }
}

fn play_one(blue: &Engine, red: &Engine, openings: usize, seed: u64) -> Outcome {
    let mut game = Game::new();
    if openings > 0 {
        let mut rng = Rng(seed);
        for _ in 0..openings {
            if game.is_over() {
                break;
            }
            let moves: Vec<Move> = game.legal_moves().iter().collect();
            if moves.is_empty() {
                break;
            }
            let mv = moves[rng.below(moves.len())];
            if game.play(mv).is_err() {
                break;
            }
        }
    }
    while !game.is_over() {
        let engine = if game.position().to_move() == Player::Blue { blue } else { red };
        match engine.best(game.position()) {
            Some(mv) => {
                game.play(mv).expect("search returns legal moves");
            }
            None => break,
        }
    }
    game.outcome().expect("game over implies outcome")
}

fn score(outcome: Outcome, perspective: Player) -> (u32, u32, u32) {
    match outcome {
        Outcome::Win(w) if w == perspective => (1, 0, 0),
        Outcome::Win(_) => (0, 1, 0),
        Outcome::Draw(_) => (0, 0, 1),
    }
}

fn main() {
    let mut budget = 512usize;
    let mut games = 4usize;
    let mut openings = 6usize;
    let mut seed = 1u64;
    let mut deep_ms = 0u64; // 0 = node-budget candidates; >0 = wall-clock candidates at this many ms
    let mut only = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--budget" => budget = args.next().and_then(|v| v.parse().ok()).unwrap_or(budget),
            "--games" => games = args.next().and_then(|v| v.parse().ok()).unwrap_or(games),
            "--openings" => openings = args.next().and_then(|v| v.parse().ok()).unwrap_or(openings),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            "--deep-ms" => deep_ms = args.next().and_then(|v| v.parse().ok()).unwrap_or(deep_ms),
            "--only" => only = args.next().unwrap_or_default(),
            _ => eprintln!("unknown arg {a}"),
        }
    }

    let baseline = Engine::Legacy(Params { budget, ..Params::default() });
    let timed = |width: usize, branch: usize, depth_max: u8| {
        Engine::Timed(
            Params { width, branch, depth_max, budget, ..Params::default() },
            std::time::Duration::from_millis(deep_ms),
        )
    };
    let deep = |width: usize, branch: usize, depth_max: u8| {
        Engine::Deep(Params { width, branch, depth_max, budget, ..Params::default() })
    };
    let candidates: Vec<(&str, Engine)> = if deep_ms > 0 {
        vec![
            ("timed-w8", timed(8, 24, 8)),
            ("timed-w16", timed(16, 24, 8)),
            ("timed-b48", timed(8, 48, 8)),
            ("timed-b12", timed(8, 12, 8)),
        ]
    } else {
        vec![
            ("deep-w8", deep(8, 24, 8)),
            ("deep-w16", deep(16, 24, 8)),
            ("deep-w4", deep(4, 24, 8)),
            ("deep-b48", deep(8, 48, 8)),
            ("deep-b12", deep(8, 12, 8)),
            ("deep-b0", deep(8, 0, 8)),
            ("deep-w16-b48", deep(16, 48, 8)),
            ("deep-d4", deep(8, 24, 4)),
            ("deep-d6", deep(8, 24, 6)),
        ]
    };

    println!(
        "tune: baseline=legacy2ply(w8,budget={budget}) deep_ms={deep_ms} games-per-pairing={games} openings={openings} seed={seed} (alternating colors)"
    );
    for (name, cand) in &candidates {
        if !only.is_empty() && !name.contains(&only as &str) {
            continue;
        }
        let t0 = Instant::now();
        let (mut w, mut l, mut d) = (0, 0, 0);
        for g in 0..games {
            // Alternate: even games candidate is Blue, odd games candidate is Red.
            let (blue, red) = if g % 2 == 0 { (cand, &baseline) } else { (&baseline, cand) };
            let outcome = play_one(blue, red, openings, seed * 1_000_003 + g as u64 * 7919 + 17);
            let perspective = if g % 2 == 0 { Player::Blue } else { Player::Red };
            let (dw, dl, dd) = score(outcome, perspective);
            w += dw;
            l += dl;
            d += dd;
        }
        println!("{name:>14}: {w}-{l}-{d} (candidate view) in {:.1}s", t0.elapsed().as_secs_f64());
    }
}
