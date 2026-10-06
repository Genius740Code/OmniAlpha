//! Tune Omni's search parameters by playing engine-vs-engine matches.
//!
//! Usage: tune [--budget N] [--games N] [--openings K] [--seed S] [--only NAME]
//!   Plays baseline (Scout defaults) vs each candidate, alternating colors,
//!   and prints W/L/D from the candidate's point of view.
//!   Each game starts from K random opening moves (seeded RNG) so that
//!   large --games values give independent games instead of replays.
//!   --only restricts to candidates whose name contains the filter.
//!
//! This is the "train" step: parameter search on the real rules engine.
//! Keep it small: full-budget games are slow; tune at reduced budget, then
//! verify the winner at full budget before shipping the .wasm.

use std::time::Instant;

use meridian_engine::{Game, Move, Outcome, Player};
use omni::{search, search::Params};

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

fn play_one(blue: &Params, red: &Params, openings: usize, seed: u64) -> Outcome {
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
        let params = if game.position().to_move() == Player::Blue { blue } else { red };
        match search::best_move_with(game.position(), params) {
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
    let mut only = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--budget" => budget = args.next().and_then(|v| v.parse().ok()).unwrap_or(budget),
            "--games" => games = args.next().and_then(|v| v.parse().ok()).unwrap_or(games),
            "--openings" => openings = args.next().and_then(|v| v.parse().ok()).unwrap_or(openings),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            "--only" => only = args.next().unwrap_or_default(),
            _ => eprintln!("unknown arg {a}"),
        }
    }

    let baseline = Params { budget, ..Params::default() };
    let candidates: Vec<(&str, Params)> = vec![
        ("width12", Params { width: 12, budget, ..Params::default() }),
        ("width6", Params { width: 6, budget, ..Params::default() }),
        ("width4", Params { width: 4, budget, ..Params::default() }),
        ("width16", Params { width: 16, budget, ..Params::default() }),
        ("horizon8", Params { horizon: 8, budget, ..Params::default() }),
        ("horizon10", Params { horizon: 10, budget, ..Params::default() }),
        ("horizon14", Params { horizon: 14, budget, ..Params::default() }),
        ("horizon16", Params { horizon: 16, budget, ..Params::default() }),
        ("room0.0", Params { room_weight: 0.0, budget, ..Params::default() }),
        ("room0.2", Params { room_weight: 0.2, budget, ..Params::default() }),
        ("room0.3", Params { room_weight: 0.3, budget, ..Params::default() }),
        ("room0.5", Params { room_weight: 0.5, budget, ..Params::default() }),
        ("room0.6", Params { room_weight: 0.6, budget, ..Params::default() }),
        ("room0.8", Params { room_weight: 0.8, budget, ..Params::default() }),
    ];

    println!(
        "tune: budget={budget} games-per-pairing={games} openings={openings} seed={seed} (alternating colors)"
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
