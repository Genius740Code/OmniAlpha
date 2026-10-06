//! Tune Omni's search parameters by playing engine-vs-engine matches.
//!
//! Usage: tune [--budget N] [--games N]
//!   Plays baseline (Scout defaults) vs each candidate, alternating colors,
//!   and prints W/L/D from the candidate's point of view.
//!
//! This is the "train" step: parameter search on the real rules engine.
//! Keep it small: full-budget games are slow; tune at reduced budget, then
//! verify the winner at full budget before shipping the .wasm.

use std::time::Instant;

use meridian_engine::{Game, Outcome, Player};
use omni::{search, search::Params};

fn play_one(blue: &Params, red: &Params) -> Outcome {
    let mut game = Game::new();
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
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--budget" => budget = args.next().and_then(|v| v.parse().ok()).unwrap_or(budget),
            "--games" => games = args.next().and_then(|v| v.parse().ok()).unwrap_or(games),
            _ => eprintln!("unknown arg {a}"),
        }
    }

    let baseline = Params { budget, ..Params::default() };
    let candidates: Vec<(&str, Params)> = vec![
        ("scout-default", Params { budget, ..Params::default() }),
        ("width12", Params { width: 12, budget, ..Params::default() }),
        ("width6", Params { width: 6, budget, ..Params::default() }),
        ("horizon8", Params { horizon: 8, budget, ..Params::default() }),
        ("horizon16", Params { horizon: 16, budget, ..Params::default() }),
        ("room0.2", Params { room_weight: 0.2, budget, ..Params::default() }),
        ("room0.6", Params { room_weight: 0.6, budget, ..Params::default() }),
    ];

    println!("tune: budget={budget} games-per-pairing={games} (alternating colors)");
    for (name, cand) in &candidates {
        if name == &"scout-default" {
            continue;
        }
        let t0 = Instant::now();
        let (mut w, mut l, mut d) = (0, 0, 0);
        for g in 0..games {
            // Alternate: even games candidate is Blue, odd games candidate is Red.
            let (blue, red) = if g % 2 == 0 { (cand, &baseline) } else { (&baseline, cand) };
            let outcome = play_one(blue, red);
            let perspective = if g % 2 == 0 { Player::Blue } else { Player::Red };
            let (dw, dl, dd) = score(outcome, perspective);
            w += dw;
            l += dl;
            d += dd;
        }
        println!("{name:>14}: {w}-{l}-{d} (candidate view) in {:.1}s", t0.elapsed().as_secs_f64());
    }
}
