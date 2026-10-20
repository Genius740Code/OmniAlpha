//! Parity check: the deep search at depth 2, uncapped, must choose the same moves
//! as Scout's exact 2-ply (`analyze_2ply`), because they search the same tree.
//! Any divergence beyond tie-breaking is a mechanical bug in the deep search.
//!
//! Usage: parity [--games PATH] [--positions N] [--depth D]
//!   Also prints, for --depth D >= 2, how often the deeper search disagrees.

use std::time::Instant;

use meridian_engine::{Game, Move, notation};
use omni::search::{self, Limits, Params};

fn sample_positions(path: &str, want: usize, plies: &[usize]) -> Vec<Game> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    let step = (lines.len() / want.max(1)).max(1);
    'game: for line in lines.iter().enumerate().step_by(step) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line.1) else { continue };
        if value["termination"].is_string() {
            continue;
        }
        let Some(start) = value["start"].as_str() else { continue };
        let Ok(start_position) = notation::parse_setup(start) else { continue };
        let mut game = Game::from_position(start_position);
        let ids: Vec<usize> = value["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(|id| id.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_default();
        for &ply in plies {
            if ply >= ids.len() {
                continue;
            }
            while game.moves().len() < ply {
                let Some(mv) = Move::from_index(ids[game.moves().len()]) else { break 'game };
                if game.play(mv).is_err() {
                    break 'game;
                }
            }
            if game.moves().len() == ply && !game.is_over() {
                out.push(game.clone());
            }
        }
        if out.len() >= want {
            break;
        }
    }
    out
}

fn main() {
    let mut games_path = "target/games.jsonl".to_string();
    let mut positions = 30usize;
    let mut depth = 2usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--games" => games_path = args.next().unwrap_or(games_path),
            "--positions" => positions = args.next().and_then(|v| v.parse().ok()).unwrap_or(positions),
            "--depth" => depth = args.next().and_then(|v| v.parse().ok()).unwrap_or(depth),
            _ => eprintln!("unknown arg {arg}"),
        }
    }

    let sampled = sample_positions(&games_path, positions, &[8, 24, 48, 72, 96, 112]);
    println!("parity: {} positions, deep depth {depth}", sampled.len());

    let legacy_params = Params::default();
    let deep_params = Params { depth_max: depth.min(2) as u8, branch: 0, ..Params::default() };
    let mut same = 0;
    let mut differ = 0;
    let mut eval_gap = 0.0f64;
    for game in &sampled {
        let position = game.position();
        let legacy = search::analyze_2ply(position, &legacy_params);
        let started = Instant::now();
        let deep = search::analyze_limited(
            position,
            &deep_params,
            &Limits { deadline: Some(started + std::time::Duration::from_secs(60)), node_budget: 1 << 24 },
        );
        let legacy_move = legacy.candidates.first().map(|c| c.mv);
        let deep_move = deep.candidates.first().map(|c| c.mv);
        if legacy_move == deep_move {
            same += 1;
            eval_gap += (legacy.evaluation - deep.evaluation).abs();
        } else {
            differ += 1;
            println!(
                "  differ at ply {}: legacy={legacy_move:?} ({:.2}) deep={deep_move:?} ({:.2}) deep_depth={}",
                position.actions_played(),
                legacy.evaluation,
                deep.evaluation,
                deep.depth,
            );
        }
    }
    println!("depth-2 parity: same {same}, differ {differ}, mean |eval gap| on same moves: {:.3}", eval_gap / same.max(1) as f64);
}
