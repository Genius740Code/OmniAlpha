//! Omni, a WebAssembly bot for Meridian.
//!
//! The site sends a request as JSON and reads a JSON reply. `"type": "move"` asks for a move and
//! `"type": "analysis"` for the best moves with their values. Both give the position as `start`
//! and the `moves` played since. This file reads requests and writes replies, and ends with the
//! three functions the site calls. Omni's strategy is in `search.rs`: an iterative-deepening
//! alpha-beta search under a wall-clock deadline, so a move can never overrun its time.
//! https://constellation.blueshrimp.uk/bot-api.md has the details.

pub mod search;

use std::cell::RefCell;
use std::time::{Duration, Instant};

use meridian_engine::{Game, Move, notation};
use serde_json::{Value, json};

use search::Analysis;

/// How many moves an analysis lists.
const LISTED_CANDIDATES: usize = 5;
const DEFAULT_VISITS: u64 = 240;
/// The site stops a move that runs past its limit and counts it as a failure
/// (bot-api.md, "Limits"), so Omni aims for 86% of `moveTimeMs`, and never
/// more than 4.3 s: the last 14% covers the request parse, the reply write and
/// any clock drift between us and the site.
const MOVE_TIME_CAP_MS: u64 = 4300;
const MOVE_TIME_SHARE: u64 = 86;
/// Analysis keeps its `visits` budget; the wall clock only stops a runaway.
const ANALYSIS_TIME_CAP: Duration = Duration::from_secs(20);

/// The reply to one request: a move, an analysis, or an error.
pub fn answer(request: &Value) -> Value {
    reply(request).unwrap_or_else(|error| json!({"error": error}))
}

fn reply(request: &Value) -> Result<Value, String> {
    let game = replay(request)?;
    match request["type"].as_str() {
        Some("move") => {
            let move_time_ms = request["limits"]["moveTimeMs"].as_u64().unwrap_or(5000);
            let aim_ms = (move_time_ms * MOVE_TIME_SHARE / 100).clamp(50, MOVE_TIME_CAP_MS);
            let limits = search::Limits {
                deadline: Some(Instant::now() + Duration::from_millis(aim_ms)),
                node_budget: usize::MAX,
            };
            let best = search::best_move_limited(game.position(), &search::Params::default(), &limits)
                .ok_or("The game is over.")?;
            Ok(json!({"move": best.index()}))
        }
        Some("analysis") => {
            let visits = request["limits"]["visits"].as_u64().unwrap_or(DEFAULT_VISITS).min(search::MOVE_BUDGET as u64);
            // Game reviews ask for `visits` 32: at that size Scout's 2-ply shape
            // gives better candidates than a deep search that never finishes an
            // iteration, so use it; the deep search needs a real node budget.
            let analysis = if visits < 256 {
                search::analyze_2ply(
                    game.position(),
                    &search::Params { budget: visits as usize, ..search::Params::default() },
                )
            } else {
                let limits = search::Limits {
                    deadline: Some(Instant::now() + ANALYSIS_TIME_CAP),
                    node_budget: visits as usize,
                };
                search::analyze_limited(game.position(), &search::Params::default(), &limits)
            };
            Ok(json!({"analysis": analysis_json(&analysis)}))
        }
        _ => Err("type must be move or analysis".into()),
    }
}

/// The request's position: `start`, then the move IDs in `moves`.
fn replay(request: &Value) -> Result<Game, String> {
    let start = request["start"].as_str().ok_or("start is required: a position such as [B:A10-D10 R:P10-S10]")?;
    let mut game = Game::from_position(notation::parse_setup(start).map_err(|error| error.to_string())?);
    let moves = request["moves"].as_array().ok_or("moves is required: an array of move IDs")?;
    for id in moves {
        let mv = id.as_u64().and_then(|id| usize::try_from(id).ok()).and_then(Move::from_index);
        let mv = mv.ok_or(format!("{id} is not a move ID"))?;
        game.play(mv).map_err(|error| error.to_string())?;
    }
    Ok(game)
}

fn analysis_json(analysis: &Analysis) -> Value {
    let candidates: Vec<Value> = analysis
        .candidates
        .iter()
        .take(LISTED_CANDIDATES)
        .map(|candidate| {
            json!({
                "move": candidate.mv.index(),
                "evaluation": candidate.evaluation,
                "pv": candidate.pv.iter().map(|mv| mv.index()).collect::<Vec<_>>(),
                "visits": candidate.visits,
            })
        })
        .collect();
    json!({
        "evaluation": analysis.evaluation,
        "unit": "points",
        "depth": analysis.depth,
        "nodes": analysis.nodes,
        "candidates": candidates,
    })
}

// How the site calls the module: version 1 of the bot interface.

thread_local! {
    /// The latest reply: its length as a little-endian u32, then the JSON. It stays until the next
    /// request.
    static REPLY: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn meridian_abi() -> u32 {
    1
}

/// Room for a request of `len` bytes, which the site then writes there.
#[unsafe(no_mangle)]
pub extern "C" fn meridian_alloc(len: usize) -> *mut u8 {
    std::mem::ManuallyDrop::new(Vec::<u8>::with_capacity(len)).as_mut_ptr()
}

/// Answers the request the site wrote, and returns where the reply is.
///
/// # Safety
/// `ptr` and `len` must be a buffer from [`meridian_alloc`], holding the request.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn meridian_run(ptr: *mut u8, len: usize) -> *const u8 {
    let request = unsafe { Vec::from_raw_parts(ptr, len, len) };
    let reply = match serde_json::from_slice(&request) {
        Ok(request) => answer(&request),
        Err(error) => json!({"error": format!("Invalid JSON: {error}")}),
    };
    let body = reply.to_string().into_bytes();
    REPLY.with_borrow_mut(|out| {
        out.clear();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out.as_ptr()
    })
}
