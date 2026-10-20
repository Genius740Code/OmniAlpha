//! Omni, a WebAssembly bot for Meridian.
//!
//! The site sends a request as JSON and reads a JSON reply. `"type": "move"` asks for a move and
//! `"type": "analysis"` for the best moves with their values. Both give the position as `start`
//! and the `moves` played since. This file reads requests and writes replies, and ends with the
//! three functions the site calls. Omni's strategy is in `search.rs`: Scout's 2-ply search,
//! bounded by its position budget, so a reply can never overrun its time limit; the deeper
//! deadline-guarded search there lost the A/B and is kept for tuning only.
//! https://constellation.blueshrimp.uk/bot-api.md has the details.

pub mod search;

use std::cell::RefCell;

use meridian_engine::{Game, Move, notation};
use serde_json::{Value, json};

use search::Analysis;

/// How many moves an analysis lists.
const LISTED_CANDIDATES: usize = 5;
const DEFAULT_VISITS: u64 = 240;
/// The smallest `moveTimeMs` the full 4096-position budget is served at: the
/// full search measures at most ~125 ms in wasm on the heaviest late-game
/// positions, so any limit from 250 ms up is met with a wide margin. Smaller
/// limits get a budget scaled to 4 positions per millisecond — best effort
/// only, because a position's cost varies ~10x with game stage; the site's
/// documented limit is 5000 ms, where the full budget is served.
const FULL_BUDGET_FROM_MS: u64 = 250;
const SCALED_POSITIONS_PER_MS: usize = 4;

/// The reply to one request: a move, an analysis, or an error.
pub fn answer(request: &Value) -> Value {
    reply(request).unwrap_or_else(|error| json!({"error": error}))
}

fn reply(request: &Value) -> Result<Value, String> {
    let game = replay(request)?;
    match request["type"].as_str() {
        Some("move") => {
            // Scout's exact 2-ply shape, bounded by its position budget instead
            // of a clock, so a reply takes at most ~125 ms in wasm. The deep
            // search lost the A/B against this at every setting, node and
            // wall-clock alike (docs/meridian/OMNI_SEARCH_CHANGES.md), so it is
            // not used for replies. `moveTimeMs` only scales the budget down
            // for limits below 250 ms; at the site's 5000 ms this is Scout's
            // full 4096 positions, bit for bit.
            let move_time_ms = request["limits"]["moveTimeMs"].as_u64().unwrap_or(5000);
            let budget = if move_time_ms >= FULL_BUDGET_FROM_MS {
                search::MOVE_BUDGET
            } else {
                (move_time_ms as usize).saturating_mul(SCALED_POSITIONS_PER_MS).clamp(16, search::MOVE_BUDGET)
            };
            let params = search::Params { budget, ..search::Params::default() };
            let best = search::analyze_2ply(game.position(), &params)
                .candidates
                .first()
                .map(|candidate| candidate.mv)
                .ok_or("The game is over.")?;
            Ok(json!({"move": best.index()}))
        }
        Some("analysis") => {
            // Scout's exact shape at the requested `visits`, as the site's own
            // bot served reviews: the deep search cannot finish an iteration at
            // review-sized budgets, and lost the A/B at large ones too.
            let visits = request["limits"]["visits"].as_u64().unwrap_or(DEFAULT_VISITS).min(search::MOVE_BUDGET as u64);
            let analysis = search::analyze_2ply(
                game.position(),
                &search::Params { budget: visits as usize, ..search::Params::default() },
            );
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
