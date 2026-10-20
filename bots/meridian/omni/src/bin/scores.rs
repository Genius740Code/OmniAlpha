//! Replay every game in the site's games.jsonl and dump per-turn cumulative
//! scores plus opening notations, for score-dynamics mining.
//!
//! Usage: scores [--games PATH] [--out PATH]
//!
//! One output line per input line (same order), so Python can join on index:
//!   {"i":n,"id":…,"kind":…,"result":…,"termination":…,"rated":…,
//!    "turns":[[blue,red],…],"moves":n,"finalArea":[b,r],"first":["D10-D13",…],
//!    "opening":[ids],"blue":…,"red":…,"matchId":…,"tournamentId":…,"index":k,
//!    "replayed":[b,r],"fileScore":[b,r],"scoreMatch":bool}
//! or on failure: {"i":n,"id":…,"error":"…"}.

use std::io::Write;

use meridian_engine::{Game, Move, Player, notation};

fn main() {
    let mut games_path =
        "/tmp/opencode/meridian-games/meridian-games/games.jsonl".to_string();
    let mut out_path = "/tmp/opencode/meridian-games/turn-scores.jsonl".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--games" => games_path = args.next().unwrap_or(games_path),
            "--out" => out_path = args.next().unwrap_or(out_path),
            _ => eprintln!("unknown arg {arg}"),
        }
    }

    let text = std::fs::read_to_string(&games_path)
        .unwrap_or_else(|error| panic!("read {games_path}: {error}"));
    let mut out = std::fs::File::create(&out_path)
        .unwrap_or_else(|error| panic!("create {out_path}: {error}"));

    let mut ok = 0usize;
    let mut failed = 0usize;
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let id = value["id"].as_str().unwrap_or_default().to_string();
        let record = match replay(i, &value) {
            Ok(record) => {
                ok += 1;
                record
            }
            Err(error) => {
                failed += 1;
                serde_json::json!({"i": i, "id": id, "error": error})
            }
        };
        let mut line = serde_json::to_string(&record).unwrap();
        line.push('\n');
        out.write_all(line.as_bytes()).unwrap();
    }
    eprintln!("scores: replayed {ok} games, {failed} failed");
}

fn player_summary(value: &serde_json::Value, color: &str) -> serde_json::Value {
    let p = &value[color];
    serde_json::json!({
        "type": p["type"],
        "id": p.get("id").cloned().unwrap_or(serde_json::Value::Null),
        "name": p["name"],
        "kind": p.get("kind").cloned().unwrap_or(serde_json::Value::Null),
    })
}

fn replay(i: usize, value: &serde_json::Value) -> Result<serde_json::Value, String> {
    let start = value["start"].as_str().ok_or("no start")?;
    let start_position =
        notation::parse_setup(start).map_err(|e| format!("parse start: {e:?}"))?;
    let mut game = Game::from_position(start_position);
    let ids: Vec<usize> = value["moves"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
        .unwrap_or_default();
    let mut turns: Vec<(f64, f64)> = Vec::new();
    let mut first_text: Vec<String> = Vec::new();
    for (n, &id) in ids.iter().enumerate() {
        let mv = Move::from_index(id).ok_or_else(|| format!("bad move id {id} at {n}"))?;
        if n < 4 {
            if let Some(target) = mv.target() {
                first_text.push(notation::move_text(mv.source, target));
            }
        }
        let outcome = game
            .play(mv)
            .map_err(|e| format!("illegal move at {n}: {e:?}"))?;
        if outcome.turn_ended || outcome.game_over {
            let position = game.position();
            turns.push((
                position.score(Player::Blue).to_f64(),
                position.score(Player::Red).to_f64(),
            ));
        }
    }
    let position = game.position();
    let replayed = [
        position.score(Player::Blue).to_f64(),
        position.score(Player::Red).to_f64(),
    ];
    let file_score = value["score"].as_object().map(|s| {
        [
            s["blue"].as_f64().unwrap_or(f64::NAN),
            s["red"].as_f64().unwrap_or(f64::NAN),
        ]
    });
    let score_match = file_score.is_some_and(|f| {
        (replayed[0] - f[0]).abs() < 0.01 && (replayed[1] - f[1]).abs() < 0.01
    });
    Ok(serde_json::json!({
        "i": i,
        "id": value["id"],
        "kind": value["kind"],
        "result": value["result"],
        "termination": value["termination"],
        "rated": value["rated"],
        "blue": player_summary(value, "blue"),
        "red": player_summary(value, "red"),
        "matchId": value.get("matchId").cloned().unwrap_or(serde_json::Value::Null),
        "tournamentId": value.get("tournamentId").cloned().unwrap_or(serde_json::Value::Null),
        "index": value.get("index").cloned().unwrap_or(serde_json::Value::Null),
        "opening": value.get("opening").cloned().unwrap_or(serde_json::Value::Null),
        "moves": ids.len(),
        "turns": turns,
        "first": first_text,
        "finalArea": [
            position.area(Player::Blue).to_f64(),
            position.area(Player::Red).to_f64(),
        ],
        "replayed": replayed,
        "scoreMatch": score_match,
    }))
}
