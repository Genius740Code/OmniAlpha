//! Replay site games with the real engine and dump training samples.
//!
//! Two modes (see `--help` output at the bottom of main):
//!
//! * dump: one `.bin` file per game into `--out`, named by the `_sha` field the
//!   extractor injects into each JSON line. Every decision position (the
//!   position before each recorded move, while the game is unfinished) becomes
//!   one fixed-size record:
//!
//!   | bytes | contents                                                     |
//!   |-------|--------------------------------------------------------------|
//!   | 414   | binary planes 0..8, bit-packed: plane p, point i -> byte p*46 + i/8, bit i%8 |
//!   | 20    | scalar planes 9..13 as f32 (blue/red score /100, blue/red area /100, actions/120) |
//!   | 2     | played move id, u16 LE                                        |
//!   | 4     | final margin (blue - red, site score), f32 LE                  |
//!   | 1     | ply index (moves already played), u8                          |
//!   | 1     | flags: bits 0..2 game kind, bits 3..4 termination              |
//!
//!   The planes are produced by `nn::input_planes` itself, so they byte-match
//!   what the Rust net sees at inference. Planes 9..13 are constant per
//!   position, so only the five f32 values are stored.
//!
//! * dense: writes the first `--dense-max` decision positions (with
//!   `--skip` plies skipped, matching the `opening_len` field the harness uses)
//!   of the games in `--dense-games`, in file order, as
//!   `IN_PLANES*POINTS + 1` little-endian f32 each: the 5054 plane values then
//!   the value target `tanh(margin / 1500)`. This is the format `nnbench
//!   --check` expects (`val_dense_head.bin`).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use meridian_engine::{Move, Position};
use nn::{IN_PLANES, MARGIN_SCALE, POINTS, input_planes};

/// One packed record: 414 + 20 + 2 + 4 + 1 + 1.
const REC: usize = 414 + 20 + 2 + 4 + 1 + 1;

fn main() {
    let mut games = PathBuf::new();
    let mut out = PathBuf::new();
    let mut dense_games = PathBuf::new();
    let mut dense_out = PathBuf::new();
    let mut dense_max = 0usize;
    let mut skip = 0usize;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => games = args[i + 1].clone().into(),
            "--out" => out = args[i + 1].clone().into(),
            "--dense-games" => dense_games = args[i + 1].clone().into(),
            "--dense-out" => dense_out = args[i + 1].clone().into(),
            "--dense-max" => dense_max = args[i + 1].parse().unwrap(),
            "--skip" => skip = args[i + 1].parse().unwrap(),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }

    if !dense_games.as_os_str().is_empty() {
        dense_mode(&dense_games, &dense_out, dense_max, skip);
        return;
    }
    if games.as_os_str().is_empty() || out.as_os_str().is_empty() {
        eprintln!(
            "usage: dump --games GAMES.jsonl --out OUT_DIR [--dense-games F --dense-out F --dense-max N --skip P]"
        );
        std::process::exit(2);
    }
    dump_mode(&games, &out);
}

struct GameLine {
    moves: Vec<Move>,
    margin: f64,
    kind: u8,
    term: u8,
    sha: String,
}

fn parse_line(line: &str) -> Option<GameLine> {
    let g: serde_json::Value = serde_json::from_str(line).ok()?;
    let moves = g["moves"]
        .as_array()?
        .iter()
        .map(|id| id.as_u64().and_then(|i| usize::try_from(i).ok()).and_then(Move::from_index))
        .collect::<Option<Vec<_>>>()?;
    let score = &g["score"];
    let margin = score["blue"].as_f64()? - score["red"].as_f64()?;
    let kind = match g["kind"].as_str() {
        Some("evaluation") => 0,
        Some("human") => 1,
        Some("selfplay") => 2,
        Some("versus") => 3,
        _ => 0,
    };
    let term = match g["termination"].as_str() {
        Some("resignation") => 1,
        Some("forfeit") => 2,
        _ => 0,
    };
    Some(GameLine { moves, margin, kind, term, sha: g["_sha"].as_str()?.to_string() })
}

/// Pushes one packed record for `position`.
fn push_record(position: &Position, played: usize, margin: f64, ply: usize, flags: u8, out: &mut Vec<u8>) {
    let mut planes = [0f32; IN_PLANES * POINTS];
    input_planes(position, &mut planes);
    // Binary planes 0..8, one bit per point, 361 bits per plane.
    for p in 0..9 {
        let mut bits = [0u8; 46];
        for (pt, &v) in planes[p * POINTS..p * POINTS + POINTS].iter().enumerate() {
            if v != 0.0 {
                bits[pt >> 3] |= 1 << (pt & 7);
            }
        }
        out.extend_from_slice(&bits);
    }
    // The five constant planes: store the plane's value once, as the exact f32
    // the engine produced.
    for p in 9..IN_PLANES {
        out.extend_from_slice(&planes[p * POINTS].to_le_bytes());
    }
    out.extend_from_slice(&(played as u16).to_le_bytes());
    out.extend_from_slice(&(margin as f32).to_le_bytes());
    out.push(ply.min(255) as u8);
    out.push(flags);
}

fn dump_mode(games: &PathBuf, out_dir: &PathBuf) {
    std::fs::create_dir_all(out_dir).expect("create out dir");
    let file = File::open(games).expect("games jsonl");
    let reader = BufReader::new(file);
    let stdout = std::io::stdout();
    let mut meta = BufWriter::new(stdout.lock());
    for line in reader.lines() {
        let line = line.expect("read line");
        let game = match parse_line(&line) {
            Some(game) => game,
            None => {
                writeln!(meta, "{{\"ok\":false,\"err\":\"parse\",\"n\":0}}").unwrap();
                continue;
            }
        };
        let flags = game.kind | game.term << 3;
        let mut buf: Vec<u8> = Vec::with_capacity(REC * game.moves.len());
        let mut position = Position::new();
        let mut ok = true;
        let mut n = 0usize;
        for (i, &mv) in game.moves.iter().enumerate() {
            if position.is_finished() {
                break;
            }
            push_record(&position, mv.index(), game.margin, i, flags, &mut buf);
            n += 1;
            if position.apply(mv).is_err() {
                ok = false;
                break;
            }
        }
        assert_eq!(buf.len(), n * REC, "record size invariant");
        let path = out_dir.join(format!("{}.bin", game.sha));
        std::fs::write(&path, &buf).expect("write game bin");
        writeln!(
            meta,
            "{{\"ok\":{ok},\"sha\":\"{}\",\"n\":{n},\"margin\":{}}}",
            game.sha, game.margin
        )
        .unwrap();
    }
}

/// The dense format `nnbench --check` replays: planes then target, f32 LE.
fn dense_mode(games: &PathBuf, out: &PathBuf, max: usize, skip: usize) {
    let file = File::open(games).expect("dense games jsonl");
    let reader = BufReader::new(file);
    let mut writer = BufWriter::with_capacity(1 << 20, File::create(out).expect("dense out"));
    let mut count = 0usize;
    'outer: for line in reader.lines() {
        let line = line.expect("read line");
        let Some(game) = parse_line(&line) else { continue };
        let mut position = Position::new();
        let mut planes = [0f32; IN_PLANES * POINTS];
        for (i, &mv) in game.moves.iter().enumerate() {
            if position.is_finished() || count >= max {
                break;
            }
            if i >= skip {
                input_planes(&position, &mut planes);
                for &v in planes.iter() {
                    writer.write_all(&v.to_le_bytes()).unwrap();
                }
                let target = ((game.margin / MARGIN_SCALE) as f32).tanh();
                writer.write_all(&target.to_le_bytes()).unwrap();
                count += 1;
            }
            if position.apply(mv).is_err() {
                break 'outer; // The harness stops at the first illegal move too.
            }
        }
        if count >= max {
            break;
        }
    }
    writer.flush().unwrap();
    eprintln!("dense: wrote {count} samples ({}) to {}", max, out.display());
}
