//! The rules of Meridian, exactly: legal moves, playing them, and scoring.
//!
//! ```
//! use meridian_engine::Game;
//!
//! let mut game = Game::new();
//! while !game.is_over() {
//!     let first = game.legal_moves().iter().next().unwrap();
//!     game.play(first).unwrap();
//! }
//! println!("{:?} after {} moves", game.outcome().unwrap(), game.moves().len());
//! ```
//!
//! Every rule is decided with integers; nothing is rounded. `README.md` explains how it works.

#![forbid(unsafe_code)]

pub mod area;
pub mod bitboard;
pub mod edge;
pub mod game;
pub mod geometry;
pub mod movegen;
pub mod moves;
pub mod notation;
pub mod position;
mod tables;
pub mod units;

pub use area::enclosed_area;
pub use bitboard::Bitboard;
pub use edge::{Edge, EdgeSet};
pub use game::Game;
pub use geometry::{DirSet, Direction, Point};
pub use movegen::LegalMoves;
pub use moves::{Move, NUM_MOVES};
pub use notation::{Line, NotationError};
pub use position::{DrawReason, IllegalMove, InvalidPosition, MoveKind, MoveOutcome, Outcome, Position};
pub use units::{Area, Player, Score};
