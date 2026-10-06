//! A game in progress: a position, its legal moves, and the moves played.

use crate::movegen::LegalMoves;
use crate::moves::Move;
use crate::position::{IllegalMove, MoveOutcome, Outcome, Position, TOTAL_ACTIONS};

/// A position that keeps its legal moves and result up to date as moves are played.
#[derive(Clone, Debug)]
pub struct Game {
    position: Position,
    legal: LegalMoves,
    outcome: Option<Outcome>,
    moves: Vec<Move>,
}

impl Default for Game {
    fn default() -> Self {
        Game::new()
    }
}

impl Game {
    pub fn new() -> Game {
        Game::from_position(Position::new())
    }

    /// A game that continues from `position`. [`Game::moves`] are the moves played from it.
    pub fn from_position(position: Position) -> Game {
        let legal = position.legal_moves();
        let outcome = position.outcome_given(&legal);
        Game { position, legal, outcome, moves: Vec::with_capacity(usize::from(TOTAL_ACTIONS)) }
    }

    /// Starts again from the starting position, keeping the memory for the moves.
    pub fn reset(&mut self) {
        self.position = Position::new();
        self.legal = self.position.legal_moves();
        self.outcome = None;
        self.moves.clear();
    }

    #[inline]
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Empty once the game is over.
    #[inline]
    pub fn legal_moves(&self) -> &LegalMoves {
        &self.legal
    }

    #[inline]
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// Refuses an illegal move, saying why.
    pub fn play(&mut self, mv: Move) -> Result<MoveOutcome, IllegalMove> {
        if !self.legal.contains(mv) {
            return Err(self.position.check_move(mv).expect_err("the generator and the checker agree"));
        }
        let outcome = self.position.apply_unchecked(mv);
        self.moves.push(mv);
        self.legal = self.position.legal_moves();
        self.outcome = self.position.outcome_given(&self.legal);
        Ok(outcome)
    }
}
