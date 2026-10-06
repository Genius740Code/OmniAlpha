//! Moves, and the move IDs the site sends and expects.

use std::fmt;

use crate::geometry::{Direction, NUM_DIRECTIONS, NUM_POINTS, Point};

/// Every pair of a source point and a direction, including those that leave the board.
pub const NUM_MOVES: usize = NUM_DIRECTIONS * NUM_POINTS;

/// A new edge from one of the mover's nodes. What it does, extend, connect or capture, depends
/// on the position; see [`MoveKind`](crate::MoveKind).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub source: Point,
    pub direction: Direction,
}

impl Move {
    /// The move from `source` to `target`, if they are 1 to 3 king steps apart.
    pub fn between(source: Point, target: Point) -> Option<Move> {
        let dx = target.x().checked_sub(source.x())?;
        let dy = target.y().checked_sub(source.y())?;
        Some(Move { source, direction: Direction::from_delta(dx, dy)? })
    }

    /// `None` when the edge would end off the board.
    #[inline]
    pub const fn target(self) -> Option<Point> {
        self.source.step(self.direction)
    }

    /// The move's ID on the site: `direction.index() * 361 + source.index()`, in `0..NUM_MOVES`.
    #[inline]
    pub const fn index(self) -> usize {
        self.direction.index() * NUM_POINTS + self.source.index()
    }

    pub fn from_index(index: usize) -> Option<Move> {
        Some(Move {
            source: Point::from_index(index % NUM_POINTS)?,
            direction: Direction::from_index(index / NUM_POINTS)?,
        })
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target() {
            Some(target) => write!(f, "{:?}->{:?}", self.source, target),
            None => write!(f, "{:?}->off-board{:?}", self.source, self.direction),
        }
    }
}
