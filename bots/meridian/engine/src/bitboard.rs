//! A set of board points: one `u32` per row, with a margin of empty rows and columns.
//!
//! The margin is three points wide on every side, the reach of an edge. Because of it, the 7 × 7
//! window around any board point can be read without bounds checks, and everything off the board
//! reads as empty.

use std::fmt;

use crate::geometry::{BOARD_SIDE, Point, REACH, WINDOW_SIDE};

const MARGIN: usize = REACH as usize;
const ROWS_WITH_MARGIN: usize = BOARD_SIDE + 2 * MARGIN;
const WINDOW_ROW: u32 = (1 << WINDOW_SIDE) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bitboard {
    rows: [u32; ROWS_WITH_MARGIN],
}

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard { rows: [0; ROWS_WITH_MARGIN] };

    #[inline]
    pub const fn contains(&self, point: Point) -> bool {
        self.rows[point.row() + MARGIN] >> (point.col() + MARGIN) & 1 != 0
    }

    #[inline]
    pub fn insert(&mut self, point: Point) {
        self.rows[point.row() + MARGIN] |= 1 << (point.col() + MARGIN);
    }

    #[inline]
    pub fn remove(&mut self, point: Point) {
        self.rows[point.row() + MARGIN] &= !(1 << (point.col() + MARGIN));
    }

    pub fn len(&self) -> usize {
        self.rows.iter().map(|row| row.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(|&row| row == 0)
    }

    /// The points in index order, row by row from the bottom.
    pub fn iter(&self) -> impl Iterator<Item = Point> + '_ {
        (0..BOARD_SIDE).flat_map(move |row| {
            let mut columns = self.row_bits(row);
            std::iter::from_fn(move || {
                if columns == 0 {
                    return None;
                }
                let col = columns.trailing_zeros() as usize;
                columns &= columns - 1;
                Some(Point::from_row_col(row, col))
            })
        })
    }

    /// The columns present in a board row: bit `c` is column `c`.
    #[inline]
    pub(crate) const fn row_bits(&self, row: usize) -> u32 {
        self.rows[row + MARGIN] >> MARGIN
    }

    /// The 7 × 7 window around `center` in [`DirSet`](crate::geometry::DirSet) layout: bit
    /// `(dy + 3) * 7 + (dx + 3)` is set when `center + (dx, dy)` is in the set.
    #[inline]
    pub(crate) fn window(&self, center: Point) -> u64 {
        let mut window = 0u64;
        for window_row in 0..WINDOW_SIDE as usize {
            // Stored row `center.row() + window_row` is board row `center.row() + window_row - 3`,
            // and shifting by `center.col()` brings board column `center.col() - 3` to bit 0.
            let columns = (self.rows[center.row() + window_row] >> center.col()) & WINDOW_ROW;
            window |= (columns as u64) << (window_row as u32 * WINDOW_SIDE);
        }
        window
    }
}

impl Default for Bitboard {
    fn default() -> Self {
        Bitboard::EMPTY
    }
}

impl FromIterator<Point> for Bitboard {
    fn from_iter<I: IntoIterator<Item = Point>>(points: I) -> Self {
        let mut set = Bitboard::EMPTY;
        for point in points {
            set.insert(point);
        }
        set
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
