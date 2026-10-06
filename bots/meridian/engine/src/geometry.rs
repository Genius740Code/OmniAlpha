//! Points, edge directions, and the exact tests for how two segments meet.
//!
//! The board is the 19 × 19 lattice `x, y ∈ -9..=9`. An edge joins two points at most 3 king
//! steps apart, so there are 48 possible edge vectors, called directions.
//!
//! # The window layout
//!
//! The 48 directions are the cells of the 7 × 7 box around a point, minus its centre. A
//! [`DirSet`] stores one bit per cell of that box, bit `(dy + 3) * 7 + (dx + 3)` for the vector
//! `(dx, dy)`. [`Bitboard::window`](crate::bitboard::Bitboard) reads the same box out of a set of
//! points in the same layout, so "which neighbours are occupied" and "which directions are
//! allowed" combine with plain bitwise operations. Move generation is built on this.

use std::fmt;

use crate::units::gcd;

pub const BOARD_SIDE: usize = 19;
pub const NUM_POINTS: usize = BOARD_SIDE * BOARD_SIDE;
/// The board spans `-COORD_LIMIT..=COORD_LIMIT` on both axes.
pub const COORD_LIMIT: i8 = 9;
/// The longest edge, in king steps.
pub const REACH: i8 = 3;
pub const NUM_DIRECTIONS: usize = 48;
/// An undirected edge points one of two opposite ways; the canonical one is half of them.
pub const NUM_CANONICAL_DIRECTIONS: usize = NUM_DIRECTIONS / 2;

pub(crate) const WINDOW_SIDE: u32 = 2 * REACH as u32 + 1;
pub(crate) const WINDOW_CELLS: u32 = WINDOW_SIDE * WINDOW_SIDE;
/// The window's centre is the point itself, never a direction.
pub(crate) const WINDOW_CENTER: u32 = WINDOW_CELLS / 2;

/// A point of the board, stored as its index `(y + 9) * 19 + (x + 9)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Point(u16);

impl Point {
    pub const fn new(x: i8, y: i8) -> Option<Point> {
        if x < -COORD_LIMIT || x > COORD_LIMIT || y < -COORD_LIMIT || y > COORD_LIMIT {
            return None;
        }
        Some(Point::from_row_col((y + COORD_LIMIT) as usize, (x + COORD_LIMIT) as usize))
    }

    pub const fn from_index(index: usize) -> Option<Point> {
        if index < NUM_POINTS { Some(Point(index as u16)) } else { None }
    }

    #[inline]
    pub(crate) const fn from_row_col(row: usize, col: usize) -> Point {
        debug_assert!(row < BOARD_SIDE && col < BOARD_SIDE);
        Point((row * BOARD_SIDE + col) as u16)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// `y + 9`: row 0 is the bottom of the board.
    #[inline]
    pub const fn row(self) -> usize {
        self.0 as usize / BOARD_SIDE
    }

    /// `x + 9`: column 0 is the left side of the board.
    #[inline]
    pub const fn col(self) -> usize {
        self.0 as usize % BOARD_SIDE
    }

    #[inline]
    pub const fn x(self) -> i8 {
        self.col() as i8 - COORD_LIMIT
    }

    #[inline]
    pub const fn y(self) -> i8 {
        self.row() as i8 - COORD_LIMIT
    }

    /// The point `(x + dx, y + dy)`, if it is on the board.
    #[inline]
    pub const fn offset(self, dx: i8, dy: i8) -> Option<Point> {
        let col = self.col() as i16 + dx as i16;
        let row = self.row() as i16 + dy as i16;
        if col < 0 || col >= BOARD_SIDE as i16 || row < 0 || row >= BOARD_SIDE as i16 {
            return None;
        }
        Some(Point::from_row_col(row as usize, col as usize))
    }

    #[inline]
    pub const fn step(self, direction: Direction) -> Option<Point> {
        self.offset(direction.dx(), direction.dy())
    }

    #[inline]
    pub(crate) const fn to_vector(self) -> Vector {
        Vector { x: self.x() as i32, y: self.y() as i32 }
    }
}

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.x(), self.y())
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// One of the 48 edge vectors `(dx, dy)`, `1 <= max(|dx|, |dy|) <= 3`.
///
/// Directions are numbered `0..48` in window order: by `dy`, then by `dx`, skipping the centre.
/// So `d` and `47 - d` are opposites, and `24..48` are the canonical directions: those pointing
/// up, or right along a row.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Direction(u8);

impl Direction {
    pub const fn from_index(index: usize) -> Option<Direction> {
        if index < NUM_DIRECTIONS { Some(Direction(index as u8)) } else { None }
    }

    pub const fn from_delta(dx: i8, dy: i8) -> Option<Direction> {
        if dx < -REACH || dx > REACH || dy < -REACH || dy > REACH || (dx == 0 && dy == 0) {
            return None;
        }
        let bit = (dy + REACH) as u32 * WINDOW_SIDE + (dx + REACH) as u32;
        Some(Direction::from_window_bit(bit))
    }

    pub fn all() -> impl Iterator<Item = Direction> {
        (0..NUM_DIRECTIONS as u8).map(Direction)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn dx(self) -> i8 {
        (self.window_bit() % WINDOW_SIDE) as i8 - REACH
    }

    #[inline]
    pub const fn dy(self) -> i8 {
        (self.window_bit() / WINDOW_SIDE) as i8 - REACH
    }

    #[inline]
    pub const fn opposite(self) -> Direction {
        Direction(NUM_DIRECTIONS as u8 - 1 - self.0)
    }

    /// Whether this is the one of `{d, d.opposite()}` that identifies an undirected edge.
    #[inline]
    pub const fn is_canonical(self) -> bool {
        self.0 as usize >= NUM_CANONICAL_DIRECTIONS
    }

    /// How many unit lattice steps the vector is made of, `gcd(|dx|, |dy|)`. An edge in this
    /// direction passes through `steps() - 1` lattice points between its ends.
    #[inline]
    pub const fn steps(self) -> i8 {
        gcd(self.dx().unsigned_abs() as i64, self.dy().unsigned_abs() as i64) as i8
    }

    /// Its bit in the window: its index, skipping the centre.
    #[inline]
    pub(crate) const fn window_bit(self) -> u32 {
        let index = self.0 as u32;
        index + (index >= WINDOW_CENTER) as u32
    }

    #[inline]
    pub(crate) const fn from_window_bit(bit: u32) -> Direction {
        debug_assert!(bit < WINDOW_CELLS && bit != WINDOW_CENTER);
        Direction((bit - (bit > WINDOW_CENTER) as u32) as u8)
    }

    #[inline]
    pub(crate) const fn to_vector(self) -> Vector {
        Vector { x: self.dx() as i32, y: self.dy() as i32 }
    }
}

impl fmt::Debug for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:+},{:+}>", self.dx(), self.dy())
    }
}

/// A set of directions, one bit per cell of the 7 × 7 window (see the module header).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DirSet(u64);

impl DirSet {
    pub const EMPTY: DirSet = DirSet(0);
    pub const ALL: DirSet = DirSet(((1 << WINDOW_CELLS) - 1) & !(1 << WINDOW_CENTER));

    /// Keeps only the bits that are directions.
    #[inline]
    pub(crate) const fn from_bits(bits: u64) -> DirSet {
        DirSet(bits & Self::ALL.0)
    }

    /// The window bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn contains(self, direction: Direction) -> bool {
        self.0 >> direction.window_bit() & 1 != 0
    }

    #[inline]
    pub fn insert(&mut self, direction: Direction) {
        self.0 |= 1 << direction.window_bit();
    }

    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The `n`-th direction in index order.
    pub fn nth(self, n: u32) -> Option<Direction> {
        self.iter().nth(n as usize)
    }

    #[inline]
    pub fn iter(self) -> DirSetIter {
        DirSetIter(self.0)
    }
}

impl IntoIterator for DirSet {
    type Item = Direction;
    type IntoIter = DirSetIter;
    fn into_iter(self) -> DirSetIter {
        self.iter()
    }
}

impl fmt::Debug for DirSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[derive(Clone)]
pub struct DirSetIter(u64);

impl Iterator for DirSetIter {
    type Item = Direction;

    #[inline]
    fn next(&mut self) -> Option<Direction> {
        if self.0 == 0 {
            return None;
        }
        let lowest = self.0.trailing_zeros();
        self.0 &= self.0 - 1;
        Some(Direction::from_window_bit(lowest))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.0.count_ones() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DirSetIter {}

/// A point or vector with integer coordinates. Every geometric rule is decided on these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Vector {
    pub x: i32,
    pub y: i32,
}

impl Vector {
    #[inline]
    pub const fn minus(self, other: Vector) -> Vector {
        Vector { x: self.x - other.x, y: self.y - other.y }
    }
}

#[inline]
pub(crate) const fn cross(a: Vector, b: Vector) -> i32 {
    a.x * b.y - a.y * b.x
}

/// Which side of the line through `a` and `b` the point `p` is on: 1 for the left, -1 for the
/// right, 0 when all three are on one line.
#[inline]
fn side(a: Vector, b: Vector, p: Vector) -> i32 {
    cross(b.minus(a), p.minus(a)).signum()
}

/// For a point `p` already known to lie on the line through `a` and `b`: whether it lies on the
/// segment between them.
#[inline]
fn within_segment(a: Vector, b: Vector, p: Vector) -> bool {
    p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x) && p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
}

/// Whether the segments `ab` and `cd`, ends included, share a point. This is what the rules
/// call touching: crossing, an end landing on the other segment, and shared ends all count.
pub(crate) fn segments_touch(a: Vector, b: Vector, c: Vector, d: Vector) -> bool {
    let (c_side, d_side) = (side(a, b, c), side(a, b, d));
    let (a_side, b_side) = (side(c, d, a), side(c, d, b));
    if c_side != d_side && a_side != b_side {
        return true;
    }
    (c_side == 0 && within_segment(a, b, c))
        || (d_side == 0 && within_segment(a, b, d))
        || (a_side == 0 && within_segment(c, d, a))
        || (b_side == 0 && within_segment(c, d, b))
}

/// Whether the segments `ab` and `cd` lie on one line and share more than a single point.
pub(crate) fn segments_overlap(a: Vector, b: Vector, c: Vector, d: Vector) -> bool {
    if side(a, b, c) != 0 || side(a, b, d) != 0 {
        return false;
    }
    // On one line: compare the two intervals along an axis the line is not perpendicular to.
    let along = |p: Vector| if a.x != b.x { p.x } else { p.y };
    let start = along(a).min(along(b)).max(along(c).min(along(d)));
    let end = along(a).max(along(b)).min(along(c).max(along(d)));
    end > start
}

/// Whether `p` lies on the segment `ab` other than at its ends.
pub(crate) fn strictly_inside(a: Vector, b: Vector, p: Vector) -> bool {
    p != a && p != b && side(a, b, p) == 0 && within_segment(a, b, p)
}

/// Whether the segments `ab` and `cd` cross at a single point inside both: not at an end of
/// either, and not along a shared line.
pub fn cross_properly(a: Point, b: Point, c: Point, d: Point) -> bool {
    let (a, b, c, d) = (a.to_vector(), b.to_vector(), c.to_vector(), d.to_vector());
    side(a, b, c) * side(a, b, d) < 0 && side(c, d, a) * side(c, d, b) < 0
}
