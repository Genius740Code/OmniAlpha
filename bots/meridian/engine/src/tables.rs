//! Lookup tables for move generation, built once from the exact tests in `geometry`.
//!
//! The tables answer questions about the 48 candidate edges from a source point, each answer a
//! set of directions in window layout:
//!
//! * which candidates end on the board;
//! * which candidates pass through a given cell of the window, and so are blocked by a node there;
//! * how an existing edge, at a given offset from the source, meets each candidate: whether it
//!   touches the candidate, and whether it runs along it.
//!
//! They also number the angles of the edge directions, for the area computation.

use std::sync::OnceLock;

use crate::edge::Edge;
use crate::geometry::{
    Direction, NUM_CANONICAL_DIRECTIONS, NUM_DIRECTIONS, NUM_POINTS, Point, REACH, Vector, WINDOW_CELLS, cross,
    segments_overlap, segments_touch,
};

/// Edge directions fall into this many angles, one per primitive vector such as (1,0) or (3,2).
pub(crate) const ANGLE_CLASSES: u32 = 32;

/// An edge can meet a candidate only if some point of it is within reach of the source, so its
/// origin is at most two reaches away in each coordinate.
const NEARBY: i32 = 2 * REACH as i32;
const NEARBY_SIDE: usize = 2 * NEARBY as usize + 1;

/// How one existing edge meets the candidates from a source, as sets of directions.
#[derive(Clone, Copy, Default)]
pub(crate) struct Contact {
    /// Candidates that share at least one point with the edge.
    pub touch: u64,
    /// Candidates that lie along the edge, sharing more than one point.
    pub overlap: u64,
}

pub(crate) struct Tables {
    /// Per source point: the candidates that end on the board.
    ends_on_board: [u64; NUM_POINTS],
    /// Per window cell: the candidates that pass through that cell between their ends.
    passing_through: [u64; WINDOW_CELLS as usize],
    /// The window cells some candidate passes through.
    cells_passed_through: u64,
    /// Per nearby edge position, see [`contact_index`].
    contacts: Vec<Contact>,
    /// Per direction: its angle, counting counter-clockwise from the positive x axis in
    /// `0..ANGLE_CLASSES`. Parallel directions share an angle, and opposite ones are exactly
    /// `ANGLE_CLASSES / 2` apart.
    angle_classes: [u8; NUM_DIRECTIONS],
}

#[inline]
pub(crate) fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(build)
}

impl Tables {
    #[inline]
    pub fn ends_on_board(&self, source: Point) -> u64 {
        self.ends_on_board[source.index()]
    }

    /// The candidates that pass through a node, given the window of nodes around the source.
    #[inline]
    pub fn blocked_by(&self, nodes_window: u64) -> u64 {
        let mut blocked = 0;
        let mut cells = nodes_window & self.cells_passed_through;
        while cells != 0 {
            blocked |= self.passing_through[cells.trailing_zeros() as usize];
            cells &= cells - 1;
        }
        blocked
    }

    #[inline]
    pub fn angle_class(&self, direction: Direction) -> u8 {
        self.angle_classes[direction.index()]
    }

    /// How the edge whose origin is `(origin_dx, origin_dy)` away from the source, pointing in
    /// `canonical`, meets the source's candidates. Both offsets must be within `2 * REACH`.
    #[inline]
    pub fn contact(&self, origin_dx: i32, origin_dy: i32, canonical: Direction) -> Contact {
        self.contacts[contact_index(origin_dx, origin_dy, canonical)]
    }

    /// Whether the candidate from `source` in `direction` shares a point with `edge`.
    #[inline]
    pub fn touches(&self, source: Point, direction: Direction, edge: Edge) -> bool {
        let origin = edge.origin();
        let origin_dx = origin.col() as i32 - source.col() as i32;
        let origin_dy = origin.row() as i32 - source.row() as i32;
        origin_dx.abs() <= NEARBY
            && origin_dy.abs() <= NEARBY
            && self.contact(origin_dx, origin_dy, edge.canonical_direction()).touch >> direction.window_bit() & 1 != 0
    }
}

#[inline]
fn contact_index(origin_dx: i32, origin_dy: i32, canonical: Direction) -> usize {
    debug_assert!(origin_dx.abs() <= NEARBY && origin_dy.abs() <= NEARBY && canonical.is_canonical());
    let offset = (origin_dy + NEARBY) as usize * NEARBY_SIDE + (origin_dx + NEARBY) as usize;
    offset * NUM_CANONICAL_DIRECTIONS + (canonical.index() - NUM_CANONICAL_DIRECTIONS)
}

fn build() -> Tables {
    let mut ends_on_board = [0u64; NUM_POINTS];
    for (index, candidates) in ends_on_board.iter_mut().enumerate() {
        let source = Point::from_index(index).expect("a board index");
        for direction in Direction::all().filter(|&direction| source.step(direction).is_some()) {
            *candidates |= 1 << direction.window_bit();
        }
    }

    let mut passing_through = [0u64; WINDOW_CELLS as usize];
    for direction in Direction::all() {
        let steps = direction.steps();
        for step in 1..steps {
            let cell = Direction::from_delta(direction.dx() / steps * step, direction.dy() / steps * step)
                .expect("part of an edge vector is an edge vector");
            passing_through[cell.window_bit() as usize] |= 1 << direction.window_bit();
        }
    }
    let cells_passed_through =
        (0..WINDOW_CELLS).filter(|&cell| passing_through[cell as usize] != 0).fold(0, |cells, cell| cells | 1 << cell);

    let source = Vector { x: 0, y: 0 };
    let mut contacts = vec![Contact::default(); NEARBY_SIDE * NEARBY_SIDE * NUM_CANONICAL_DIRECTIONS];
    for origin_dy in -NEARBY..=NEARBY {
        for origin_dx in -NEARBY..=NEARBY {
            for canonical in Direction::all().filter(|direction| direction.is_canonical()) {
                let start = Vector { x: origin_dx, y: origin_dy };
                let end = Vector { x: origin_dx + canonical.dx() as i32, y: origin_dy + canonical.dy() as i32 };
                let contact = &mut contacts[contact_index(origin_dx, origin_dy, canonical)];
                for candidate in Direction::all() {
                    let bit = 1u64 << candidate.window_bit();
                    if segments_touch(source, candidate.to_vector(), start, end) {
                        contact.touch |= bit;
                    }
                    if segments_overlap(source, candidate.to_vector(), start, end) {
                        contact.overlap |= bit;
                    }
                }
            }
        }
    }

    Tables { ends_on_board, passing_through, cells_passed_through, contacts, angle_classes: angle_classes() }
}

fn angle_classes() -> [u8; NUM_DIRECTIONS] {
    let upper_half = |v: Vector| v.y > 0 || (v.y == 0 && v.x > 0);
    let mut primitive: Vec<Direction> = Direction::all().filter(|direction| direction.steps() == 1).collect();
    // Counter-clockwise from the positive x axis: the upper half first, then by cross product,
    // which orders any two vectors less than half a turn apart.
    primitive.sort_by(|a, b| {
        let (a, b) = (a.to_vector(), b.to_vector());
        upper_half(b).cmp(&upper_half(a)).then(0.cmp(&cross(a, b)))
    });
    assert_eq!(primitive.len() as u32, ANGLE_CLASSES);
    let mut classes = [0u8; NUM_DIRECTIONS];
    for direction in Direction::all() {
        let steps = direction.steps();
        let unit = Direction::from_delta(direction.dx() / steps, direction.dy() / steps).expect("a unit step");
        classes[direction.index()] = primitive.iter().position(|&p| p == unit).expect("a primitive vector") as u8;
    }
    classes
}
