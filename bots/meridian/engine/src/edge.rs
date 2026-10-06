//! Edges, and the set of edges one player owns.
//!
//! An edge is stored as a 16-bit slot, `origin.index() * 24 + canonical direction`. Its origin is
//! the end from which it points in a canonical direction: the lower end, or the left one when the
//! edge is level. Every edge has exactly one slot, so a sorted list of slots is a canonical edge
//! set: two positions with the same edges are equal, whatever order the edges were placed in.

use std::fmt;
use std::hash::{Hash, Hasher};

use crate::geometry::{BOARD_SIDE, Direction, NUM_CANONICAL_DIRECTIONS, NUM_POINTS, Point};

/// The starting edge plus one per action: 60 each.
pub const MAX_EDGES_PER_PLAYER: usize = 61;
/// The two starting nodes plus at most one per action.
pub const MAX_NODES_PER_PLAYER: usize = 62;

const NUM_SLOTS: usize = NUM_POINTS * NUM_CANONICAL_DIRECTIONS;

/// A straight edge between two board points at most 3 king steps apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Edge(u16);

impl Edge {
    /// Fills unused places in fixed-size arrays. It is never a real edge.
    pub(crate) const NONE: Edge = Edge(u16::MAX);

    /// The edge joining `a` and `b`, if they are 1 to 3 king steps apart.
    pub fn between(a: Point, b: Point) -> Option<Edge> {
        let direction = Direction::from_delta(b.x().checked_sub(a.x())?, b.y().checked_sub(a.y())?)?;
        Edge::from_move(a, direction)
    }

    /// The edge from `source` in `direction`, if it ends on the board.
    #[inline]
    pub fn from_move(source: Point, direction: Direction) -> Option<Edge> {
        let target = source.step(direction)?;
        Some(if direction.is_canonical() {
            Edge::from_origin(source, direction)
        } else {
            Edge::from_origin(target, direction.opposite())
        })
    }

    /// The edge with this slot, if the slot is an edge on the board.
    pub fn from_slot(slot: u16) -> Option<Edge> {
        if slot as usize >= NUM_SLOTS {
            return None;
        }
        let edge = Edge(slot);
        edge.origin().step(edge.canonical_direction()).map(|_| edge)
    }

    #[inline]
    fn from_origin(origin: Point, canonical: Direction) -> Edge {
        debug_assert!(canonical.is_canonical());
        let canonical_number = canonical.index() - NUM_CANONICAL_DIRECTIONS;
        Edge((origin.index() * NUM_CANONICAL_DIRECTIONS + canonical_number) as u16)
    }

    /// A unique number in `0..8664`.
    #[inline]
    pub const fn slot(self) -> u16 {
        self.0
    }

    /// The lower end, or the left one when the edge is level.
    #[inline]
    pub fn origin(self) -> Point {
        let point_index = self.0 as usize / NUM_CANONICAL_DIRECTIONS;
        Point::from_row_col(point_index / BOARD_SIDE, point_index % BOARD_SIDE)
    }

    /// The direction from [`Edge::origin`] to [`Edge::far`].
    #[inline]
    pub fn canonical_direction(self) -> Direction {
        let index = self.0 as usize % NUM_CANONICAL_DIRECTIONS + NUM_CANONICAL_DIRECTIONS;
        Direction::from_index(index).expect("a canonical direction")
    }

    /// The end that is not the origin.
    #[inline]
    pub fn far(self) -> Point {
        self.origin().step(self.canonical_direction()).expect("edges are checked to end on the board")
    }

    /// Origin first.
    #[inline]
    pub fn endpoints(self) -> (Point, Point) {
        (self.origin(), self.far())
    }

    #[inline]
    pub fn has_endpoint(self, point: Point) -> bool {
        let (origin, far) = self.endpoints();
        origin == point || far == point
    }

    /// The lattice points strictly between the ends: none, one or two.
    pub fn interior_points(self) -> impl Iterator<Item = Point> {
        let direction = self.canonical_direction();
        let steps = direction.steps();
        let (step_x, step_y) = (direction.dx() / steps, direction.dy() / steps);
        let origin = self.origin();
        (1..steps).map(move |k| origin.offset(step_x * k, step_y * k).expect("between two board points"))
    }
}

impl fmt::Debug for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Edge::NONE {
            return f.write_str("-");
        }
        write!(f, "{:?}-{:?}", self.origin(), self.far())
    }
}

/// The edges one player owns, sorted by slot, in a fixed-size array.
#[derive(Clone, Copy)]
pub struct EdgeSet {
    len: u8,
    slots: [Edge; MAX_EDGES_PER_PLAYER],
}

impl EdgeSet {
    pub const EMPTY: EdgeSet = EdgeSet { len: 0, slots: [Edge::NONE; MAX_EDGES_PER_PLAYER] };

    /// The edges, sorted by slot.
    #[inline]
    pub fn as_slice(&self) -> &[Edge] {
        &self.slots[..self.len as usize]
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Edge> + '_ {
        self.as_slice().iter().copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, edge: Edge) -> bool {
        self.as_slice().binary_search(&edge).is_ok()
    }

    /// Whether any edge of the set ends at `point`.
    pub fn touches_node(&self, point: Point) -> bool {
        self.iter().any(|edge| edge.has_endpoint(point))
    }

    /// Adds `edge`, unless it is already there or the set is full. Returns whether it was added.
    pub(crate) fn insert(&mut self, edge: Edge) -> bool {
        let len = self.len as usize;
        let Err(place) = self.as_slice().binary_search(&edge) else { return false };
        if len == MAX_EDGES_PER_PLAYER {
            return false;
        }
        self.slots.copy_within(place..len, place + 1);
        self.slots[place] = edge;
        self.len += 1;
        true
    }

    /// Removes `edge`. Returns whether it was there.
    pub(crate) fn remove(&mut self, edge: Edge) -> bool {
        let len = self.len as usize;
        let Ok(place) = self.as_slice().binary_search(&edge) else { return false };
        self.slots.copy_within(place + 1..len, place);
        self.slots[len - 1] = Edge::NONE;
        self.len -= 1;
        true
    }
}

impl Default for EdgeSet {
    fn default() -> Self {
        EdgeSet::EMPTY
    }
}

impl FromIterator<Edge> for EdgeSet {
    fn from_iter<T: IntoIterator<Item = Edge>>(edges: T) -> Self {
        let mut set = Self::EMPTY;
        for edge in edges {
            assert!(set.contains(edge) || set.insert(edge), "too many edges");
        }
        set
    }
}

/// Equality and hashing look at the edges only, not the unused places.
impl PartialEq for EdgeSet {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for EdgeSet {}

impl Hash for EdgeSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl fmt::Debug for EdgeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
