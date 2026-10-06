//! The exact area a player encloses, computed with integers only.
//!
//! 1. Split the player's edges wherever they cross or touch, which gives a planar graph. Scaled by
//!    [`CROSSING_LCM`], every vertex has integer coordinates.
//! 2. Walk around the outside of each connected part of the graph. That walk goes around exactly
//!    the faces the part encloses, so the shoelace formula over it gives their area. An edge that
//!    encloses nothing is walked once each way and cancels out.
//! 3. A part that lies inside another part's walk adds nothing. Separate parts never touch, so one
//!    of its vertices decides whether it is inside.
//!
//! Edge directions come in only 32 angles, and no two edges leave a vertex at the same angle. So
//! the edges around a vertex are a 32-bit mask of angles, the next edge clockwise is a bit scan,
//! and nothing is ever sorted by angle.
//!
//! The edges must be one player's edges from a legal position: at most 61 of them, none lying
//! along another. Crossings, shared ends, several edges through one point and T-junctions are
//! all handled.

use std::cell::RefCell;
use std::ops::{Range, RangeInclusive};

use crate::edge::{Edge, MAX_EDGES_PER_PLAYER};
use crate::geometry::{BOARD_SIDE, COORD_LIMIT, NUM_POINTS, Point};
use crate::tables::{ANGLE_CLASSES, tables};
use crate::units::{Area, CROSSING_LCM};

/// Vertices are stored at `SCALE` times their coordinates. That stays below 2^22, so coordinates
/// fit an `i32` and their products an `i64`.
pub(crate) const SCALE: i32 = CROSSING_LCM as i32;

const NO_VERTEX: u32 = u32::MAX;
const NO_SPLIT: u32 = u32::MAX;

/// Angles from the positive x axis up to straight up, 0° to 90°.
const RIGHT_AND_UP: u32 = (1 << 9) - 1;
/// Angles pointing left, above 90° and below 270°.
const LEFT: u32 = (1 << 25) - (1 << 9);

thread_local! {
    static GRAPH: RefCell<PlanarGraph> = RefCell::new(PlanarGraph::new());
}

/// The exact area enclosed by one player's edges.
///
/// # Panics
/// If there are more than [`MAX_EDGES_PER_PLAYER`] edges.
pub fn enclosed_area(edges: &[Edge]) -> Area {
    assert!(edges.len() <= MAX_EDGES_PER_PLAYER, "a player never has more than {MAX_EDGES_PER_PLAYER} edges");
    // A loop takes three edges, even with crossings: two straight edges meet at most once.
    if edges.len() < 3 {
        return Area::ZERO;
    }
    with_graph(edges, |graph| Area::from_numerator(graph.outer_walks().map(|walk| walk.area).sum()))
}

/// Builds the planar graph of `edges` in this thread's reusable buffers and reads it.
pub(crate) fn with_graph<T>(edges: &[Edge], read: impl FnOnce(&PlanarGraph) -> T) -> T {
    GRAPH.with_borrow_mut(|graph| {
        graph.build(edges);
        read(graph)
    })
}

/// An edge in lattice units: from `(x, y)` along its canonical direction `(dx, dy)`.
#[derive(Clone, Copy)]
struct Segment {
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    angle: u32,
    start: u32,
    end: u32,
}

impl Segment {
    /// The board rows its bounding box covers. Canonical directions never point down.
    fn rows(self) -> RangeInclusive<usize> {
        let first = (self.y + i32::from(COORD_LIMIT)) as usize;
        first..=first + self.dy as usize
    }

    fn cols(self) -> RangeInclusive<usize> {
        let (start, end) = (self.x + i32::from(COORD_LIMIT), self.x + self.dx + i32::from(COORD_LIMIT));
        start.min(end) as usize..=start.max(end) as usize
    }
}

/// A vertex inside a segment, `along / SCALE` of the way from its start.
#[derive(Clone, Copy)]
struct Split {
    segment: u32,
    along: i32,
    vertex: u32,
    next_on_segment: u32,
}

/// The part of an edge between two neighbouring vertices, with its angle seen from `from`.
#[derive(Clone, Copy)]
struct Piece {
    from: u32,
    to: u32,
    angle: u32,
}

/// The outside boundary of one connected part that encloses something.
pub(crate) struct Walk {
    /// Where its vertices are in `PlanarGraph::walk_vertices`, in walking order.
    vertices: Range<usize>,
    /// The area it encloses, as a numerator over `AREA_DENOMINATOR`.
    pub area: i64,
}

/// One player's edges split into a planar graph, and the walk around each connected part. The
/// buffers are reused from call to call and only grow.
pub(crate) struct PlanarGraph {
    /// The vertex at each lattice point, valid where `stamps` holds the current `stamp`. A new
    /// stamp forgets every lattice vertex at once.
    lattice_vertices: [u32; NUM_POINTS],
    stamps: [u32; NUM_POINTS],
    stamp: u32,
    /// Scaled coordinates of every vertex.
    positions: Vec<(i32, i32)>,
    segments: Vec<Segment>,
    splits: Vec<Split>,
    first_split: [u32; MAX_EDGES_PER_PLAYER],
    pieces: Vec<Piece>,
    /// Per vertex, the angles of the pieces that leave it.
    angles_out: Vec<u32>,
    /// `neighbours[first_neighbour[v] + k]` is where the `k`-th piece leaving `v`, by angle,
    /// leads.
    first_neighbour: Vec<u32>,
    neighbours: Vec<u32>,
    visited: Vec<bool>,
    stack: Vec<u32>,
    walks: Vec<Walk>,
    walk_vertices: Vec<u32>,
}

impl PlanarGraph {
    fn new() -> PlanarGraph {
        PlanarGraph {
            lattice_vertices: [NO_VERTEX; NUM_POINTS],
            stamps: [0; NUM_POINTS],
            stamp: 0,
            positions: Vec::new(),
            segments: Vec::new(),
            splits: Vec::new(),
            first_split: [NO_SPLIT; MAX_EDGES_PER_PLAYER],
            pieces: Vec::new(),
            angles_out: Vec::new(),
            first_neighbour: Vec::new(),
            neighbours: Vec::new(),
            visited: Vec::new(),
            stack: Vec::new(),
            walks: Vec::new(),
            walk_vertices: Vec::new(),
        }
    }

    fn build(&mut self, edges: &[Edge]) {
        self.load_segments(edges);
        self.split_segments();
        self.join_pieces();
        self.walk_parts();
    }

    /// The walks of the parts that are not inside another part.
    pub(crate) fn outer_walks(&self) -> impl Iterator<Item = &Walk> {
        self.walks.iter().filter(|walk| !self.is_nested(walk))
    }

    /// A walk as a closed chain of straight sides, `(from, to)` in scaled coordinates, clockwise.
    pub(crate) fn sides(&self, walk: &Walk) -> impl Iterator<Item = ((i32, i32), (i32, i32))> + '_ {
        let corners = self.walk_vertices[walk.vertices.clone()].iter().map(|&vertex| self.positions[vertex as usize]);
        corners.clone().zip(corners.cycle().skip(1))
    }

    fn lattice_vertex(&mut self, point: Point) -> u32 {
        let index = point.index();
        if self.stamps[index] != self.stamp {
            self.stamps[index] = self.stamp;
            self.lattice_vertices[index] = self.positions.len() as u32;
            self.positions.push((i32::from(point.x()) * SCALE, i32::from(point.y()) * SCALE));
        }
        self.lattice_vertices[index]
    }

    fn load_segments(&mut self, edges: &[Edge]) {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.stamps = [0; NUM_POINTS];
            self.stamp = 1;
        }
        self.positions.clear();
        self.segments.clear();
        let tables = tables();
        for &edge in edges {
            let direction = edge.canonical_direction();
            let (origin, far) = edge.endpoints();
            let segment = Segment {
                x: i32::from(origin.x()),
                y: i32::from(origin.y()),
                dx: i32::from(direction.dx()),
                dy: i32::from(direction.dy()),
                angle: u32::from(tables.angle_class(direction)),
                start: self.lattice_vertex(origin),
                end: self.lattice_vertex(far),
            };
            self.segments.push(segment);
        }
    }

    /// Finds every vertex inside a segment: wherever two segments meet other than end to end.
    fn split_segments(&mut self) {
        self.splits.clear();
        self.first_split.fill(NO_SPLIT);
        // Two segments can only meet if their bounding boxes share a row and a column, so a few
        // unions of "which segments reach this row or column" replace testing every pair.
        let mut in_row = [0u64; BOARD_SIDE];
        let mut in_col = [0u64; BOARD_SIDE];
        for (index, segment) in self.segments.iter().enumerate() {
            segment.rows().for_each(|row| in_row[row] |= 1 << index);
            segment.cols().for_each(|col| in_col[col] |= 1 << index);
        }
        for first in 0..self.segments.len() {
            let segment = self.segments[first];
            let sharing_a_row = segment.rows().fold(0, |found, row| found | in_row[row]);
            let sharing_a_col = segment.cols().fold(0, |found, col| found | in_col[col]);
            let mut later = sharing_a_row & sharing_a_col & !((2u64 << first) - 1);
            while later != 0 {
                let second = later.trailing_zeros() as usize;
                later &= later - 1;
                self.split_pair(first, second);
            }
        }
    }

    /// Where two segments meet inside one or both of them, records a vertex there.
    #[inline]
    fn split_pair(&mut self, first_index: usize, second_index: usize) {
        let (first, second) = (self.segments[first_index], self.segments[second_index]);
        // first.start + s · first.d = second.start + u · second.d, by Cramer's rule:
        // s = (w × second.d) / det and u = (w × first.d) / det, with w = second.start − first.start.
        let determinant = first.dx * second.dy - first.dy * second.dx;
        let (wx, wy) = (second.x - first.x, second.y - first.y);
        let sign = determinant.signum();
        let determinant = determinant * sign;
        let along_first = (wx * second.dy - wy * second.dx) * sign;
        let along_second = (wx * first.dy - wy * first.dx) * sign;
        // Parallel segments (determinant 0) stop here too: not lying along each other, they can
        // share at most an end.
        let meet =
            (0 <= along_first) & (along_first <= determinant) & (0 <= along_second) & (along_second <= determinant);
        let inside_first = (0 < along_first) & (along_first < determinant);
        let inside_second = (0 < along_second) & (along_second < determinant);
        if !(meet & (inside_first | inside_second)) {
            return;
        }
        // The determinant divides SCALE: that is what CROSSING_LCM is.
        let along_first = along_first * (SCALE / determinant);
        let along_second = along_second * (SCALE / determinant);

        // Several segments through one point must share one vertex. They are pairwise not
        // parallel, so the lowest-numbered of them meets all the others before any other pair
        // does, and from then on each of them has the vertex among its splits.
        let known_on_first = if inside_first { self.split_at(first_index, along_first) } else { None };
        let known_on_second = if inside_second { self.split_at(second_index, along_second) } else { None };
        let vertex = known_on_first.or(known_on_second).unwrap_or_else(|| {
            self.vertex_at(first.x * SCALE + along_first * first.dx, first.y * SCALE + along_first * first.dy)
        });
        if inside_first && known_on_first.is_none() {
            self.add_split(first_index, along_first, vertex);
        }
        if inside_second && known_on_second.is_none() {
            self.add_split(second_index, along_second, vertex);
        }
    }

    fn add_split(&mut self, segment: usize, along: i32, vertex: u32) {
        let next_on_segment = std::mem::replace(&mut self.first_split[segment], self.splits.len() as u32);
        self.splits.push(Split { segment: segment as u32, along, vertex, next_on_segment });
    }

    fn split_at(&self, segment: usize, along: i32) -> Option<u32> {
        let mut index = self.first_split[segment];
        while index != NO_SPLIT {
            let split = self.splits[index as usize];
            if split.along == along {
                return Some(split.vertex);
            }
            index = split.next_on_segment;
        }
        None
    }

    /// The vertex at a scaled position no split has yet. At a lattice point it may exist all the
    /// same, as the end of a third segment.
    fn vertex_at(&mut self, x: i32, y: i32) -> u32 {
        if x % SCALE == 0 && y % SCALE == 0 {
            let point = Point::new((x / SCALE) as i8, (y / SCALE) as i8).expect("segments meet on the board");
            return self.lattice_vertex(point);
        }
        self.positions.push((x, y));
        self.positions.len() as u32 - 1
    }

    /// Cuts each segment at its splits into pieces, and indexes the pieces by vertex and angle.
    fn join_pieces(&mut self) {
        // The splits are all found, so their per-segment links are no longer needed.
        self.splits.sort_unstable_by_key(|split| (split.segment, split.along));
        self.pieces.clear();
        let mut splits = self.splits.iter().peekable();
        for (index, segment) in self.segments.iter().enumerate() {
            let mut from = segment.start;
            while let Some(split) = splits.next_if(|split| split.segment == index as u32) {
                self.pieces.push(Piece { from, to: split.vertex, angle: segment.angle });
                from = split.vertex;
            }
            self.pieces.push(Piece { from, to: segment.end, angle: segment.angle });
        }

        self.angles_out.clear();
        self.angles_out.resize(self.positions.len(), 0);
        for piece in &self.pieces {
            debug_assert_eq!(
                self.angles_out[piece.from as usize] & 1 << piece.angle,
                0,
                "two edges leave at one angle"
            );
            self.angles_out[piece.from as usize] |= 1 << piece.angle;
            self.angles_out[piece.to as usize] |= 1 << opposite(piece.angle);
        }
        self.first_neighbour.clear();
        self.first_neighbour.push(0);
        let mut total = 0;
        for angles in &self.angles_out {
            total += angles.count_ones();
            self.first_neighbour.push(total);
        }
        self.neighbours.clear();
        self.neighbours.resize(total as usize, NO_VERTEX);
        for index in 0..self.pieces.len() {
            let piece = self.pieces[index];
            let forward = self.neighbour_index(piece.from, piece.angle);
            self.neighbours[forward] = piece.to;
            let backward = self.neighbour_index(piece.to, opposite(piece.angle));
            self.neighbours[backward] = piece.from;
        }
    }

    /// Where in `neighbours` the piece leaving `vertex` at `angle` is.
    #[inline]
    fn neighbour_index(&self, vertex: u32, angle: u32) -> usize {
        let smaller_angles = self.angles_out[vertex as usize] & ((1 << angle) - 1);
        (self.first_neighbour[vertex as usize] + smaller_angles.count_ones()) as usize
    }

    /// Walks around the outside of each connected part, and keeps the walks that enclose area.
    fn walk_parts(&mut self) {
        self.walks.clear();
        self.walk_vertices.clear();
        self.visited.clear();
        self.visited.resize(self.positions.len(), false);

        for root in 0..self.positions.len() as u32 {
            if self.visited[root as usize] {
                continue;
            }
            // The part's leftmost vertex (the lowest, if several) is on its outside, and every
            // piece leaving it points right or straight up. The one turned furthest
            // counter-clockwise has the outside on its left.
            let start = self.mark_part(root);
            let angles = self.angles_out[start as usize];
            debug_assert_eq!(angles & LEFT, 0, "a piece points left of the leftmost vertex");
            let up = angles & RIGHT_AND_UP;
            let first_angle = highest_angle(if up != 0 { up } else { angles });

            let first_vertex = self.walk_vertices.len();
            let mut doubled_area: i64 = 0;
            let (mut from, mut angle) = (start, first_angle);
            let mut steps = 0;
            loop {
                let to = self.neighbours[self.neighbour_index(from, angle)];
                let (from_x, from_y) = self.positions[from as usize];
                let (to_x, to_y) = self.positions[to as usize];
                doubled_area += i64::from(from_x) * i64::from(to_y) - i64::from(to_x) * i64::from(from_y);
                self.walk_vertices.push(from);

                // Keep the outside on the left: leave `to` by the first piece clockwise from the
                // one we came in on. At a dead end that is the same piece, going back.
                angle = next_clockwise(self.angles_out[to as usize], opposite(angle));
                from = to;

                steps += 1;
                assert!(steps <= 2 * self.pieces.len(), "the walk around a part did not close");
                if from == start && angle == first_angle {
                    break;
                }
            }

            // The walk goes clockwise, so the shoelace sum is at most zero. In scaled coordinates
            // it is −2 · area · SCALE², which is −SCALE times the area's numerator.
            debug_assert!(doubled_area <= 0 && doubled_area % i64::from(SCALE) == 0);
            if doubled_area == 0 {
                self.walk_vertices.truncate(first_vertex);
            } else {
                let vertices = first_vertex..self.walk_vertices.len();
                self.walks.push(Walk { vertices, area: -doubled_area / i64::from(SCALE) });
            }
        }
    }

    /// Marks the part containing `root` as visited, and returns its leftmost, then lowest, vertex.
    fn mark_part(&mut self, root: u32) -> u32 {
        let mut leftmost = root;
        self.visited[root as usize] = true;
        self.stack.clear();
        self.stack.push(root);
        while let Some(vertex) = self.stack.pop() {
            if self.positions[vertex as usize] < self.positions[leftmost as usize] {
                leftmost = vertex;
            }
            let first = self.first_neighbour[vertex as usize] as usize;
            let last = self.first_neighbour[vertex as usize + 1] as usize;
            for &neighbour in &self.neighbours[first..last] {
                if !self.visited[neighbour as usize] {
                    self.visited[neighbour as usize] = true;
                    self.stack.push(neighbour);
                }
            }
        }
        leftmost
    }

    fn is_nested(&self, inner: &Walk) -> bool {
        let (witness, _) = self.sides(inner).next().expect("a walk has sides");
        // Only a larger walk can contain it, which also keeps a walk from containing itself.
        self.walks.iter().any(|outer| outer.area > inner.area && self.walk_contains(outer, witness))
    }

    /// Even-odd test of a point known not to lie on the walk. A piece walked twice, on an edge
    /// that encloses nothing, counts twice and cancels.
    fn walk_contains(&self, walk: &Walk, (x, y): (i32, i32)) -> bool {
        let mut inside = false;
        for ((from_x, from_y), (to_x, to_y)) in self.sides(walk) {
            if (from_y > y) == (to_y > y) {
                continue;
            }
            // The piece crosses the level line through the point. It counts if it crosses to the
            // right: x < from_x + (y − from_y)(to_x − from_x) / (to_y − from_y).
            let left = i64::from(x - from_x) * i64::from(to_y - from_y);
            let right = i64::from(y - from_y) * i64::from(to_x - from_x);
            if (to_y > from_y && left < right) || (to_y < from_y && left > right) {
                inside = !inside;
            }
        }
        inside
    }
}

#[inline]
fn opposite(angle: u32) -> u32 {
    (angle + ANGLE_CLASSES / 2) % ANGLE_CLASSES
}

#[inline]
fn highest_angle(angles: u32) -> u32 {
    debug_assert_ne!(angles, 0);
    31 - angles.leading_zeros()
}

/// The first of `angles` met turning clockwise from `angle`: the highest one below it, or else the
/// highest of all. `angle` itself comes last.
#[inline]
fn next_clockwise(angles: u32, angle: u32) -> u32 {
    let below = angles & ((1 << angle) - 1);
    highest_angle(if below != 0 { below } else { angles })
}
