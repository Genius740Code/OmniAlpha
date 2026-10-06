//! Legal move generation.
//!
//! For each of the mover's nodes, the legal moves are a [`DirSet`]: the directions that end on the
//! board, minus every direction a rule strikes out.
//!
//! | Struck out | Found from |
//! |---|---|
//! | passing through one of the mover's nodes | the window of the mover's nodes, then a table |
//! | ending on one of the mover's edges | the window of points inside the mover's edges |
//! | repeating one of the mover's edges | the directions of the source's own edges |
//! | running along an opposing edge | the contact table |
//! | touching two or more opposing edges | the contact table, counted per direction |
//! | touching an opposing edge placed last turn | the contact table |
//!
//! Against the mover's own edges only an exact repeat needs checking: any other way of running
//! along one of them puts a node inside the new edge or the new node inside an old one, which
//! the first two rows already strike out.
//!
//! The work grows with the number of edges and of (own node, nearby opposing edge) pairs; no
//! candidate move is ever tested geometrically.

use crate::bitboard::Bitboard;
use crate::edge::{Edge, MAX_NODES_PER_PLAYER};
use crate::geometry::{BOARD_SIDE, DirSet, NUM_POINTS, Point, REACH};
use crate::moves::Move;
use crate::position::Position;
use crate::tables::tables;
use crate::units::Player;

/// The legal moves of a position, grouped by source node.
#[derive(Clone)]
pub struct LegalMoves {
    sources_len: u8,
    total: u16,
    sources: [Point; MAX_NODES_PER_PLAYER],
    targets: [DirSet; MAX_NODES_PER_PLAYER],
}

impl LegalMoves {
    const NONE: LegalMoves = LegalMoves {
        sources_len: 0,
        total: 0,
        sources: [Point::from_row_col(0, 0); MAX_NODES_PER_PLAYER],
        targets: [DirSet::EMPTY; MAX_NODES_PER_PLAYER],
    };

    #[inline]
    pub fn len(&self) -> usize {
        self.total as usize
    }

    /// Before the last action, no legal move means the game is a draw.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Each of the mover's nodes, in point order, with its legal directions (possibly none).
    pub fn by_source(&self) -> impl Iterator<Item = (Point, DirSet)> + '_ {
        let count = self.sources_len as usize;
        self.sources[..count].iter().copied().zip(self.targets[..count].iter().copied())
    }

    /// Every legal move, by source point, then by direction.
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.by_source().flat_map(|(source, targets)| targets.iter().map(move |direction| Move { source, direction }))
    }

    pub fn contains(&self, mv: Move) -> bool {
        self.sources[..self.sources_len as usize]
            .binary_search(&mv.source)
            .is_ok_and(|place| self.targets[place].contains(mv.direction))
    }

    /// The `n`-th move of [`LegalMoves::iter`]. With `n` uniform below `len()`, a uniform draw.
    pub fn nth(&self, mut n: usize) -> Option<Move> {
        for (source, targets) in self.by_source() {
            let count = targets.len() as usize;
            if n < count {
                return targets.nth(n as u32).map(|direction| Move { source, direction });
            }
            n -= count;
        }
        None
    }
}

impl std::fmt::Debug for LegalMoves {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// What the edges strike out for one source node, as sets of directions.
#[derive(Clone, Copy, Default)]
struct StruckOut {
    repeating_own_edge: u64,
    touching_one: u64,
    touching_two: u64,
    /// Touching an opposing edge placed last turn.
    forbidden: u64,
}

pub(crate) fn generate(position: &Position) -> LegalMoves {
    generate_for(position, position.to_move(), |edge| position.is_shielded(edge))
}

/// The moves `mover` could make now, if the opposing edges for which `shielded` is true could not
/// be touched.
pub(crate) fn generate_for(position: &Position, mover: Player, shielded: impl Fn(Edge) -> bool) -> LegalMoves {
    let mut legal = LegalMoves::NONE;
    if position.is_finished() {
        return legal;
    }
    let tables = tables();
    let own_nodes = position.nodes(mover);

    // Number the mover's nodes, so that what is struck out for each fits a small array.
    let mut node_number = [0u8; NUM_POINTS];
    let mut struck = [StruckOut::default(); MAX_NODES_PER_PLAYER];
    for (number, source) in own_nodes.iter().enumerate() {
        node_number[source.index()] = number as u8;
        legal.sources[number] = source;
        legal.sources_len += 1;
    }

    let mut inside_own_edges = Bitboard::EMPTY;
    for edge in position.edges(mover).iter() {
        let direction = edge.canonical_direction();
        struck[node_number[edge.origin().index()] as usize].repeating_own_edge |= 1 << direction.window_bit();
        struck[node_number[edge.far().index()] as usize].repeating_own_edge |= 1 << direction.opposite().window_bit();
        edge.interior_points().for_each(|point| inside_own_edges.insert(point));
    }

    for edge in position.edges(mover.opponent()).iter() {
        let shielded = shielded(edge);
        let canonical = edge.canonical_direction();
        let (origin, far) = edge.endpoints();
        // Only sources within reach of the edge's bounding box can touch it. The origin is the
        // lower end, so rows run from the origin's to the far end's; columns may run either way.
        let reach = REACH as usize;
        let first_row = origin.row().saturating_sub(reach);
        let last_row = (far.row() + reach).min(BOARD_SIDE - 1);
        let first_col = origin.col().min(far.col()).saturating_sub(reach);
        let last_col = (origin.col().max(far.col()) + reach).min(BOARD_SIDE - 1);
        let columns = ((1u32 << (last_col - first_col + 1)) - 1) << first_col;

        for row in first_row..=last_row {
            let mut nearby_sources = own_nodes.row_bits(row) & columns;
            while nearby_sources != 0 {
                let col = nearby_sources.trailing_zeros() as usize;
                nearby_sources &= nearby_sources - 1;
                let contact =
                    tables.contact(origin.col() as i32 - col as i32, origin.row() as i32 - row as i32, canonical);
                let struck = &mut struck[node_number[row * BOARD_SIDE + col] as usize];
                struck.touching_two |= struck.touching_one & contact.touch;
                struck.touching_one |= contact.touch;
                if shielded {
                    struck.forbidden |= contact.touch;
                }
            }
        }
    }

    let count = legal.sources_len as usize;
    for ((&source, struck), targets) in legal.sources[..count].iter().zip(&struck).zip(&mut legal.targets) {
        let all_struck = struck.repeating_own_edge
            | struck.touching_two
            | struck.forbidden
            | inside_own_edges.window(source)
            | tables.blocked_by(own_nodes.window(source));
        *targets = DirSet::from_bits(tables.ends_on_board(source) & !all_struck);
        legal.total += targets.len() as u16;
    }
    legal
}
