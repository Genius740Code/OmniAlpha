//! The state of a game, and the rules that change it.
//!
//! A [`Position`] is a small value with no heap data, so it is cheap to clone. It stores only
//! what decides the rest of the game, in one canonical form, so positions reached by different
//! move orders compare and hash equal:
//!
//! * each player's edges, sorted; the nodes are exactly the ends of the edges;
//! * the number of actions played, which decides whose turn it is and how many actions it has left;
//! * the *shielded* edges: the opponent's edges from their last turn, which the player to move may
//!   not touch;
//! * the *fresh* edges: those the player to move has placed this turn, shielded in the next one;
//! * both scores.
//!
//! It also keeps each player's nodes and enclosed area, which follow from the edges.

use std::fmt;
use std::hash::{Hash, Hasher};

use crate::area::enclosed_area;
use crate::bitboard::Bitboard;
use crate::edge::{Edge, EdgeSet};
use crate::geometry::{BOARD_SIDE, Point, segments_overlap, segments_touch, strictly_inside};
use crate::movegen::{self, LegalMoves};
use crate::moves::Move;
use crate::tables::tables;
use crate::units::{Area, Player, Score};

pub const TOTAL_ACTIONS: u8 = 120;
/// After actions 1, 3, 5, …, 119, and after action 120.
pub const SCORING_EVENTS: u8 = TOTAL_ACTIONS / 2 + 1;
/// In unit squares: no region is larger than the board.
const BOARD_AREA: i64 = ((BOARD_SIDE - 1) * (BOARD_SIDE - 1)) as i64;

/// What a move does at its target.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MoveKind {
    /// The target was empty and becomes the mover's node.
    Extend,
    /// The target was the mover's node. This is how loops close.
    Connect,
    /// The target was an opposing node with one edge: the edge is cut and the node changes hands.
    Capture,
}

/// Why a move is illegal, in the order [`Position::check_move`] checks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IllegalMove {
    GameOver,
    SourceNotOwned,
    OffBoard,
    PathThroughOwnNode,
    TargetOnOwnEdge,
    OverlapsEdge,
    BreaksShieldedEdge,
    /// The number of opposing edges the move would touch.
    BreaksSeveralEdges(u8),
}

impl fmt::Display for IllegalMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IllegalMove::GameOver => f.write_str("the game is over"),
            IllegalMove::SourceNotOwned => f.write_str("the source is not one of the mover's nodes"),
            IllegalMove::OffBoard => f.write_str("the target is off the board"),
            IllegalMove::PathThroughOwnNode => f.write_str("the edge would pass through one of the mover's own nodes"),
            IllegalMove::TargetOnOwnEdge => f.write_str("the new node would land on one of the mover's own edges"),
            IllegalMove::OverlapsEdge => f.write_str("the edge would lie on top of one of the mover's own edges"),
            IllegalMove::BreaksShieldedEdge => {
                f.write_str("the edge would break an edge placed last turn, which is still invincible")
            }
            IllegalMove::BreaksSeveralEdges(count) => {
                write!(f, "the edge would break {count} opposing edges; the limit is one per move")
            }
        }
    }
}

impl std::error::Error for IllegalMove {}

/// What playing a move did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoveOutcome {
    pub kind: MoveKind,
    pub placed: Edge,
    /// The opposing edge the move cut.
    pub broken: Option<Edge>,
    /// The areas added to `[Blue, Red]`, when the move was followed by a scoring event.
    pub scored: Option<[Area; 2]>,
    pub turn_ended: bool,
    pub game_over: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Outcome {
    /// All actions were played and this player has the higher score.
    Win(Player),
    Draw(DrawReason),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DrawReason {
    /// All actions were played and the scores are equal.
    EqualScores,
    /// The player to move had no legal move, which ends the game whatever the scores.
    NoLegalMove,
}

impl Outcome {
    pub fn winner(self) -> Option<Player> {
        match self {
            Outcome::Win(player) => Some(player),
            Outcome::Draw(_) => None,
        }
    }
}

/// Why a set of edges, actions and scores is not a position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InvalidPosition {
    TooManyActions,
    /// More edges than the starting edge plus one per action the player has made.
    TooManyEdges(Player),
    /// More nodes than the two starting nodes plus one per action the player has made.
    TooManyNodes(Player),
    /// Negative, or more than the whole board at every scoring event so far.
    ScoreOutOfRange(Player),
    DuplicateEdge(Edge),
    /// A Blue edge and a Red edge share a point, which no move can bring about.
    OpposingEdgesTouch(Edge, Edge),
    /// Two edges of one player lie along each other.
    OwnEdgesOverlap(Edge, Edge),
    /// A node lies inside an edge of its own player.
    NodeInsideOwnEdge(Point, Edge),
    /// Wrong number or owner of shielded edges, or one listed twice.
    BadShieldedEdge,
    /// Wrong number or owner of fresh edges.
    BadFreshEdge,
}

impl fmt::Display for InvalidPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid position: {self:?}")
    }
}

impl std::error::Error for InvalidPosition {}

#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    edges: [EdgeSet; 2],
    scores: [Score; 2],
    /// Sorted, with [`Edge::NONE`] in unused places.
    shielded: [Edge; 2],
    /// Sorted, with [`Edge::NONE`] in unused places.
    fresh: [Edge; 2],
    actions_played: u8,
    nodes: [Bitboard; 2],
    areas: [Area; 2],
}

/// How many shielded and fresh edges a position may have.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RecentEdges {
    /// As many as a game has at that point.
    Exactly,
    /// Up to that many, as a position editor allows.
    AtMost,
}

impl Default for Position {
    fn default() -> Self {
        Position::new()
    }
}

impl Position {
    /// Blue holds `(-9,0)–(-6,0)`, Red holds `(6,0)–(9,0)`, and Blue moves.
    pub fn new() -> Position {
        let point = |x, y| Point::new(x, y).expect("on the board");
        let edge = |a, b| Edge::between(a, b).expect("3 apart");
        let blue = [edge(point(-9, 0), point(-6, 0))];
        let red = [edge(point(6, 0), point(9, 0))];
        Position::from_parts(&blue, &red, 0, [Score::ZERO; 2], &[], &[]).expect("the starting position is valid")
    }

    /// A position from its parts, with the shielded and fresh edges a game would have at this
    /// point: one shielded edge after actions 1 and 2, then two until action 119, and none at
    /// the start or the end; one fresh edge after actions 2, 4, …, 118.
    ///
    /// The edges, counts and scores are checked. Whether a game can reach the position is not.
    pub fn from_parts(
        blue_edges: &[Edge],
        red_edges: &[Edge],
        actions_played: u8,
        scores: [Score; 2],
        shielded: &[Edge],
        fresh: &[Edge],
    ) -> Result<Position, InvalidPosition> {
        Position::build(blue_edges, red_edges, actions_played, scores, shielded, fresh, RecentEdges::Exactly)
    }

    /// A position as an editor sets one up. `recent` lists edges of either player placed in the
    /// latest turns: the opponent's become shielded and the mover's fresh. Fewer of them than a
    /// game would have are allowed, so [`Position::validate`] may fail until two more turns end.
    pub fn setup(
        blue_edges: &[Edge],
        red_edges: &[Edge],
        actions_played: u8,
        scores: [Score; 2],
        recent: &[Edge],
    ) -> Result<Position, InvalidPosition> {
        let mover = mover_after(actions_played);
        let owner = |edge: &Edge| {
            if blue_edges.contains(edge) {
                Some(Player::Blue)
            } else if red_edges.contains(edge) {
                Some(Player::Red)
            } else {
                None
            }
        };
        if recent.iter().any(|edge| owner(edge).is_none()) {
            return Err(InvalidPosition::BadShieldedEdge);
        }
        let owned_by = |player| recent.iter().copied().filter(|edge| owner(edge) == Some(player)).collect::<Vec<_>>();
        let (shielded, fresh) = (owned_by(mover.opponent()), owned_by(mover));
        Position::build(blue_edges, red_edges, actions_played, scores, &shielded, &fresh, RecentEdges::AtMost)
    }

    fn build(
        blue_edges: &[Edge],
        red_edges: &[Edge],
        actions_played: u8,
        scores: [Score; 2],
        shielded: &[Edge],
        fresh: &[Edge],
        recent_edges: RecentEdges,
    ) -> Result<Position, InvalidPosition> {
        if actions_played > TOTAL_ACTIONS {
            return Err(InvalidPosition::TooManyActions);
        }
        // Every action adds one edge and at most one node. The fixed-size arrays rely on this, so
        // positions with more are refused.
        let mut edges = [EdgeSet::EMPTY; 2];
        for (player, list) in Player::BOTH.into_iter().zip([blue_edges, red_edges]) {
            if list.len() > 1 + actions_made_by(player, actions_played) {
                return Err(InvalidPosition::TooManyEdges(player));
            }
            for &edge in list {
                if !edges[player.index()].insert(edge) {
                    return Err(InvalidPosition::DuplicateEdge(edge));
                }
            }
        }

        let mut position = Position {
            edges,
            scores,
            shielded: [Edge::NONE; 2],
            fresh: [Edge::NONE; 2],
            actions_played,
            nodes: [Bitboard::EMPTY; 2],
            areas: [Area::ZERO; 2],
        };
        position.check_edge_geometry()?;

        let mover = position.to_move();
        let (shielded_in_game, fresh_in_game) = position.recent_edges_in_game();
        let count_allowed = |count: usize, in_game: usize| match recent_edges {
            RecentEdges::Exactly => count == in_game,
            RecentEdges::AtMost => count <= in_game,
        };
        let shielded_ok = count_allowed(shielded.len(), shielded_in_game)
            && (shielded.len() != 2 || shielded[0] != shielded[1])
            && shielded.iter().all(|&edge| position.edges(mover.opponent()).contains(edge));
        if !shielded_ok {
            return Err(InvalidPosition::BadShieldedEdge);
        }
        let fresh_ok =
            count_allowed(fresh.len(), fresh_in_game) && fresh.iter().all(|&edge| position.edges(mover).contains(edge));
        if !fresh_ok {
            return Err(InvalidPosition::BadFreshEdge);
        }
        position.shielded = sorted_pair(shielded);
        position.fresh = sorted_pair(fresh);

        let events_so_far = i64::from(SCORING_EVENTS - position.scoring_events_left());
        let highest_score = Score::from_halves(2 * BOARD_AREA * events_so_far);
        for player in Player::BOTH {
            let nodes: Bitboard = position.edges(player).iter().flat_map(|edge| [edge.origin(), edge.far()]).collect();
            if nodes.len() > 2 + actions_made_by(player, actions_played) {
                return Err(InvalidPosition::TooManyNodes(player));
            }
            if position.score(player) < Score::ZERO || position.score(player) > highest_score {
                return Err(InvalidPosition::ScoreOutOfRange(player));
            }
            position.nodes[player.index()] = nodes;
            position.refresh_area(player);
        }
        Ok(position)
    }

    /// How many shielded and fresh edges a game has at this point.
    fn recent_edges_in_game(&self) -> (usize, usize) {
        if self.is_finished() {
            return (0, 0);
        }
        let shielded = usize::from(self.turn_index()).min(2);
        let mid_turn = self.actions_played >= 2 && self.actions_played % 2 == 0;
        (shielded, usize::from(mid_turn))
    }

    /// Checks what every move keeps true: Blue and Red edges never share a point, a player's
    /// edges never lie along each other, and no node lies inside an edge of its own player.
    fn check_edge_geometry(&self) -> Result<(), InvalidPosition> {
        for blue in self.edges(Player::Blue).iter() {
            for red in self.edges(Player::Red).iter() {
                if touches(blue, red.origin(), red.far()) {
                    return Err(InvalidPosition::OpposingEdgesTouch(blue, red));
                }
            }
        }
        for player in Player::BOTH {
            let own = self.edges(player).as_slice();
            for (index, &first) in own.iter().enumerate() {
                for &second in &own[index + 1..] {
                    let (first_start, first_end) = (first.origin().to_vector(), first.far().to_vector());
                    let (second_start, second_end) = (second.origin().to_vector(), second.far().to_vector());
                    if segments_overlap(first_start, first_end, second_start, second_end) {
                        return Err(InvalidPosition::OwnEdgesOverlap(first, second));
                    }
                    for (node, edge) in [
                        (first.origin(), second),
                        (first.far(), second),
                        (second.origin(), first),
                        (second.far(), first),
                    ] {
                        if strictly_inside(edge.origin().to_vector(), edge.far().to_vector(), node.to_vector()) {
                            return Err(InvalidPosition::NodeInsideOwnEdge(node, edge));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn actions_played(&self) -> u8 {
        self.actions_played
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.actions_played >= TOTAL_ACTIONS
    }

    /// The turn the next action belongs to: turn 0 is action 1, and turn `t` is actions `2t` and
    /// `2t + 1`.
    #[inline]
    pub fn turn_index(&self) -> u8 {
        self.actions_played.div_ceil(2)
    }

    /// Blue plays the even turns.
    #[inline]
    pub fn to_move(&self) -> Player {
        mover_after(self.actions_played)
    }

    /// The mover's actions left in this turn, the next one included: 2 at the start of a
    /// two-action turn, otherwise 1, and 0 once the game is finished.
    pub fn actions_left_in_turn(&self) -> u8 {
        if self.is_finished() {
            0
        } else if self.actions_played % 2 == 1 && self.actions_played + 1 < TOTAL_ACTIONS {
            2
        } else {
            1
        }
    }

    pub fn scoring_events_left(&self) -> u8 {
        if self.is_finished() {
            return 0;
        }
        // One after each odd action still to come, up to 119, and one after the last action.
        (TOTAL_ACTIONS - self.actions_played) / 2 + 1
    }

    #[inline]
    pub fn edges(&self, player: Player) -> &EdgeSet {
        &self.edges[player.index()]
    }

    #[inline]
    pub fn nodes(&self, player: Player) -> &Bitboard {
        &self.nodes[player.index()]
    }

    pub fn node_owner(&self, point: Point) -> Option<Player> {
        Player::BOTH.into_iter().find(|&player| self.nodes(player).contains(point))
    }

    #[inline]
    pub fn score(&self, player: Player) -> Score {
        self.scores[player.index()]
    }

    /// What `player` encloses now, which the next scoring event adds to their score.
    #[inline]
    pub fn area(&self, player: Player) -> Area {
        self.areas[player.index()]
    }

    /// The opponent's edges from their last turn, which the player to move may not touch.
    pub fn shielded_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.shielded.iter().copied().filter(|&edge| edge != Edge::NONE)
    }

    /// The edges the player to move has placed this turn.
    pub fn fresh_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.fresh.iter().copied().filter(|&edge| edge != Edge::NONE)
    }

    #[inline]
    pub(crate) fn is_shielded(&self, edge: Edge) -> bool {
        self.shielded[0] == edge || self.shielded[1] == edge
    }

    /// The result once all actions are played. A game also ends, in a draw, when the player to
    /// move has no legal move; [`Position::outcome_given`] includes that.
    pub fn outcome(&self) -> Option<Outcome> {
        if !self.is_finished() {
            return None;
        }
        Some(match self.score(Player::Blue).cmp(&self.score(Player::Red)) {
            std::cmp::Ordering::Greater => Outcome::Win(Player::Blue),
            std::cmp::Ordering::Less => Outcome::Win(Player::Red),
            std::cmp::Ordering::Equal => Outcome::Draw(DrawReason::EqualScores),
        })
    }

    /// The result, given this position's legal moves.
    pub fn outcome_given(&self, legal: &LegalMoves) -> Option<Outcome> {
        self.outcome().or_else(|| legal.is_empty().then_some(Outcome::Draw(DrawReason::NoLegalMove)))
    }

    /// Empty once the game is finished. Empty before that, the game is a draw.
    pub fn legal_moves(&self) -> LegalMoves {
        movegen::generate(self)
    }

    /// Checks one move rule by rule with the exact geometric tests, and says what it would do.
    /// It shares no code with [`Position::legal_moves`], so each checks the other.
    pub fn check_move(&self, mv: Move) -> Result<MoveKind, IllegalMove> {
        if self.is_finished() {
            return Err(IllegalMove::GameOver);
        }
        let mover = self.to_move();
        let source = mv.source;
        if !self.nodes(mover).contains(source) {
            return Err(IllegalMove::SourceNotOwned);
        }
        let target = mv.target().ok_or(IllegalMove::OffBoard)?;
        let (start, end) = (source.to_vector(), target.to_vector());

        if self.nodes(mover).iter().any(|node| strictly_inside(start, end, node.to_vector())) {
            return Err(IllegalMove::PathThroughOwnNode);
        }
        let kind = self.kind_at(target);
        let target_inside_own_edge = || {
            self.edges(mover).iter().any(|edge| strictly_inside(edge.origin().to_vector(), edge.far().to_vector(), end))
        };
        if kind != MoveKind::Connect && target_inside_own_edge() {
            return Err(IllegalMove::TargetOnOwnEdge);
        }
        let runs_along = |edge: Edge| segments_overlap(start, end, edge.origin().to_vector(), edge.far().to_vector());
        if self.edges(mover).iter().any(runs_along) {
            return Err(IllegalMove::OverlapsEdge);
        }
        let touched = || self.edges(mover.opponent()).iter().filter(|&edge| touches(edge, source, target));
        if touched().any(|edge| self.is_shielded(edge)) {
            return Err(IllegalMove::BreaksShieldedEdge);
        }
        match touched().count() {
            0 | 1 => Ok(kind),
            several => Err(IllegalMove::BreaksSeveralEdges(several as u8)),
        }
    }

    pub fn apply(&mut self, mv: Move) -> Result<MoveOutcome, IllegalMove> {
        self.check_move(mv)?;
        Ok(self.apply_unchecked(mv))
    }

    /// Plays a move known to be legal, such as one from [`Position::legal_moves`]. An illegal
    /// move breaks the position; debug builds check.
    pub fn apply_unchecked(&mut self, mv: Move) -> MoveOutcome {
        debug_assert_eq!(self.check_move(mv).err(), None, "apply_unchecked was given an illegal move: {mv:?}");
        let mover = self.to_move();
        let opponent = mover.opponent();
        let source = mv.source;
        let target = mv.target().expect("legal moves end on the board");
        let placed = Edge::from_move(source, mv.direction).expect("legal moves end on the board");
        let kind = self.kind_at(target);

        // Cut the opposing edge the move touches, if any. For a capture this also removes the
        // opponent's node at the target, which is left with no edges.
        let tables = tables();
        let broken = self.edges(opponent).iter().find(|&edge| tables.touches(source, mv.direction, edge));
        if let Some(broken) = broken {
            self.remove_edge(opponent, broken);
        }

        // Only a new loop changes the mover's area, and closing one takes a connect or an edge
        // that crosses one of the mover's edges.
        let may_close_loop = kind == MoveKind::Connect
            || self.edges(mover).iter().any(|edge| {
                tables.touches(source, mv.direction, edge) && !edge.has_endpoint(source) && !edge.has_endpoint(target)
            });
        let inserted = self.edges[mover.index()].insert(placed);
        debug_assert!(inserted);
        self.nodes[mover.index()].insert(target);
        if may_close_loop {
            self.refresh_area(mover);
        }
        let unused = self.fresh.iter().position(|&edge| edge == Edge::NONE).expect("at most two actions a turn");
        self.fresh[unused] = placed;
        self.fresh.sort_unstable();

        self.actions_played += 1;
        let turn_ended = self.actions_played % 2 == 1;
        let game_over = self.is_finished();
        let scored = (turn_ended || game_over).then(|| {
            for player in Player::BOTH {
                self.scores[player.index()] += self.areas[player.index()];
            }
            self.areas
        });
        if game_over {
            // Keeps finished positions canonical: nothing is shielded or fresh any more.
            self.shielded = [Edge::NONE; 2];
            self.fresh = [Edge::NONE; 2];
        } else if turn_ended {
            self.shielded = std::mem::replace(&mut self.fresh, [Edge::NONE; 2]);
        }
        MoveOutcome { kind, placed, broken, scored, turn_ended, game_over }
    }

    fn kind_at(&self, target: Point) -> MoveKind {
        match self.node_owner(target) {
            None => MoveKind::Extend,
            Some(owner) if owner == self.to_move() => MoveKind::Connect,
            Some(_) => MoveKind::Capture,
        }
    }

    /// Removes the edge, and any of its ends left with no edges.
    fn remove_edge(&mut self, owner: Player, edge: Edge) {
        let removed = self.edges[owner.index()].remove(edge);
        debug_assert!(removed);
        for end in [edge.origin(), edge.far()] {
            if !self.edges(owner).touches_node(end) {
                self.nodes[owner.index()].remove(end);
            }
        }
        // Removing an edge can only shrink an area, so an empty one stays empty.
        if !self.areas[owner.index()].is_zero() {
            self.refresh_area(owner);
        }
    }

    fn refresh_area(&mut self, player: Player) {
        self.areas[player.index()] = enclosed_area(self.edges(player).as_slice());
    }

    /// Rebuilds the position from its edges and compares, which also checks everything
    /// [`Position::from_parts`] checks. Thorough, not fast: for tests and debugging.
    pub fn validate(&self) -> Result<(), String> {
        let blue: Vec<Edge> = self.edges(Player::Blue).iter().collect();
        let red: Vec<Edge> = self.edges(Player::Red).iter().collect();
        let shielded: Vec<Edge> = self.shielded_edges().collect();
        let fresh: Vec<Edge> = self.fresh_edges().collect();
        let rebuilt = Position::from_parts(&blue, &red, self.actions_played, self.scores, &shielded, &fresh)
            .map_err(|error| error.to_string())?;
        if rebuilt != *self {
            return Err(format!("cached state is stale\n stored: {self:?}\nrebuilt: {rebuilt:?}"));
        }
        Ok(())
    }
}

/// Who plays the action after `actions_played` actions: Blue in even turns, Red in odd ones.
fn mover_after(actions_played: u8) -> Player {
    let turn = actions_played.div_ceil(2);
    if turn % 2 == 0 { Player::Blue } else { Player::Red }
}

/// How many of the first `actions_played` actions were `player`'s.
fn actions_made_by(player: Player, actions_played: u8) -> usize {
    (0..actions_played).filter(|&before| mover_after(before) == player).count()
}

/// Whether the segment from `a` to `b` shares a point with `edge`.
#[inline]
fn touches(edge: Edge, a: Point, b: Point) -> bool {
    segments_touch(a.to_vector(), b.to_vector(), edge.origin().to_vector(), edge.far().to_vector())
}

fn sorted_pair(edges: &[Edge]) -> [Edge; 2] {
    let mut pair = [Edge::NONE; 2];
    pair[..edges.len()].copy_from_slice(edges);
    pair.sort_unstable();
    pair
}

/// Hashes what defines the position; the rest follows from it.
impl Hash for Position {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&self.edges, &self.scores, &self.shielded, &self.fresh, self.actions_played).hash(state);
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Position")
            .field("actions_played", &self.actions_played)
            .field("to_move", &self.to_move())
            .field("blue", &self.edges(Player::Blue))
            .field("red", &self.edges(Player::Red))
            .field("scores", &self.scores)
            .field("areas", &self.areas)
            .field("shielded", &self.shielded)
            .field("fresh", &self.fresh)
            .finish()
    }
}
