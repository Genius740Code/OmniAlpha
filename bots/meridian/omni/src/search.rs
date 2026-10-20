//! Omni's strategy: a wall-clock-guarded iterative-deepening alpha-beta search
//! over actions (forked from Scout's 2-ply shape).
//!
//! Scout's old shape is kept as [`analyze_2ply`]: value a position by points +
//! enclosed area counted as if held for up to `horizon` more scoring events,
//! plus `room_weight` for the room the nodes span; try own actions longest
//! first, keep the best `width`, value each by the best reply. The new
//! [`analyze_limited`] runs the same idea to any depth:
//!
//! * iterative deepening, one action per ply, stopping at a wall-clock
//!   deadline or a node budget, so a move can never overrun its time;
//! * alpha-beta in negamax form, with the turn structure of Meridian: the
//!   mover of a two-action turn keeps the same sign for their second action;
//! * a transposition table keyed by the engine's exact position hash;
//! * cut extensions: an action that cuts an opponent edge does not cost depth,
//!   so every cut and its answer stays inside the horizon (up to 4 per line);
//! * move ordering: transposition-table move first, then longest edges first;
//!   interior nodes cap at `branch` actions, root candidates at `width`;
//! * the first iteration is Scout's depth 2 (own action + best reply), and
//!   the root candidate list comes from Scout's 1-ply ranking with its loop
//!   bonus, so the search is at least Scout-shaped when it runs out of time.
//!
//! Leaf values skip the "no legal move ⇒ draw" move generation at depth 0:
//! interior nodes already generate the moves there, so a stuck player is
//! caught one action later instead of never.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::time::Instant;

use meridian_engine::{Direction, LegalMoves, Move, Outcome, Player, Point, Position};

/// Positions searched for a move in a game. Analysis asks for its own number, up to this.
pub const MOVE_BUDGET: usize = 4096;
const SMALLEST_BUDGET: usize = 16;
/// Beyond the largest score the evaluation can reach, so finished games sort by result.
const WIN: f64 = 1.0e6;
const INF: f64 = 1.0e12;
/// How many cut extensions one line may use.
const MAX_EXTENSIONS: u8 = 4;
/// Transposition table size limit: kept well under the site's 256 MiB.
const TT_LIMIT: usize = 1 << 19;
/// How deep the deadline checks are amortised: every 32 nodes the clock is read.
const STOP_EVERY: usize = 32;
/// A new depth iteration is started only if this share of the budget is still unused.
const SOFT_STOP_SHARE: f64 = 0.3;

/// Tunable search parameters. Defaults keep Scout's values for `width`,
/// `horizon`, `room_weight` and `budget`.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// How many root candidates get searched each iteration.
    pub width: usize,
    /// Enclosed area counts as if held for at most this many more scoring events.
    pub horizon: u8,
    /// What the room a player's nodes span is worth, per unit of area.
    pub room_weight: f64,
    /// Legacy: positions searched by [`analyze_2ply`], nodes visited by the deep search.
    pub budget: usize,
    /// How many actions interior nodes try, 0 for all of them.
    pub branch: usize,
    /// The deepest iteration, counted in actions.
    pub depth_max: u8,
}

impl Default for Params {
    fn default() -> Self {
        Params { width: 8, horizon: 12, room_weight: 0.4, budget: MOVE_BUDGET, branch: 24, depth_max: 8 }
    }
}

/// What stops the deep search: a wall-clock deadline, a node budget, or both.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// `Instant::now()` at which the search must have finished and backed out.
    pub deadline: Option<Instant>,
    /// How many nodes (leaf values or searched positions) may be visited.
    pub node_budget: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { deadline: None, node_budget: usize::MAX }
    }
}

pub struct Analysis {
    /// Blue's lead after the best move and its best reply line.
    pub evaluation: f64,
    /// Actions deep the best candidate was searched.
    pub depth: usize,
    /// Positions searched.
    pub nodes: usize,
    /// The best moves first.
    pub candidates: Vec<Candidate>,
}

pub struct Candidate {
    pub mv: Move,
    /// Blue's lead after this move and the best line found.
    pub evaluation: f64,
    /// This move and the expected replies.
    pub pv: Vec<Move>,
    /// 1, plus the replies searched.
    pub visits: usize,
}

pub fn best_move(position: &Position) -> Option<Move> {
    best_move_with(position, &Params::default())
}

pub fn best_move_with(position: &Position, params: &Params) -> Option<Move> {
    // Without a deadline the node budget is the only stop; cap it hard so a
    // huge `budget` can never turn into an unbounded search.
    let limits = Limits { deadline: None, node_budget: params.budget.clamp(1, 1 << 24) };
    best_move_limited(position, params, &limits)
}

pub fn best_move_limited(position: &Position, params: &Params, limits: &Limits) -> Option<Move> {
    analyze_limited(position, params, limits).candidates.first().map(|candidate| candidate.mv)
}

pub fn analyze(position: &Position, budget: usize) -> Analysis {
    let mut params = Params::default();
    params.budget = budget;
    analyze_with(position, &params)
}

pub fn analyze_with(position: &Position, params: &Params) -> Analysis {
    let limits = Limits { deadline: None, node_budget: params.budget.clamp(1, 1 << 24) };
    analyze_limited(position, params, &limits)
}

/// The deep search: iterative-deepening alpha-beta under [`Limits`]. Always returns
/// a legal candidate when the position has one, even on an immediate abort: the
/// 1-ply ranking pass runs under the same deadline and its best action stands.
pub fn analyze_limited(position: &Position, params: &Params, limits: &Limits) -> Analysis {
    let started = Instant::now();
    let legal = position.legal_moves();
    if position.is_finished() || legal.is_empty() {
        return Analysis { evaluation: value(position, params), depth: 0, nodes: 0, candidates: Vec::new() };
    }
    if legal.len() == 1 {
        let mv = legal.iter().next().expect("one legal move");
        return Analysis {
            evaluation: value(position, params),
            depth: 0,
            nodes: 0,
            candidates: vec![Candidate { mv, evaluation: value(position, params), pv: vec![mv], visits: 1 }],
        };
    }

    // Scout's 1-ply ranking with loop bonus orders the root, under the deadline.
    let mut candidates = ranked(position, MOVE_BUDGET / 2, true, params, limits.deadline);
    if candidates.is_empty() {
        // The deadline passed before even one ranking value: play anything legal.
        let mv = legal.iter().next().expect("position is not finished");
        return Analysis {
            evaluation: value(position, params),
            depth: 1,
            nodes: 1,
            candidates: vec![Candidate { mv, evaluation: value(position, params), pv: vec![mv], visits: 1 }],
        };
    }
    candidates.truncate(params.width.max(1));

    let mut searcher = Searcher::new(params, limits);
    let mut best: Option<(Vec<(Move, f64, usize)>, usize)> = None; // (per-candidate values, nodes used)
    for depth in 2..=usize::from(params.depth_max.max(2)) {
        if best.is_some() && !affordable(&searcher, started, limits) {
            break;
        }
        match searcher.search_root(position, &candidates, depth) {
            Ok(values) => {
                let nodes = searcher.nodes;
                // Order the root by the new values for the next, deeper iteration.
                let side = sign(position.to_move());
                candidates.sort_by(|a, b| {
                    let value = |mv: &Move| {
                        values.iter().find(|(m, _, _)| m == mv).map_or(-INF, |(_, v, _)| *v)
                    };
                    (side * value(&b.0)).total_cmp(&(side * value(&a.0))).then(a.0.index().cmp(&b.0.index()))
                });
                best = Some((values, nodes));
            }
            Err(()) => break,
        }
    }

    let mover = position.to_move();
    let evaluation_of = |value: f64| sign(mover) * value;
    match best {
        Some((values, nodes)) => {
            let depth = searcher.completed_depth;
            let mut list: Vec<Candidate> = candidates
                .iter()
                .map(|(mv, after, _)| {
                    let found = values.iter().find(|(m, _, _)| m == mv);
                    let (value, visits) = found.map_or((-INF, 1usize), |(_, v, n)| (*v, *n));
                    Candidate {
                        mv: *mv,
                        evaluation: evaluation_of(value),
                        pv: std::iter::once(*mv).chain(searcher.pv(after, 8)).collect(),
                        visits,
                    }
                })
                .collect();
            list.sort_by(|a, b| {
                (sign(mover) * b.evaluation).total_cmp(&(sign(mover) * a.evaluation)).then(a.mv.index().cmp(&b.mv.index()))
            });
            let evaluation = list.first().map_or_else(|| value(position, params), |c| c.evaluation);
            Analysis { evaluation, depth, nodes, candidates: list }
        }
        None => {
            // No iteration completed: Scout's 1-ply order stands, as the old search did
            // when its budget ran out during the ranking pass.
            let list: Vec<Candidate> = candidates
                .iter()
                .map(|(mv, _, one_ply)| Candidate {
                    mv: *mv,
                    evaluation: evaluation_of(*one_ply),
                    pv: vec![*mv],
                    visits: 1,
                })
                .collect();
            let evaluation = list.first().map_or_else(|| value(position, params), |c| c.evaluation);
            Analysis { evaluation, depth: 1, nodes: searcher.nodes, candidates: list }
        }
    }
}

/// Scout's own search, kept exactly: the best `width` actions by 1-ply value, each
/// valued by the best of up to `budget` replies. This is the A/B baseline.
pub fn analyze_2ply(position: &Position, params: &Params) -> Analysis {
    let budget = params.budget.clamp(SMALLEST_BUDGET, MOVE_BUDGET);
    let first_actions = ranked(position, budget / 2, true, params, None);
    let mut nodes = first_actions.len();
    let mut depth = usize::from(nodes > 0);
    let width = first_actions.len().min(params.width);
    let reply_budget = (budget - nodes) / width.max(1);

    let mut candidates = Vec::new();
    for (mv, after, mut evaluation) in first_actions.into_iter().take(width) {
        let replies = ranked(&after, reply_budget, false, params, None);
        nodes += replies.len();
        let mut pv = vec![mv];
        if let Some((reply, _, after_reply)) = replies.first() {
            evaluation = *after_reply;
            pv.push(*reply);
            depth = 2;
        }
        candidates.push(Candidate { mv, evaluation, pv, visits: 1 + replies.len() });
    }
    let side = sign(position.to_move());
    candidates
        .sort_by(|a, b| (side * b.evaluation).total_cmp(&(side * a.evaluation)).then(a.mv.index().cmp(&b.mv.index())));
    let evaluation = candidates.first().map_or_else(|| value(position, params), |best| best.evaluation);
    Analysis { evaluation, depth, nodes, candidates }
}

/// Blue's lead as Omni sees it. Positive favours Blue.
pub fn evaluate(position: &Position) -> f64 {
    evaluate_with(position, &Params::default())
}

pub fn evaluate_with(position: &Position, params: &Params) -> f64 {
    let events_left = f64::from(position.scoring_events_left());
    let horizon = f64::from(params.horizon);
    let worth = |player| {
        position.score(player).to_f64()
            + position.area(player).to_f64() * events_left.min(horizon)
            + room(position, player) * params.room_weight * events_left.min(1.0)
    };
    worth(Player::Blue) - worth(Player::Red)
}

/// [`evaluate`], except that a drawn game is worth 0.
fn value(position: &Position, params: &Params) -> f64 {
    let drawn = matches!(position.outcome_given(&position.legal_moves()), Some(Outcome::Draw(_)));
    if drawn { 0.0 } else { evaluate_with(position, params) }
}

/// The mover's actions, best first, with the position after each and its value. Only the first
/// `budget` of them, longest edges first, are tried, and `deadline` can cut the pass short.
fn ranked(
    position: &Position,
    budget: usize,
    plan_loops: bool,
    params: &Params,
    deadline: Option<Instant>,
) -> Vec<(Move, Position, f64)> {
    let mover = position.to_move();
    let room_before = room(position, mover);
    let mut moves: Vec<Move> =
        position.legal_moves().iter().filter(|&mv| !repeats_a_connection(position, mv)).collect();
    moves.sort_by_key(|mv| (Reverse(length(*mv)), mv.index()));

    let mut tried: Vec<(f64, (Move, Position, f64))> = Vec::with_capacity(moves.len().min(budget));
    for (count, mv) in moves.into_iter().take(budget).enumerate() {
        if count % STOP_EVERY == 0 && count > 0 {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
        }
        let mut after = position.clone();
        after.apply_unchecked(mv);
        let value = value(&after, params);
        let mut priority = sign(mover) * value;
        if plan_loops && after.to_move() == mover && !after.is_finished() {
            priority += loop_bonus(position, mv, &after, room_before, params);
        }
        tried.push((priority, (mv, after, value)));
    }
    tried.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1 .0.index().cmp(&b.1 .0.index())));
    tried.into_iter().map(|(_, entry)| entry).collect()
}

/// A connection between two of the mover's nodes is listed from both ends. This is the second.
fn repeats_a_connection(position: &Position, mv: Move) -> bool {
    let target = mv.target().expect("legal moves end on the board");
    position.nodes(position.to_move()).contains(target) && target < mv.source
}

fn length(mv: Move) -> i8 {
    mv.direction.dx().abs().max(mv.direction.dy().abs())
}

/// For a first action the second can close into a triangle: the room it adds, up to that
/// triangle's area, for the scoring events left. It keeps such plans ahead of long edges that
/// cannot be closed in time.
fn loop_bonus(position: &Position, mv: Move, after: &Position, room_before: f64, params: &Params) -> f64 {
    let mover = position.to_move();
    let target = mv.target().expect("legal moves end on the board");
    let mut triangle: f64 = 0.0;
    for edge in position.edges(mover).iter().filter(|edge| edge.has_endpoint(mv.source)) {
        let (origin, far) = edge.endpoints();
        let third = if origin == mv.source { far } else { origin };
        let closes = Move::between(target, third).is_some_and(|closing| after.check_move(closing).is_ok());
        if closes {
            triangle = triangle.max(f64::from(cross(mv.source, target, third).abs()) * 0.5);
        }
    }
    let added_room = (room(after, mover) - room_before).max(0.0).min(triangle);
    added_room * f64::from(after.scoring_events_left().min(params.horizon))
}

// The deep search.

/// Transposition table flags.
const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;

#[derive(Clone, Copy)]
struct TtEntry {
    /// The value for the player to move at the stored position.
    value: f64,
    /// How many actions deep the position was searched.
    depth: u8,
    flag: u8,
    best: Option<Move>,
}

/// The error type of the search: `Err(())` means the deadline or node budget
/// ended it, and the caller keeps the last completed iteration.
struct Searcher<'a> {
    params: &'a Params,
    deadline: Option<Instant>,
    node_budget: usize,
    nodes: usize,
    stopped: bool,
    completed_depth: usize,
    tt: HashMap<u64, TtEntry, BuildHasherDefault<FxHasher>>,
    /// The directions of each edge length, index order within each length.
    by_length: [Vec<Direction>; 4],
    /// Every direction, for the capture scan.
    all_directions: Vec<Direction>,
}

impl<'a> Searcher<'a> {
    fn new(params: &'a Params, limits: &Limits) -> Searcher<'a> {
        let mut by_length: [Vec<Direction>; 4] = Default::default();
        for direction in Direction::all() {
            by_length[length_of(direction) as usize].push(direction);
        }
        Searcher {
            params,
            deadline: limits.deadline,
            node_budget: limits.node_budget,
            nodes: 0,
            stopped: false,
            completed_depth: 0,
            tt: HashMap::default(),
            by_length,
            all_directions: Direction::all().collect(),
        }
    }

    /// Whether to stop. Called at most every [`STOP_EVERY`] nodes, so a move
    /// overshoots the deadline by at most a few dozen node visits.
    fn check_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes >= self.node_budget {
            self.stopped = true;
            return true;
        }
        if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            self.stopped = true;
            return true;
        }
        false
    }

    /// One depth of the root: every candidate searched with a full window, so
    /// its value is exact and the analysis list stays honest.
    fn search_root(
        &mut self,
        position: &Position,
        candidates: &[(Move, Position, f64)],
        depth: usize,
    ) -> Result<Vec<(Move, f64, usize)>, ()> {
        let mover = position.to_move();
        let mut values = Vec::with_capacity(candidates.len());
        for (mv, after, _) in candidates {
            let before = self.nodes;
            self.nodes += 1;
            if self.check_stop() {
                return Err(());
            }
            let value = if after.is_finished() {
                sign(mover) * terminal_blue(after)
            } else if after.to_move() == mover {
                self.negamax(after, depth - 1, 1, -INF, INF, 0)?
            } else {
                -self.negamax(after, depth - 1, 1, -INF, INF, 0)?
            };
            values.push((*mv, value, self.nodes - before));
        }
        self.completed_depth = depth;
        Ok(values)
    }

    /// The value of `position` for its mover, searching `depth` more actions.
    /// The mover of an unfinished turn keeps the same sign for their next action.
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        position: &Position,
        depth: usize,
        ply: usize,
        alpha: f64,
        beta: f64,
        extensions: u8,
    ) -> Result<f64, ()> {
        if position.is_finished() {
            return Ok(sign(position.to_move()) * terminal_blue(position));
        }
        self.nodes += 1;
        if self.nodes % STOP_EVERY == 0 && self.check_stop() {
            return Err(());
        }
        if depth == 0 {
            return Ok(sign(position.to_move()) * evaluate_with(position, self.params));
        }
        let legal = position.legal_moves();
        if legal.is_empty() {
            // Rule 9: the player to move has no legal move, and the game is drawn.
            return Ok(0.0);
        }

        let key = hash_of(position);
        let mut tt_move = None;
        if let Some(entry) = self.tt.get(&key) {
            tt_move = entry.best;
            if usize::from(entry.depth) >= depth {
                match entry.flag {
                    EXACT => return Ok(entry.value),
                    LOWER if entry.value >= beta => return Ok(entry.value),
                    UPPER if entry.value <= alpha => return Ok(entry.value),
                    _ => {}
                }
            }
        }

        let mover = position.to_move();
        let mut moves = self.ordered(position, &legal, tt_move, ply);
        if moves.is_empty() {
            // Only the duplicate listing of a connect is filtered, so this cannot
            // happen in a legal position; keep the search alive regardless.
            moves.push(legal.iter().next().expect("position has a legal move"));
        }
        let mut alpha = alpha;
        let mut best = -INF;
        let mut best_move = None;
        let mut flag = UPPER;
        let mut capped = false;
        let mut index = 0usize;
        for mv in moves {
            index += 1;
            if index > self.branch_width() {
                capped = true;
                break;
            }
            let mut child = position.clone();
            let outcome = child.apply_unchecked(mv);
            let value = if child.is_finished() {
                sign(mover) * terminal_blue(&child)
            } else if outcome.turn_ended {
                -self.negamax(&child, depth - 1, ply + 1, -beta, -alpha, extensions)?
            } else {
                let extend = outcome.broken.is_some() && extensions < MAX_EXTENSIONS;
                let child_depth = if extend { depth } else { depth - 1 };
                let child_extensions = extensions + u8::from(extend);
                self.negamax(&child, child_depth, ply + 1, alpha, beta, child_extensions)?
            };
            if value > best {
                best = value;
                best_move = Some(mv);
            }
            if best > alpha {
                alpha = best;
                flag = EXACT;
            }
            if alpha >= beta {
                flag = LOWER;
                break;
            }
        }

        // A node whose moves were capped only proves a lower bound, so it must not
        // be trusted as exact or as an upper bound.
        let flag = if capped && flag != LOWER { LOWER } else { flag };
        if self.tt.len() >= TT_LIMIT {
            self.tt.clear();
        }
        self.tt.insert(
            key,
            TtEntry { value: best, depth: depth.min(u8::MAX as usize) as u8, flag, best: best_move },
        );
        Ok(best)
    }

    fn branch_width(&self) -> usize {
        if self.params.branch == 0 { usize::MAX } else { self.params.branch }
    }

    /// The legal moves, the most forcing first: captures (an opponent node falls,
    /// with it an edge), then the transposition move, then the longest edges. Only
    /// the first `branch` are collected; iteration order inside a length is move
    /// index order, so within a length the order is Scout's.
    fn ordered(&mut self, position: &Position, legal: &LegalMoves, tt_move: Option<Move>, ply: usize) -> Vec<Move> {
        let _ = ply;
        let mover = position.to_move();
        let opponent = mover.opponent();
        let cap = self.branch_width();
        let mut list: Vec<Move> = Vec::with_capacity(64);
        // Captures first, from every length: they are the moves the opponent must
        // answer, and the ones a capped reply set can least afford to miss.
        for (source, targets) in legal.by_source() {
            for &direction in &self.all_directions {
                if !targets.contains(direction) {
                    continue;
                }
                let mv = Move { source, direction };
                if repeats_a_connection(position, mv) {
                    continue;
                }
                if mv.target().is_some_and(|target| position.node_owner(target) == Some(opponent)) {
                    list.push(mv);
                }
            }
        }
        'outer: for length in (1..=3).rev() {
            for (source, targets) in legal.by_source() {
                for &direction in &self.by_length[length] {
                    if !targets.contains(direction) {
                        continue;
                    }
                    let mv = Move { source, direction };
                    if repeats_a_connection(position, mv) {
                        continue;
                    }
                    if mv.target().is_some_and(|target| position.node_owner(target) == Some(opponent)) {
                        continue; // already collected as a capture
                    }
                    list.push(mv);
                    if list.len() >= cap {
                        break 'outer;
                    }
                }
            }
        }
        if let Some(tt_move) = tt_move {
            if list.first() != Some(&tt_move) {
                if let Some(at) = list.iter().position(|&mv| mv == tt_move) {
                    list.remove(at);
                }
                list.insert(0, tt_move);
            }
        }
        list
    }

    /// The expected line after `mv`: the best move of each stored position,
    /// checked for legality against hash collisions.
    fn pv(&self, position: &Position, max: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current = position.clone();
        for _ in 0..max {
            if current.is_finished() {
                break;
            }
            let legal = current.legal_moves();
            if legal.is_empty() {
                break;
            }
            let Some(entry) = self.tt.get(&hash_of(&current)) else { break };
            let Some(mv) = entry.best else { break };
            if !legal.contains(mv) {
                break;
            }
            pv.push(mv);
            current.apply_unchecked(mv);
        }
        pv
    }
}

fn length_of(direction: Direction) -> i8 {
    direction.dx().abs().max(direction.dy().abs())
}

/// The score difference of a finished game, Blue's view.
fn terminal_blue(position: &Position) -> f64 {
    let (blue, red) = (position.score(Player::Blue), position.score(Player::Red));
    match blue.cmp(&red) {
        std::cmp::Ordering::Greater => WIN,
        std::cmp::Ordering::Less => -WIN,
        std::cmp::Ordering::Equal => 0.0,
    }
}

/// Whether starting another iteration can still pay off: an iteration costs
/// several times the last one, so past [`SOFT_STOP_SHARE`] of the budget it
/// most likely never finishes.
fn affordable(searcher: &Searcher, started: Instant, limits: &Limits) -> bool {
    if let Some(deadline) = limits.deadline {
        let full = deadline - started;
        if started.elapsed() >= full.mul_f64(SOFT_STOP_SHARE) {
            return false;
        }
    } else if searcher.nodes >= limits.node_budget / 10 * 3 {
        return false;
    }
    true
}

fn hash_of(position: &Position) -> u64 {
    let mut hasher = FxHasher::default();
    position.hash(&mut hasher);
    hasher.finish()
}

/// The area of the convex hull of a player's nodes: room to grow into.
fn room(position: &Position, player: Player) -> f64 {
    let mut points: Vec<Point> = position.nodes(player).iter().collect();
    points.sort_by_key(|point| (point.x(), point.y()));
    if points.len() < 3 {
        return 0.0;
    }
    let mut hull = half_hull(points.iter().copied());
    hull.extend(half_hull(points.iter().rev().copied()));
    let first = hull[0];
    hull.windows(2).map(|pair| cross(first, pair[0], pair[1]) as f64).sum::<f64>().abs() * 0.5
}

/// One side of the convex hull of points sorted left to right (Andrew's monotone chain), without
/// its last point, which starts the other side.
fn half_hull(points: impl Iterator<Item = Point>) -> Vec<Point> {
    let mut chain: Vec<Point> = Vec::new();
    for point in points {
        while chain.len() >= 2 && cross(chain[chain.len() - 2], chain[chain.len() - 1], point) <= 0 {
            chain.pop();
        }
        chain.push(point);
    }
    chain.pop();
    chain
}

/// Twice the signed area of the triangle `a`, `b`, `c`: positive when it turns counter-clockwise.
fn cross(a: Point, b: Point, c: Point) -> i32 {
    i32::from(b.x() - a.x()) * i32::from(c.y() - a.y()) - i32::from(b.y() - a.y()) * i32::from(c.x() - a.x())
}

/// 1 for Blue and -1 for Red, to turn Blue's lead into the mover's.
fn sign(player: Player) -> f64 {
    if player == Player::Blue { 1.0 } else { -1.0 }
}

/// FNV-flavoured Fx hashing: fast on the ~300 bytes a position hashes, and the
/// engine's exact `Hash`, so identical positions always collide.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.hash = (self.hash.rotate_left(5) ^ u64::from_le_bytes(chunk.try_into().expect("8 bytes")))
                .wrapping_mul(FX_SEED);
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.hash = (self.hash.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(FX_SEED);
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.hash = (self.hash.rotate_left(5) ^ u64::from(value)).wrapping_mul(FX_SEED);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.hash = (self.hash.rotate_left(5) ^ value).wrapping_mul(FX_SEED);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
