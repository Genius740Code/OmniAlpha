//! Text for points, moves, positions and games. Reading accepts exactly what writing produces.
//!
//! * A point is a column letter `A`–`S` (x from −9 to 9) and a row number `1`–`19` (y from −9 to
//!   9): `D10` is (−6, 0). A move is its two points, `D10-D13`.
//! * A position is written in brackets: `[A:12 B:A10-D10-D7,D10=D13 R:P10-S10 S:5,3.5]`. `A` is
//!   the number of actions played, `B` and `R` are Blue's and Red's edges as paths, and `S` is
//!   both scores, Blue's first. The fields come in that order, one space apart, and empty ones
//!   are left out, so the starting position is `[B:A10-D10 R:P10-S10]`. `=` in place of `-` marks
//!   an edge from the latest turns: from the opponent's last turn, so it cannot be touched now,
//!   or from earlier in the current turn.
//! * A game, a [`Line`], is its position and then one word per turn:
//!   `[B:A10-D10 R:P10-S10] D10-D13 P10-P13-M13 D13-G13,D10-D7`. A turn's two moves are one chain
//!   when the second starts where the first ended, and are joined by a comma otherwise. Each word
//!   ends where its turn ends: after a position taken mid-turn, the first word finishes that turn,
//!   and the last word may stop partway through one.

use std::fmt;

use crate::edge::Edge;
use crate::game::Game;
use crate::geometry::Point;
use crate::moves::Move;
use crate::position::{InvalidPosition, Position, TOTAL_ACTIONS};
use crate::units::{AREA_DENOMINATOR, Player, Score, gcd};

/// Why a text could not be read, as a sentence for people.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotationError(String);

impl fmt::Display for NotationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotationError {}

fn error<T>(message: impl Into<String>) -> Result<T, NotationError> {
    Err(NotationError(message.into()))
}

/// A position and legal moves played from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    start: Position,
    moves: Vec<Move>,
}

impl Line {
    /// Refuses the first illegal move, saying which action it is.
    pub fn new(start: Position, moves: Vec<Move>) -> Result<Line, NotationError> {
        let mut game = Game::from_position(start.clone());
        for (index, &mv) in moves.iter().enumerate() {
            if let Err(reason) = game.play(mv) {
                let action = usize::from(start.actions_played()) + index + 1;
                return error(format!("Action {action}: {reason}."));
            }
        }
        Ok(Line { start, moves })
    }

    pub fn parse(text: &str) -> Result<Line, NotationError> {
        let Some(after_bracket) = text.trim().strip_prefix('[') else {
            return error("A line starts with its position, such as [B:A10-D10 R:P10-S10].");
        };
        let Some((fields, turns)) = after_bracket.split_once(']') else {
            return error("The position is missing its closing ].");
        };
        let start = parse_fields(fields)?;
        let mut position = start.clone();
        let mut moves = Vec::new();
        let words: Vec<&str> = turns.split_whitespace().collect();
        for (index, word) in words.iter().enumerate() {
            let turn = position.turn_index();
            play_turn(word, &mut position, &mut moves)?;
            let is_last = index + 1 == words.len();
            if !is_last && position.turn_index() == turn && !position.is_finished() {
                return error(format!("{word} is only part of a turn. A turn's two moves are one word."));
            }
        }
        Ok(Line { start, moves })
    }

    #[inline]
    pub fn start(&self) -> &Position {
        &self.start
    }

    #[inline]
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// The game after all the moves.
    pub fn game(&self) -> Game {
        let mut game = Game::from_position(self.start.clone());
        for &mv in &self.moves {
            game.play(mv).expect("a line holds legal moves");
        }
        game
    }
}

/// Plays the moves of one word, `D13-G13-G10` or `D13-G13,D10-D7`, which must stay in one turn.
fn play_turn(word: &str, position: &mut Position, moves: &mut Vec<Move>) -> Result<(), NotationError> {
    let turn = position.turn_index();
    let mut chain_end: Option<Point> = None;
    for chain in word.split(',') {
        let names: Vec<&str> = chain.split('-').collect();
        if names.len() < 2 || names.contains(&"") {
            return error(format!("{word} is not a turn."));
        }
        let points = names
            .iter()
            .map(|name| parse_square(name).ok_or_else(|| NotationError(format!("{name} is not a point."))))
            .collect::<Result<Vec<_>, _>>()?;
        if chain_end == Some(points[0]) {
            return error(format!("{word}: a move that starts where the last one ended joins its chain."));
        }
        for pair in points.windows(2) {
            let (source, target) = (pair[0], pair[1]);
            if position.turn_index() != turn {
                return error(format!("{word} runs past the end of its turn."));
            }
            let action = usize::from(position.actions_played()) + 1;
            let named = move_text(source, target);
            let Some(mv) = Move::between(source, target) else {
                return error(format!("Action {action}, {named}: the points must be 1 to 3 apart."));
            };
            if let Err(reason) = position.apply(mv) {
                return error(format!("Action {action}, {named}: {reason}."));
            }
            moves.push(mv);
        }
        chain_end = points.last().copied();
    }
    Ok(())
}

impl fmt::Display for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut words = vec![setup_text(&self.start)];
        let mut position = self.start.clone();
        let mut moves = self.moves.iter().copied().peekable();
        while let Some(first) = moves.next() {
            let turn = position.turn_index();
            let first_target = first.target().expect("legal moves end on the board");
            let mut word = move_text(first.source, first_target);
            position.apply_unchecked(first);
            if let Some(second) = moves.next_if(|_| position.turn_index() == turn) {
                let second_target = second.target().expect("legal moves end on the board");
                if second.source == first_target {
                    word = format!("{word}-{}", square(second_target));
                } else {
                    word = format!("{word},{}", move_text(second.source, second_target));
                }
                position.apply_unchecked(second);
            }
            words.push(word);
        }
        f.write_str(&words.join(" "))
    }
}

/// `D10` for (−6, 0).
pub fn square(point: Point) -> String {
    format!("{}{}", char::from(b'A' + point.col() as u8), point.row() + 1)
}

/// The point named `D10`.
pub fn parse_square(text: &str) -> Option<Point> {
    let mut chars = text.chars();
    let letter = chars.next()?;
    let row = whole_number(chars.as_str()).filter(|row| (1..=19).contains(row))?;
    if !('A'..='S').contains(&letter) {
        return None;
    }
    Point::new((letter as u8 - b'A') as i8 - 9, row as i8 - 10)
}

/// A number written the way `Display` writes one that is not negative: digits, no leading zero.
fn whole_number(text: &str) -> Option<i64> {
    let digits_only = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    let no_leading_zero = text == "0" || !text.starts_with('0');
    if digits_only && no_leading_zero { text.parse().ok() } else { None }
}

/// `D10-D13`.
pub fn move_text(source: Point, target: Point) -> String {
    format!("{}-{}", square(source), square(target))
}

/// The bracketed position, which [`parse_setup`] reads back.
pub fn setup_text(position: &Position) -> String {
    let recent: Vec<Edge> = position.shielded_edges().chain(position.fresh_edges()).collect();
    let mut fields = Vec::new();
    if position.actions_played() > 0 {
        fields.push(format!("A:{}", position.actions_played()));
    }
    for (tag, player) in [("B", Player::Blue), ("R", Player::Red)] {
        let edges: Vec<Edge> = position.edges(player).iter().collect();
        if !edges.is_empty() {
            fields.push(format!("{tag}:{}", paths_text(&edges, &recent)));
        }
    }
    let [blue, red] = Player::BOTH.map(|player| position.score(player));
    if !blue.is_zero() || !red.is_zero() {
        fields.push(format!("S:{},{}", score_text(blue), score_text(red)));
    }
    format!("[{}]", fields.join(" "))
}

/// Reads a bracketed position, such as a request's `start`.
pub fn parse_setup(text: &str) -> Result<Position, NotationError> {
    match text.strip_prefix('[').and_then(|inside| inside.strip_suffix(']')) {
        Some(fields) => parse_fields(fields),
        None => error("A position is written in brackets, such as [B:A10-D10 R:P10-S10]."),
    }
}

fn parse_fields(text: &str) -> Result<Position, NotationError> {
    const TAGS: [&str; 4] = ["A", "B", "R", "S"];
    let mut actions = 0;
    let mut edges: [Vec<Edge>; 2] = Default::default();
    let mut scores = [Score::ZERO; 2];
    let mut recent = Vec::new();
    let mut next_tag = 0;
    for field in text.split(' ').filter(|_| !text.is_empty()) {
        if field.is_empty() {
            return error("Fields are one space apart.");
        }
        let tagged =
            field.split_once(':').and_then(|(tag, value)| Some((TAGS.iter().position(|known| *known == tag)?, value)));
        let Some((tag, value)) = tagged else {
            return error(format!("{field} is not a field. Fields are A:, B:, R: and S:, one space apart."));
        };
        if tag < next_tag {
            return error("Fields go in the order A:, B:, R:, S:, each once.");
        }
        next_tag = tag + 1;
        match TAGS[tag] {
            "A" => match whole_number(value).and_then(|count| u8::try_from(count).ok()) {
                Some(count) if count <= TOTAL_ACTIONS => actions = count,
                _ => return error(format!("A: takes the actions played, 0 to {TOTAL_ACTIONS}.")),
            },
            "B" => edges[0] = parse_paths(value, &mut recent)?,
            "R" => edges[1] = parse_paths(value, &mut recent)?,
            _ => {
                let both = value.split_once(',').and_then(|(blue, red)| Some([parse_score(blue)?, parse_score(red)?]));
                let Some(both) = both else {
                    return error("S: takes two scores, such as S:5,3.5.");
                };
                scores = both;
            }
        }
    }
    Position::setup(&edges[0], &edges[1], actions, scores, &recent)
        .map_err(|reason| NotationError(describe(reason, actions)))
}

/// Reads paths such as `A10-D10=D13,D10-D7`, adding the edges joined by `=` to `recent`.
fn parse_paths(text: &str, recent: &mut Vec<Edge>) -> Result<Vec<Edge>, NotationError> {
    let mut edges = Vec::new();
    for path in text.split(',') {
        if path.is_empty() {
            return error("A path is empty.");
        }
        let edges_before = edges.len();
        let mut rest = path;
        let mut previous: Option<Point> = None;
        let mut joined_by_equals = false;
        loop {
            let name_end = rest.find(['-', '=']).unwrap_or(rest.len());
            let name = &rest[..name_end];
            let Some(point) = parse_square(name) else {
                return error(format!("{name} is not a point."));
            };
            if let Some(from) = previous {
                let Some(edge) = Edge::between(from, point) else {
                    return error(format!("{}: the points must be 1 to 3 apart.", move_text(from, point)));
                };
                edges.push(edge);
                if joined_by_equals {
                    recent.push(edge);
                }
            }
            previous = Some(point);
            let Some(separator) = rest[name_end..].chars().next() else { break };
            joined_by_equals = separator == '=';
            rest = &rest[name_end + separator.len_utf8()..];
        }
        if edges.len() == edges_before {
            return error(format!("{path} is not a path."));
        }
    }
    Ok(edges)
}

/// A player's edges as few paths as a simple rule gives: each path starts at the lowest point
/// with an odd number of the edges left, or else the lowest point, and always follows the edge
/// to the lowest next point. `=` joins the recent edges.
fn paths_text(edges: &[Edge], recent: &[Edge]) -> String {
    let order = |point: &Point| (point.x(), point.y());
    let other_end = |edge: Edge, end: Point| if edge.origin() == end { edge.far() } else { edge.origin() };
    let mut left: Vec<Edge> = edges.to_vec();
    let mut paths = Vec::new();
    while !left.is_empty() {
        let ends = || left.iter().flat_map(|edge| [edge.origin(), edge.far()]);
        let degree = |point: Point| left.iter().filter(|edge| edge.has_endpoint(point)).count();
        let odd_end = ends().filter(|&point| degree(point) % 2 == 1).min_by_key(order);
        let mut at = odd_end.or_else(|| ends().min_by_key(order)).expect("an edge has ends");
        let mut path = square(at);
        while let Some(index) = (0..left.len())
            .filter(|&index| left[index].has_endpoint(at))
            .min_by_key(|&index| order(&other_end(left[index], at)))
        {
            let edge = left.swap_remove(index);
            at = other_end(edge, at);
            path.push(if recent.contains(&edge) { '=' } else { '-' });
            path.push_str(&square(at));
        }
        paths.push(path);
    }
    paths.join(",")
}

/// Exact: `5` or `3.5`, or `2+1/6` when the value has no finite decimal.
pub fn score_text(score: Score) -> String {
    let whole = score.numerator().div_euclid(AREA_DENOMINATOR);
    let fraction = score.numerator().rem_euclid(AREA_DENOMINATOR);
    if fraction == 0 {
        return whole.to_string();
    }
    // A decimal ends if the fraction's denominator has no prime factors but 2 and 5, and needs as
    // many places as the larger power of the two.
    let mut denominator = AREA_DENOMINATOR / gcd(fraction, AREA_DENOMINATOR);
    let mut powers = [0usize; 2];
    for (power, prime) in powers.iter_mut().zip([2, 5]) {
        while denominator % prime == 0 {
            denominator /= prime;
            *power += 1;
        }
    }
    if denominator != 1 {
        return score.to_string();
    }
    let places = powers[0].max(powers[1]);
    let digits = fraction * 10i64.pow(places as u32) / AREA_DENOMINATOR;
    format!("{whole}.{digits:0places$}")
}

/// Reads a score written as [`score_text`] writes it, or with more decimal places. `None` unless
/// the text is exactly a value a score can have.
pub fn parse_score(text: &str) -> Option<Score> {
    let wholes = |text: &str| whole_number(text)?.checked_mul(AREA_DENOMINATOR);
    // A proper fraction whose denominator divides the score denominator.
    let fraction = |text: &str| -> Option<i64> {
        let (top, bottom) = text.split_once('/')?;
        let (top, bottom) = (whole_number(top)?, whole_number(bottom)?);
        (0 < top && top < bottom && AREA_DENOMINATOR % bottom == 0).then(|| top * (AREA_DENOMINATOR / bottom))
    };
    let numerator = if let Some((whole, rest)) = text.split_once('+') {
        wholes(whole)?.checked_add(fraction(rest)?)?
    } else if let Some((whole, digits)) = text.split_once('.') {
        if digits.is_empty() || digits.len() > 6 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let tens = 10i64.pow(digits.len() as u32);
        let scaled = digits.parse::<i64>().ok()? * AREA_DENOMINATOR;
        if scaled % tens != 0 {
            return None;
        }
        wholes(whole)?.checked_add(scaled / tens)?
    } else {
        wholes(text)?
    };
    Some(Score::from_numerator(numerator))
}

/// A sentence saying why a set-up with `actions` actions played is not a position.
pub fn describe(reason: InvalidPosition, actions: u8) -> String {
    let edge = |edge: Edge| move_text(edge.origin(), edge.far());
    let name = |player: Player| if player == Player::Blue { "Blue" } else { "Red" };
    let turn = actions.div_ceil(2);
    let mover = if turn % 2 == 0 { Player::Blue } else { Player::Red };
    let safe_edges_allowed = if actions >= TOTAL_ACTIONS { 0 } else { turn.min(2) };
    match reason {
        InvalidPosition::TooManyActions => format!("A game has {TOTAL_ACTIONS} actions."),
        InvalidPosition::TooManyEdges(player) => {
            format!("{} has more edges than {actions} actions allow.", name(player))
        }
        InvalidPosition::TooManyNodes(player) => {
            format!("{} has more nodes than {actions} actions allow.", name(player))
        }
        InvalidPosition::ScoreOutOfRange(player) => format!("{}'s score is too high.", name(player)),
        InvalidPosition::DuplicateEdge(duplicate) => format!("{} appears twice.", edge(duplicate)),
        InvalidPosition::OpposingEdgesTouch(blue, red) => format!("Blue {} touches Red {}.", edge(blue), edge(red)),
        InvalidPosition::OwnEdgesOverlap(a, b) => format!("{} overlaps {}.", edge(a), edge(b)),
        InvalidPosition::NodeInsideOwnEdge(point, inside) => format!("{} lies on {}.", square(point), edge(inside)),
        InvalidPosition::BadShieldedEdge => match safe_edges_allowed {
            0 => format!("{} can have no safe edges yet.", name(mover.opponent())),
            1 => format!("{} can have at most 1 safe edge.", name(mover.opponent())),
            n => format!("{} can have at most {n} safe edges.", name(mover.opponent())),
        },
        InvalidPosition::BadFreshEdge => format!("{} can mark only one edge, and only mid-turn.", name(mover)),
    }
}
