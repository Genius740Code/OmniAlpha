// Game API contract (compile-time, zero-overhead). A game is a struct `Game` in namespace
// `game_<dirname>` exposing the static members checked by GameLike below.
//
// CONVENTIONS (the validator enforces most of these):
//  * Move is an int16 and ALSO the policy index: 0 <= move < Game::kActionCount.
//  * State is a cheap-to-copy value type (MCTS copies it per simulation). No heap, no pointers.
//  * Exactly 2 players, ids 0 and 1. outcome() is from the given player's view: +1 win, 0 draw, -1 loss.
//  * outcome() is zero-sum: outcome(s,0) == -outcome(s,1).
//  * encode() writes from the view of the player to move (planes: "mine", "theirs", ...), layout [plane][h][w].
//  * apply() on an illegal move is undefined behaviour (not tested).
//  * Optional: `static void undo(State&, Move)` (checked by validator if present).
//  * Optional symmetries: kNumSymmetries>1 plus transform_move(Move,int) and symmetry_input(in,out,int).
#pragma once
#include <concepts>
#include <cstdint>
#include <string>

namespace gai {
using Move = int16_t;
using Hash = uint64_t;

template <class G>
concept GameLike = requires(typename G::State s, const typename G::State cs, Move m, Move* mv, float* f,
                            const float* cf) {
  typename G::State;
  { G::kActionCount } -> std::convertible_to<int>;   // size of policy vector
  { G::kMaxMoves } -> std::convertible_to<int>;      // max legal moves in any position
  { G::kMaxGameLength } -> std::convertible_to<int>; // hard upper bound on plies
  { G::kInputPlanes } -> std::convertible_to<int>;
  { G::kInputH } -> std::convertible_to<int>;
  { G::kInputW } -> std::convertible_to<int>;
  { G::kNumSymmetries } -> std::convertible_to<int>; // 1 = none
  { G::initial() } -> std::same_as<typename G::State>;
  { G::legal_moves(cs, mv) } -> std::convertible_to<int>; // writes <= kMaxMoves, returns count
  { G::apply(s, m) } -> std::same_as<void>;
  { G::is_terminal(cs) } -> std::convertible_to<bool>;
  { G::outcome(cs, 0) } -> std::convertible_to<float>;
  { G::current_player(cs) } -> std::convertible_to<int>;
  { G::hash(cs) } -> std::convertible_to<Hash>;
  { G::encode(cs, f) } -> std::same_as<void>;
  { G::transform_move(m, 0) } -> std::convertible_to<Move>;
  { G::symmetry_input(cf, f, 0) } -> std::same_as<void>;
  { G::to_string(cs) } -> std::convertible_to<std::string>;
};

// Reference implementations only need the logic part (no encode/hash/symmetry).
template <class R>
concept RefGameLike = requires(typename R::State s, const typename R::State cs, Move m, Move* mv) {
  typename R::State;
  { R::initial() } -> std::same_as<typename R::State>;
  { R::legal_moves(cs, mv) } -> std::convertible_to<int>;
  { R::apply(s, m) } -> std::same_as<void>;
  { R::is_terminal(cs) } -> std::convertible_to<bool>;
  { R::outcome(cs, 0) } -> std::convertible_to<float>;
  { R::current_player(cs) } -> std::convertible_to<int>;
};

template <class G>
concept HasUndo = requires(typename G::State s, Move m) { G::undo(s, m); };

template <class G> constexpr int input_size() { return G::kInputPlanes * G::kInputH * G::kInputW; }
}  // namespace gai
