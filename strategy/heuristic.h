#pragma once
#include "random.h"
#include "../core/rules.h"
#include <algorithm>
#include <cstdlib>
namespace jq {
// Integer preference components, not probabilities or proven tactical values.
struct MoveScore {
  int objective = 0, position = 0, exposure = 0;
  int total() const { return objective + position + exposure; }
};
inline int unknown_contact_preference(int attacker_rank) { return 22-3*attacker_rank; }
inline int distance(int a, int b) {
  const auto& board = Board::get();
  return std::abs(int(board.node(a).gr)-int(board.node(b).gr)) +
         std::abs(int(board.node(a).gc)-int(board.node(b).gc));
}
inline int known_collision_objective(int attacker,int defender,const Rules& rules) {
  const auto outcome=resolve(attacker,defender,rules);
  if (outcome==Outcome::AttackerWins) return defender==Flag?10000:60;
  if (outcome==Outcome::DefenderWins) return -60;
  return 0;
}

// No rules pointer selects the frozen heuristic-1 preference used by old logs and studies.
inline MoveScore score_move(const Observation& o, Move m, const Rules* rules=nullptr) {
  const auto& board = Board::get();
  MoveScore score;
  const int rank = o.rank_at(m.from), target = o.rank_at(m.to);
  if (o.seat_at(m.to) >= 0) {
    if (target == Unknown) score.exposure = unknown_contact_preference(rank);
    else if (rules) score.objective = known_collision_objective(rank,target,*rules);
    else if (target == Flag) score.objective = 10000;
    else score.objective = target < rank ? 60 : -60;
  }
  int before = 100, after = 100;
  for (int seat = 0; seat < kSeats; ++seat) {
    if (!participates(o.match, seat) || o.team(seat) == o.team(o.viewer)) continue;
    for (int col : {1,3}) {
      const int goal = board.seat_node(seat,5,col);
      before = std::min(before,distance(m.from,goal));
      after = std::min(after,distance(m.to,goal));
    }
  }
  score.position = 3 * (before-after);
  if (board.node(m.to).kind == NodeKind::Camp) score.position += 3;
  if (board.node(m.from).kind == NodeKind::Camp) score.position -= 3;
  if (board.node(m.to).kind == NodeKind::HQ && target != Flag) score.position -= 80;
  return score;
}
template <class Score>
bool preferred_move(const std::vector<Move>& moves, Rng& rng, Move& selected, Score&& score) {
  if (moves.empty()) return false;
  int best = -100000, ties = 0;
  for (auto move : moves) {
    const int value = score(move);
    if (value > best) { best=value; ties=1; selected=move; }
    else if (value == best && rng.below(++ties)==0) selected=move;
  }
  return true;
}
inline bool heuristic_move(const Observation& o, Rng& rng, Move& selected) {
  if (o.viewer != o.turn) return false;
  std::vector<Move> moves;
  gen_moves(o,o.viewer,moves);
  return preferred_move(moves,rng,selected,[&](Move m){return score_move(o,m).total();});
}
}
