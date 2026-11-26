#pragma once
#include "../observe/observation.h"
#include "../core/moves.h"
#include "../core/rng.h"
namespace jq {
inline bool random_move(const Observation& o, Rng& rng, Move& move) {
  if (o.viewer != o.turn) return false;
  std::vector<Move> moves; gen_moves(o, o.viewer, moves);
  if (moves.empty()) return false;
  move = moves[rng.below(static_cast<uint32_t>(moves.size()))]; return true;
}
}
