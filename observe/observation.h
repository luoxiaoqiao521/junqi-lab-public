#pragma once
#include <algorithm>
#include "../core/types.h"
namespace jq {
struct Observation {
  int viewer = 0, turn = 0, ply = 0;
  int quiet_clock = 0;
  int8_t seats[kNodes], ranks[kNodes];
  MatchKind match = MatchKind::FourSeats;
  uint8_t revealed_flags = 0;
  int seat_at(int n) const { return seats[n]; }
  int rank_at(int n) const { return ranks[n]; }
  int team(int seat) const { return team_of(seat, match); }
};

inline bool same_observation(const Observation& a, const Observation& b) {
  return a.viewer == b.viewer && a.turn == b.turn && a.ply == b.ply &&
    a.quiet_clock == b.quiet_clock && a.match == b.match && a.revealed_flags == b.revealed_flags &&
    std::equal(a.seats, a.seats + kNodes, b.seats) && std::equal(a.ranks, a.ranks + kNodes, b.ranks);
}
}
