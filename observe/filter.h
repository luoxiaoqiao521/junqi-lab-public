#pragma once
#include "observation.h"
#include "../core/state.h"
namespace jq {
enum class IdentityVisibility { OwnAndPublic, AllyAndPublic, AllIdentities };

inline Observation observe(const State& s, int viewer,
                           IdentityVisibility mode=IdentityVisibility::OwnAndPublic) {
  Observation o; o.viewer = viewer; o.turn = s.to_move; o.ply = s.ply; o.match = s.match;
  o.quiet_clock = s.no_progress;
  for (int seat = 0; seat < kSeats; ++seat)
    if (s.flag_revealed[seat]) o.revealed_flags |= static_cast<uint8_t>(1u << seat);
  for (int n = 0; n < kNodes; ++n) {
    int owner = s.seat_at[n]; o.seats[n] = s.seat_at[n]; o.ranks[n] = Unknown;
    if (owner < 0) continue;
    const bool partner_visible = mode == IdentityVisibility::AllyAndPublic && o.team(owner) == o.team(viewer);
    if (owner == viewer || partner_visible || mode == IdentityVisibility::AllIdentities ||
        (s.flag_revealed[owner] && s.rank_at[n] == Flag))
      o.ranks[n] = s.rank_at[n];
  }
  return o;
}
}
