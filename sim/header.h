#pragma once
#include "log.h"
#include "../core/rng.h"
namespace jq {
inline void write_header(std::ostream& log, uint64_t seed, const Rules& rules,
 const std::string& rt, const std::string& ot, int first,
 const int8_t layouts[kSeats][kPiecesPerSeat], const std::string& policy = "random",
 MatchKind match = MatchKind::FourSeats) {
  log << "{\"type\":\"header\",\"schema\":1,\"engine\":\"junqi-mvp-1\",\"topology\":\"experimental-129-v1\",\"info_level\":\"referee_private\",\"history_complete\":true,\"seed\":\"" << seed
      << "\",\"rules_hash\":\"" << rules_hash(rules, match) << "\",\"observation_hash\":\"" << fnv1a(ot.data(), ot.size())
      << "\",\"first\":" << first << ",\"rules\":" << rt << ",\"observation\":" << ot << ",\"layouts\":[";
  for (int seat = 0; seat < kSeats; ++seat) {
    if (seat) log << ',';
    array_json(log, layouts[seat], participates(match,seat) ? kPiecesPerSeat : 0);
  }
  log << "],"; board_json(log); if (policy != "random") log << ",\"policy\":" << quote_json(policy);
  if (match == MatchKind::SouthNorth) log << ",\"match\":\"single-dark\"";
  if (rules.eliminate_exhausted_movers) log << ",\"settlement_detail\":true";
  log << "}\n";
}
}

