#pragma once
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "../observe/filter.h"
#include "../core/json.h"
#include "settlement.h"
namespace jq {
inline std::string read_text(const std::string& file) {
  std::ifstream in(file, std::ios::binary); if (!in) throw std::runtime_error("cannot open " + file);
  std::string text;
  char buffer[8192];
  while (in) {
    in.read(buffer, sizeof(buffer));
    const auto count = static_cast<size_t>(in.gcount());
    if (text.size() + count > 64000000) throw std::runtime_error("file too large");
    text.append(buffer, count);
  }
  if (!in.eof()) throw std::runtime_error("file read failed");
  return text;
}
template<class T> void array_json(std::ostream& out, const T* data, int n) {
  out << '['; for (int i = 0; i < n; ++i) { if (i) out << ','; out << int(data[i]); } out << ']';
}
inline void board_json(std::ostream& out) {
  const auto& b = Board::get(); out << "\"nodes\":[";
  for (int n = 0; n < kNodes; ++n) {
    if (n) out << ','; const auto& p = b.node(n);
    out << '[' << int(p.gc) << ',' << int(p.gr) << ',' << int(p.kind) << ']';
  }
  out << "],\"roads\":["; bool first = true;
  for (int n = 0; n < kNodes; ++n) for (int m : b.road(n)) if (n < m) {
    if (!first) out << ','; first = false; out << '[' << n << ',' << m << ']';
  }
  out << "],\"rails\":["; first = true;
  for (int n = 0; n < kNodes; ++n) for (int m : b.rail_adj(n)) if (n < m) {
    if (!first) out << ','; first = false; out << '[' << n << ',' << m << ']';
  }
  out << ']';
}
inline void settlement_json(std::ostream& out, const StepInfo& event, int ply) {
  out << ",\"eliminations\":[";
  bool comma=false;
  for (int seat=0;seat<kSeats;++seat) {
    const auto reason=event.exit_reason[seat];
    if (reason==ExitReason::None) continue;
    if (comma) out << ',';
    comma=true;
    out << "{\"seat\":" << seat << ",\"ply\":" << ply
        << ",\"reason\":\"" << exit_names[int(reason)] << "\"}";
  }
  out << "],\"auto_skipped_mask\":" << int(event.skipped_mask)
      << ",\"end_reason\":\"" << end_names[int(event.end_reason)] << '"';
}

inline std::string frame_json(const State& s, const StepInfo* event,
                              IdentityVisibility visibility = IdentityVisibility::OwnAndPublic,
                              bool settlement = false, bool review = false) {
  std::ostringstream out;
  out << "{\"type\":\"frame\",\"seq\":" << s.ply << ",\"turn\":" << int(s.to_move)
      << ",\"result\":" << int(s.result) << ",\"no_progress\":" << s.no_progress
      << ",\"hash\":\"" << state_hash(s) << "\",\"seats\":";
  array_json(out, s.seat_at, kNodes); out << ",\"truth\":"; array_json(out, s.rank_at, kNodes);
  out << ",\"out\":"; array_json(out, s.out, kSeats); out << ",\"views\":[";
  for (int v = 0; v < kSeats; ++v) {
    if (v) out << ','; auto o = observe(s, v,visibility); array_json(out, o.ranks, kNodes);
  }
  out << "],\"legal_move_counts\":[";
  int origins[kSeats]{};
  std::vector<std::array<int,3>> flag_routes;
  for (int v = 0; v < kSeats; ++v) {
    if (v) out << ','; auto o = observe(s, v,visibility); std::vector<Move> moves;
    if (!s.out[v] && s.result == Result::Ongoing) gen_moves(o, v, moves);
    out << moves.size();
    if (!review) continue;
    int previous = kNone;
    for (const auto move:moves) {
      if (move.from != previous) ++origins[v];
      previous = move.from;
      if (s.rank_at[move.to] == Flag)
        flag_routes.push_back({v,move.from,move.to});
    }
  }
  if (review) {
    out << "],\"move_origins\":"; array_json(out,origins,kSeats);
    out << ",\"flag_routes\":[";
    for (size_t i=0;i<flag_routes.size();++i) {
      if (i) out << ',';
      array_json(out,flag_routes[i].data(),3);
    }
  }
  out << "],\"event\":";
  if (!event) out << "null";
  else out << '[' << int(event->seat) << ',' << int(event->move.from) << ',' << int(event->move.to)
           << ',' << int(event->outcome) << ',' << int(event->eliminated_mask) << ',' << int(event->revealed_mask) << ']';
  if (event && settlement) settlement_json(out,*event,s.ply);
  out << '}'; return out.str();
}
}
