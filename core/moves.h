#pragma once
#include "board.h"
namespace jq {
inline bool can_leave_node(int rank,int node) {
  return rank!=Flag && rank!=Mine && Board::get().node(node).kind!=NodeKind::HQ;
}

template <class View>
bool can_enter(const View& v, int seat, int n) {
  int o = v.seat_at(n);
  if (o == kNone) return true;
  if (v.team(o) == v.team(seat)) return false;
  return Board::get().node(n).kind != NodeKind::Camp;
}

template <class View>
void gen_piece_moves(const View& v, int seat, int from, std::vector<Move>& out) {
  const Board& b = Board::get();
  int rank = v.rank_at(from);
  if (!can_leave_node(rank,from)) return;
  bool seen[kNodes] = {};
  auto add = [&](int to) {
    if (!seen[to]) { seen[to] = true; out.push_back(Move{uint8_t(from), uint8_t(to)}); }
  };
  for (int n : b.road(from)) if (can_enter(v, seat, n)) add(n);
  if (!b.node(from).rail) return;
  if (rank == Engineer) {
    uint8_t stack[kNodes];
    bool visited[kNodes] = {};
    int top = 0;
    stack[top++] = static_cast<uint8_t>(from);
    visited[from] = true;
    while (top) {
      int cur = stack[--top];
      for (int n : b.rail_adj(cur)) {
        if (visited[n]) continue;
        visited[n] = true;
        if (v.seat_at(n) == kNone) { add(n); stack[top++] = static_cast<uint8_t>(n); }
        else if (can_enter(v, seat, n)) add(n);
      }
    }
    return;
  }
  for (const RailRef& ref : b.rail_refs(from)) {
    const std::vector<uint8_t>& ln = b.line(ref.line);
    for (int dir : {-1, 1}) {
      for (int i = ref.index + dir; i >= 0 && i < static_cast<int>(ln.size()); i += dir) {
        int n = ln[i];
        if (v.seat_at(n) == kNone) { add(n); continue; }
        if (can_enter(v, seat, n)) add(n);
        break;
      }
    }
  }
}

template <class View>
void gen_moves(const View& v, int seat, std::vector<Move>& out) {
  out.clear();
  for (int n = 0; n < kNodes; ++n)
    if (v.seat_at(n) == seat) gen_piece_moves(v, seat, n, out);
}

}
