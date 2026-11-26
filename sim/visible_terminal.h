#pragma once
#include <algorithm>
#include <iterator>
#include <optional>
#include "../observe/filter.h"

namespace jq {
// Rebuild an equivalent live position only when every occupied rank is visible.
// Local IDs are assigned from the observation and need not match referee history.
inline std::optional<State> visible_terminal_board(const Observation& view) {
  State board{};
  board.match=view.match;
  board.to_move=static_cast<int8_t>(view.turn);
  board.ply=view.ply;
  board.no_progress=view.quiet_clock;
  std::fill(std::begin(board.seat_at),std::end(board.seat_at),int8_t(kNone));
  std::fill(std::begin(board.id_at),std::end(board.id_at),int8_t(kNone));
  std::fill(std::begin(board.rank_at),std::end(board.rank_at),int8_t(Unknown));
  std::fill(std::begin(board.rank_of),std::end(board.rank_of),int8_t(Unknown));
  std::fill(std::begin(board.out),std::end(board.out),true);
  int count[kSeats]{};
  for (int n=0;n<kNodes;++n) {
    const int seat=view.seat_at(n), rank=view.rank_at(n);
    if (seat==kNone) continue;
    if (!participates(view.match,seat) || rank<Flag || rank>=RankCount || count[seat]==kPiecesPerSeat) return {};
    const int id=seat*kPiecesPerSeat+count[seat]++;
    board.seat_at[n]=static_cast<int8_t>(seat);
    board.id_at[n]=static_cast<int8_t>(id);
    board.rank_at[n]=board.rank_of[id]=static_cast<int8_t>(rank);
    board.node_of[id]=static_cast<uint8_t>(n);
    board.alive[id]=true;
    board.out[seat]=false;
    board.flag_revealed[seat]=(view.revealed_flags&(1u<<seat))!=0;
  }
  if (!participates(view.match,view.turn) || board.out[view.turn]) return {};
  return board;
}

inline std::optional<Result> visible_result(const std::optional<State>& root, Move move, const Rules& rules) {
  if (!root) return {};
  auto branch=*root;
  step(branch,move,rules);
  return branch.result;
}
}
