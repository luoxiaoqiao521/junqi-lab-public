#include "layout.h"
#include "board.h"
#include <algorithm>
#include <cassert>

namespace jq {
int layout_slot_node(int seat, int slot) {
  if (seat < 0 || seat >= kSeats || slot < 0 || slot >= kPiecesPerSeat) return kNone;
  const auto& b = Board::get();
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 5; ++c) {
      int n = b.seat_node(seat, r, c);
      if (b.node(n).kind != NodeKind::Camp && slot-- == 0) return n;
    }
  return kNone;
}

bool deployment_rank_allowed(int node, int rank) {
  if (node < 0 || node >= kNodes || rank < Flag || rank >= RankCount) return false;
  const auto& n = Board::get().node(node);
  if (n.r < 0 || n.kind == NodeKind::Camp) return false;
  if (rank == Flag) return n.kind == NodeKind::HQ;
  if (rank == Mine) return n.r >= 4;
  return rank != Bomb || n.r != 0;
}

bool valid_layout(const int8_t (&layout)[kPiecesPerSeat]) {
  int counts[RankCount] = {};
  for (int k = 0; k < kPiecesPerSeat; ++k) {
    int rank = layout[k];
    if (rank < Flag || rank >= RankCount) return false;
    ++counts[rank];
    if (!deployment_rank_allowed(layout_slot_node(0, k), rank)) return false;
  }
  return std::equal(counts, counts + RankCount, kRankCount);
}

void generate_layout(int8_t (&layout)[kPiecesPerSeat], Rng& rng) {
  std::fill(std::begin(layout), std::end(layout), int8_t(Unknown));
  // Sequential constrained placement is reproducible, not uniform over layouts.
  for (int rank : {Flag, Mine, Bomb}) {
    for (int j = 0; j < kRankCount[rank]; ++j) {
      int candidates[kPiecesPerSeat], count = 0;
      for (int k = 0; k < kPiecesPerSeat; ++k) {
        if (layout[k] == Unknown && deployment_rank_allowed(layout_slot_node(0, k), rank))
          candidates[count++] = k;
      }
      assert(count > 0);
      layout[candidates[rng.below(count)]] = static_cast<int8_t>(rank);
    }
  }
  int8_t rest[kPiecesPerSeat];
  int count = 0;
  for (int rank = Engineer; rank < RankCount; ++rank)
    for (int j = 0; j < kRankCount[rank]; ++j) rest[count++] = static_cast<int8_t>(rank);
  for (int i = count - 1; i > 0; --i) std::swap(rest[i], rest[rng.below(i + 1)]);
  int index = 0;
  for (auto& rank : layout) if (rank == Unknown) rank = rest[index++];
}
}
