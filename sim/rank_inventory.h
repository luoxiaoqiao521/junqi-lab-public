#pragma once
#include "replay.h"

namespace jq {

// Rank copies are individual inventory slots. Living and removed pieces both
// occupy one slot. Several slots of the same rank remain interchangeable.
struct RankAssignment {
  std::array<unsigned,kPiecesPerSeat> allowed{}, slots{};
  std::array<int,kPiecesPerSeat> order{}, piece_for_slot{}, slot_rank{};

  RankAssignment() {
    slots.fill((1u<<kPiecesPerSeat)-1);
    for (int rank=Flag,slot=0;rank<RankCount;++rank)
      for (int count=0;count<kRankCount[rank];++count,++slot) {
        order[slot]=slot;
        slot_rank[slot]=rank;
      }
  }

  bool match(size_t pieces) {
    if (pieces>allowed.size()) return false;
    piece_for_slot.fill(kNone);
    for (size_t piece=0;piece<pieces;++piece) {
      unsigned visited=0;
      if (!add(static_cast<int>(piece),visited)) return false;
    }
    return true;
  }

  std::array<unsigned,kPiecesPerSeat> supported_ranks() const {
    std::array<unsigned,kPiecesPerSeat> paths{}, supports{};
    unsigned free=0;
    for (int a=0;a<kPiecesPerSeat;++a) {
      const int piece=piece_for_slot[a];
      if (piece==kNone) { free|=1u<<a; continue; }
      for (int b=0;b<kPiecesPerSeat;++b)
        if ((slots[piece]&(1u<<b)) && (allowed[piece]&(1u<<slot_rank[b]))) paths[a]|=1u<<b;
    }
    const auto direct=paths;
    for (int via=0;via<kPiecesPerSeat;++via)
      for (auto& path:paths)
        if (path&(1u<<via)) path|=paths[via];
    // Another slot can take this piece if the displaced chain reaches
    // a free slot, or returns to the piece's original slot.
    for (int a=0;a<kPiecesPerSeat;++a) if (piece_for_slot[a]!=kNone)
      for (int b=0;b<kPiecesPerSeat;++b)
        if ((direct[a]&(1u<<b)) && ((paths[b]|(1u<<b))&(free|(1u<<a))))
          supports[piece_for_slot[a]]|=1u<<slot_rank[b];
    return supports;
  }

 private:
  bool add(int piece,unsigned& visited) {
    for (int slot:order) {
      const unsigned bit=1u<<slot;
      if ((visited&bit) || !(slots[piece]&bit) || !(allowed[piece]&(1u<<slot_rank[slot]))) continue;
      visited|=bit;
      if (piece_for_slot[slot]==kNone || add(piece_for_slot[slot],visited)) {
        piece_for_slot[slot]=piece;
        return true;
      }
    }
    return false;
  }
};

inline unsigned compatible_rank_mask(const Observation& view,const OpeningMarks& marks,int node) {
  const int seat=view.seat_at(node), rank=view.rank_at(node);
  if (seat<0) return 0;
  const auto unavailable=marks.removed[seat].exhausted |
    ((view.revealed_flags&(1u<<seat)) ? 1u<<Marshal : 0u);
  return (rank==Unknown ? marks.ranks[node] : 1u<<rank) & ~unavailable;
}

struct SeatRankInventory {
  RankAssignment matching;
  std::array<int,kPiecesPerSeat> nodes{};
  size_t live=0, lost=0;

  SeatRankInventory(const Observation& view,const OpeningMarks& marks,int seat) {
    expect_value(seat>=0 && seat<kSeats,"invalid rank inventory seat");
    for (int node=0;node<kNodes;++node) if (view.seat_at(node)==seat) {
      expect_value(live<nodes.size(),"too many live pieces in rank inventory");
      nodes[live]=node;
      matching.allowed[live++]=compatible_rank_mask(view,marks,node);
    }
    const auto& removed=marks.removed[seat];
    lost=removed.size;
    expect_value(live+lost<=matching.allowed.size(),"live and lost pieces exceed inventory");
    std::copy_n(removed.masks.begin(),lost,matching.allowed.begin()+live);
  }
};

inline void constrain_rank_inventory(const Observation& view,OpeningMarks& marks,int seat) {
  SeatRankInventory inventory(view,marks,seat);
  if (!inventory.live) return;
  expect_value(inventory.matching.match(inventory.live+inventory.lost),
               "rank inventory conflicts with collision history");
  const auto supported=inventory.matching.supported_ranks();
  for (size_t i=0;i<inventory.live;++i) marks.ranks[inventory.nodes[i]]=supported[i];
  RemovedRanks proven;
  for (size_t i=0;i<inventory.lost;++i) proven.add(supported[inventory.live+i]);
  marks.removed[seat]=proven;
}

struct RankCountRange { int minimum=0, maximum=0; };
struct RankStock { RankCountRange on_board, collision_lost; };

// Look for each attainable count, including zero. Missing observations occupy
// unconstrained slots outside both tracked groups, never guessed casualties.
inline RankCountRange category_rank_count(const SeatRankInventory& inventory,int rank,size_t first,size_t last) {
  expect_value(rank>=Flag && rank<RankCount,"invalid stock rank");
  unsigned target=0;
  for (int slot=0;slot<kPiecesPerSeat;++slot)
    if (inventory.matching.slot_rank[slot]==rank) target|=1u<<slot;
  auto filled=inventory.matching;
  std::fill(filled.allowed.begin()+inventory.live+inventory.lost,filled.allowed.end(),all_evidence_ranks);
  RankCountRange result{kRankCount[rank]+1,-1};
  unsigned for_category=0;
  for (int count=0;count<=kRankCount[rank];++count) {
    auto probe=filled;
    for (size_t piece=0;piece<probe.slots.size();++piece)
      probe.slots[piece]&=~(piece>=first && piece<last?target^for_category:for_category);
    if (probe.match(probe.slots.size())) {
      result.minimum=std::min(result.minimum,count);
      result.maximum=count;
    }
    const unsigned remaining=target^for_category;
    for_category|=remaining&(~remaining+1);
  }
  expect_value(result.maximum>=0,"rank inventory has no feasible counts");
  return result;
}

inline RankStock rank_stock(const Observation& view,const OpeningMarks& marks,int seat,int rank) {
  const SeatRankInventory inventory(view,marks,seat);
  return {category_rank_count(inventory,rank,0,inventory.live),
          category_rank_count(inventory,rank,inventory.live,inventory.live+inventory.lost)};
}

}
