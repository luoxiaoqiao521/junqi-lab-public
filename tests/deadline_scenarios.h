#pragma once

template<class List> static void shuffle_slots(List& slots,jq::Rng& rng) {
  for (size_t i=slots.size();i>1;--i)
    std::swap(slots[i-1],slots[rng.below(static_cast<uint32_t>(i))]);
}

static bool arrange_seat(jq::State& sample,const jq::Observation& view,const jq::OpeningMarks& marks,
                         int seat,jq::Rng& rng) {
  using namespace jq;
  std::vector<int> nodes,flags;
  const auto& board=Board::get();
  for (int n=0;n<kNodes;++n) if (view.seat_at(n)==seat) {
    if (board.node(n).seat==seat && board.node(n).kind==NodeKind::HQ &&
        (compatible_rank_mask(view,marks,n)&(1u<<Flag))) flags.push_back(n);
    nodes.push_back(n);
  }
  sample.out[seat]=nodes.empty();
  if (nodes.empty()) return true;
  const auto& removed=marks.removed[seat];
  if (flags.empty() || nodes.size()+removed.size>kPiecesPerSeat) return false;
  shuffle_slots(nodes,rng);
  shuffle_slots(flags,rng);
  RankAssignment match;
  std::copy_n(removed.masks.begin(),removed.size,match.allowed.begin()+nodes.size());
  shuffle_slots(match.order,rng);
  for (const int flag : flags) {
    for (size_t i=0;i<nodes.size();++i) {
      const auto allowed=compatible_rank_mask(view,marks,nodes[i]);
      match.allowed[i]=nodes[i]==flag?(1u<<Flag):allowed&~(1u<<Flag);
    }
    if (!match.match(nodes.size()+removed.size)) continue;
    for (int slot=0;slot<kPiecesPerSeat;++slot) {
      const int i=match.piece_for_slot[slot];
      if (i<0 || static_cast<size_t>(i)>=nodes.size()) continue;
      const int n=nodes[i], id=seat*kPiecesPerSeat+slot;
      sample.seat_at[n]=static_cast<int8_t>(seat);
      sample.id_at[n]=static_cast<int8_t>(id);
      sample.rank_at[n]=sample.rank_of[id]=static_cast<int8_t>(match.slot_rank[slot]);
      sample.node_of[id]=static_cast<uint8_t>(n);
      sample.alive[id]=true;
    }
    return true;
  }
  return false;
}

static std::optional<jq::State> public_sample(const jq::Observation& view,const jq::OpeningMarks& marks,
                                              uint64_t seed) {
  using namespace jq;
  Rng rng(seed);
  State sample{};
  std::fill(std::begin(sample.id_at),std::end(sample.id_at),static_cast<int8_t>(kNone));
  std::fill(std::begin(sample.seat_at),std::end(sample.seat_at),static_cast<int8_t>(kNone));
  std::fill(std::begin(sample.rank_at),std::end(sample.rank_at),Unknown);
  sample.ply=view.ply;
  sample.no_progress=view.quiet_clock;
  sample.to_move=static_cast<int8_t>(view.turn);
  sample.match=view.match;
  for (int seat=0;seat<kSeats;++seat) {
    sample.flag_revealed[seat]=(view.revealed_flags&(1u<<seat))!=0;
    if (!arrange_seat(sample,view,marks,seat,rng)) return std::nullopt;
  }
  if (!same_observation(view,observe(sample,view.viewer))) return std::nullopt;
  for (int seat=0;seat<kSeats;++seat) if (!sample.out[seat]) {
    bool mobile=false;
    for (int n=0;n<kNodes;++n)
      if (sample.seat_at[n]==seat && can_leave_node(sample.rank_at[n],n)) mobile=true;
    if (!mobile) return std::nullopt;
  }
  return sample;
}
