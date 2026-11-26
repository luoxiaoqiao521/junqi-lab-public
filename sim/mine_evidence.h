#pragma once
#include "decision_branches.h"

namespace jq {
inline std::array<int,4> opening_mine_neighbors(const State& opening) {
  std::array<int,4> counts{};
  for (int flag=0; flag<kNodes; ++flag)
    if (opening.rank_at[flag]==Flag)
      for (const int neighbor : Board::get().road(flag))
        if (opening.seat_at[neighbor]==opening.seat_at[flag] && opening.rank_at[neighbor]==Mine)
          ++counts[opening.seat_at[flag]];
  return counts;
}

struct MineContacts {
  int decisions=0, changed_decisions=0;
  std::array<std::array<int,4>,3> by_attacker{};
};

inline void count_public_mine_contacts(const State& state, const SeatDecision& decision,
                                       const OpeningMarks& marks, const Rules& rules,
                                       MineContacts& counts) {
  const auto& view=decision.observation;
  ++counts.decisions;
  bool changed=false;
  for (const auto action : decision.actions) {
    const int owner=view.seat_at(action.to);
    if (owner<0 || view.rank_at(action.to)!=Unknown || owner!=Board::get().node(action.to).seat ||
        !deployment_rank_allowed(action.to,Mine)) continue;
    const int attacker=view.rank_at(action.from);
    auto& row=counts.by_attacker[attacker==Engineer ? 0 : attacker==Bomb ? 2 : 1];
    const unsigned possible=marks.ranks[action.to];
    // Research verification only. The public mask is already fixed at this point.
    expect_value(possible & (1u << state.rank_at[action.to]),"public exclusions removed a compatible rank");
    ++row[0];
    row[1]+=bool(possible & (1u << Mine));
    row[2]+=marks.moved[action.to];
    if (!marks.moved[action.to]) continue;
    expect_value(!(possible & ((1u << Mine) | (1u << Flag))),"moved piece still has an immobile rank");
    if (collision_outcome_mask(attacker,possible,rules)!=collision_outcome_mask(attacker,possible | (1u << Mine),rules)) {
      ++row[3];
      changed=true;
    }
  }
  counts.changed_decisions+=changed;
}

inline void mine_evidence_probe(const Rules& rules,uint64_t seed,int learner,int variant,
                                const ShortWeights& weights) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State state;
  init_state(state,layouts,first,rules);
  const auto formations=opening_mine_neighbors(state);
  auto streams=review_streams(seed);
  OpeningMarks marks(observe(state,learner),true);
  MineContacts counts;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner) count_public_mine_contacts(state,d,marks,rules,counts);
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    step_with_public_ranks(state,rules,learner,action,&marks);
  }
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant
            << ",\"score\":" << review_outcome(state.result,learner) << ",\"decisions\":" << counts.decisions
            << ",\"changed_decisions\":" << counts.changed_decisions << ",\"formations\":[";
  for (size_t index=0; index<formations.size(); ++index)
    std::cout << (index ? "," : "") << formations[index];
  std::cout << "],\"contacts\":[";
  for (size_t index=0; index<counts.by_attacker.size(); ++index) {
    const auto& row=counts.by_attacker[index];
    std::cout << (index ? ",[" : "[") << row[0] << ',' << row[1] << ',' << row[2] << ',' << row[3] << ']';
  }
  std::cout << "]}\n";
}
}
