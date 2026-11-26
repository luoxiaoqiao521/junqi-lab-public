#pragma once
#include "exchange_counterexamples.h"
#include "transfer_probe.h"

namespace jq {
inline std::optional<FlagThreat> urgent_public_guard(const SeatDecision& d,const MovedSquares& moved,
                                                    const Rules& rules) {
  const auto offer=public_flag_threats(d,moved,rules)[0];
  if (!offer) return std::nullopt;
  const auto& view=d.observation;
  const int a=sole_enemy_hq(view,(view.viewer+1)%4,rules);
  const int b=sole_enemy_hq(view,(view.viewer+3)%4,rules);
  for (const auto action : d.actions)
    if (view.rank_at(action.to)==Flag || action.to==a || action.to==b)
      return std::nullopt;
  const ExchangeOffer pair{offer->flag,offer->threat,offer->intercept,offer->advance};
  return guarantees_interception(view,pair,rules) ? offer : std::nullopt;
}

struct GuardTransferResult {
  int score=-1, opportunity=-1, changed=0, plies=0, decisions=0;
};

inline GuardTransferResult guard_transfer_trial(State state,const Rules& rules,const ShortWeights& weights,
                                               uint64_t seed,int learner,int variant,bool enabled) {
  auto streams=review_streams(seed);
  MovedSquares moved{};
  GuardTransferResult result;
  const int next_enemy=(learner+(rules.ccw ? 1 : 3))%4;
  const FrozenSeat contact{"contact",{}};
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const int seat=d.observation.viewer;
    auto& rng=streams[seat];
    Move action=variant==1 && seat==next_enemy ? frozen_seat_move(d,contact,rng)
                                               : review_move(d,learner,0,true,weights,rng);
    if (seat==learner) {
      ++result.decisions;
      if (result.opportunity<0) {
        if (const auto guard=urgent_public_guard(d,moved,rules)) {
          result.opportunity=state.ply;
          if (enabled && !(action==guard->intercept)) {
            action=guard->intercept;
            result.changed=1;
          }
        }
      }
    }
    const auto event=step(state,action,rules);
    observe_movement(moved,event,observe(state,learner));
  }
  result.score=review_outcome(state.result,learner);
  result.plies=state.ply;
  return result;
}

inline void flag_guard_transfer(const Rules& rules,uint64_t seed,int learner,int variant,
                                const ShortWeights& weights) {
  expect_value(variant==0 || variant==1,"unsupported guard transfer opponent");
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  const auto base=guard_transfer_trial(initial,rules,weights,seed,learner,variant,false);
  const auto candidate=guard_transfer_trial(initial,rules,weights,seed,learner,variant,true);
  expect_value(base.opportunity==candidate.opportunity,"guard policies diverged before intervention");
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant
            << ",\"first\":" << first << ",\"rows\":[";
  const auto print=[](const GuardTransferResult& result) {
    std::cout << '[' << result.score << ',' << result.opportunity << ',' << result.changed
              << ',' << result.plies << ',' << result.decisions << ']';
  };
  print(base); std::cout << ','; print(candidate); std::cout << "]}\n";
}
}
