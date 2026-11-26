#pragma once
#include "decision_branches.h"
#include "transfer_probe.h"

namespace jq {
struct OpeningBranch {
  Result result;
  int plies;
  std::optional<Move> intervening, partner;
};

inline void print_opening_move(const std::optional<Move>& action) {
  if (action) std::cout << '[' << int(action->from) << ',' << int(action->to) << ']';
  else std::cout << "null";
}

inline OpeningBranch finish_opening_branch(State state, const Rules& rules, int learner,
                                            const std::array<FrozenSeat,4>& profiles,
                                            std::array<Rng,4> streams, Move first) {
  step(state,first,rules);
  OpeningBranch branch{};
  bool opening_window=true;
  const int intervening=(learner+(rules.ccw ? 1 : 3))%4;
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    if (actor==learner) opening_window=false;
    const auto action=frozen_seat_move(seat_decision(state),profiles[actor],streams[actor]);
    if (opening_window && actor==intervening && !branch.intervening) branch.intervening=action;
    if (opening_window && actor==(learner+2)%4 && !branch.partner) branch.partner=action;
    step(state,action,rules);
  }
  branch.result=state.result;
  branch.plies=state.ply;
  return branch;
}

// A public occupancy proxy. It does not imply the partner can or should use the square.
inline bool vacates_partner_rail(const Observation& view, Move move) {
  if (view.seat_at(move.to)!=kNone) return false;
  for (const auto neighbor : Board::get().rail_adj(move.from))
    if (view.seat_at(neighbor)==(view.viewer+2)%4) return true;
  return false;
}

inline void print_opening_branch(const OpeningBranch& branch, int learner) {
  std::cout << "{\"score\":" << review_outcome(branch.result,learner)
            << ",\"plies\":" << branch.plies << ",\"intervening\":";
  print_opening_move(branch.intervening);
  std::cout << ",\"partner\":";
  print_opening_move(branch.partner);
  std::cout << '}';
}

inline void first_opening_divergence(State state, const Rules& rules, uint64_t seed, int learner,
                                     std::array<FrozenSeat,4> profiles) {
  auto streams=review_streams(seed);
  auto candidate_stream=streams[learner];
  const auto candidate=profiles[learner];
  profiles[learner]=FrozenSeat{"heuristic",{}};
  int turn=0;
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto d=seat_decision(state);
    const auto baseline=frozen_seat_move(d,profiles[actor],streams[actor]);
    if (actor==learner) {
      ++turn;
      const auto proposed=frozen_seat_move(d,candidate,candidate_stream);
      if (!(baseline==proposed)) {
        const auto a=finish_opening_branch(state,rules,learner,profiles,streams,baseline);
        streams[learner]=candidate_stream;
        profiles[learner]=candidate;
        const auto b=finish_opening_branch(state,rules,learner,profiles,streams,proposed);
        std::cout << ",\"split\":{\"ply\":" << state.ply << ",\"turn\":" << turn << ",\"moves\":[";
        print_opening_move(baseline);
        std::cout << ',';
        print_opening_move(proposed);
        std::cout << "],\"vacates_partner_rail\":["
                  << int(vacates_partner_rail(d.observation,baseline)) << ','
                  << int(vacates_partner_rail(d.observation,proposed)) << "]},\"branches\":[";
        print_opening_branch(a,learner);
        std::cout << ',';
        print_opening_branch(b,learner);
        std::cout << "]}\n";
        return;
      }
    }
    step(state,baseline,rules);
  }
  const OpeningBranch shared{state.result,state.ply,{},{}};
  std::cout << ",\"split\":null,\"branches\":[";
  print_opening_branch(shared,learner);
  std::cout << ',';
  print_opening_branch(shared,learner);
  std::cout << "]}\n";
}

inline void partner_divergence(const Rules& rules, uint64_t start, uint64_t count,
                               const std::array<FrozenSeat,4>& roles) {
  for (uint64_t seed=start; seed<start+count; ++seed) {
    Rng layout_rng(seed);
    int8_t layouts[kSeats][kPiecesPerSeat];
    for (auto& layout : layouts) generate_layout(layout,layout_rng);
    const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
    for (int learner=0; learner<kSeats; ++learner) {
      State state;
      init_state(state,layouts,first,rules);
      std::array<FrozenSeat,4> profiles;
      profiles[learner]=roles[0];
      profiles[(learner+2)%4]=roles[1];
      profiles[(learner+(rules.ccw ? 1 : 3))%4]=roles[2];
      profiles[(learner+(rules.ccw ? 3 : 1))%4]=roles[3];
      std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"first\":" << first;
      first_opening_divergence(state,rules,seed,learner,profiles);
    }
  }
}
}
