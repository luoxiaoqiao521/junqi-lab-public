#pragma once
#include <iostream>
#include "config.h"
#include "log.h"
#include "seat_episode.h"
#include "../core/layout.h"
#include "../core/rng.h"

namespace jq {
inline void sample_short_branches(const State& start, const Rules& rules, uint64_t seed) {
  const auto decision = seat_decision(start);
  const int learner = decision.observation.viewer;
  const auto& board = Board::get();
  std::cout << "{\"seed\":" << seed << ",\"actions\":[";
  bool first = true;
  for (const Move action : decision.actions) {
    State branch = start;
    step(branch, action, rules);
    Rng rng(mix_seed(seed, 2));
    while (branch.result == Result::Ongoing) {
      const auto next = seat_decision(branch);
      step(branch, next.actions.at(rng.below(static_cast<uint32_t>(next.actions.size()))), rules);
    }
    const auto score = team_score(branch.result, learner);
    const auto& o = decision.observation;
    if (!first) std::cout << ',';
    first = false;
    std::cout << '[' << int(action.from) << ',' << int(action.to) << ','
              << (o.seat_at(action.to) >= 0) << ','
              << (board.node(action.to).kind == NodeKind::HQ) << ','
              << (o.rank_at(action.to) == Flag) << ',' << int(o.rank_at(action.from)) << ',';
    if (score) std::cout << *score;
    else std::cout << "null";
    std::cout << ',' << (branch.ply - start.ply) << ',' << int(branch.out[learner]) << ']';
  }
  std::cout << "]}\n";
}

// Use a legal forward prefix. The final flag capture selects the training
// distribution on the referee side. No label enters the player's features.
inline void learning_probe(const Rules& rules, uint64_t first_seed, uint64_t count) {
  for (uint64_t offset = 0; offset < count; ++offset) {
    const uint64_t seed = first_seed + offset;
    Rng layout_rng(seed), play_rng(mix_seed(seed, 1));
    int8_t layouts[kSeats][kPiecesPerSeat];
    for (auto& layout : layouts) generate_layout(layout, layout_rng);
    State s;
    const int first = rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
    init_state(s, layouts, first, rules);
    while (s.result == Result::Ongoing) {
      const auto decision = seat_decision(s);
      const Move action = decision.actions.at(play_rng.below(static_cast<uint32_t>(decision.actions.size())));
      const State before = s;
      const bool attacks_flag = before.rank_at[action.to] == Flag;
      step(s, action, rules);
      const bool wins = s.result == Result::Team0Wins || s.result == Result::Team1Wins;
      if (attacks_flag && wins && decision.actions.size() > 1 &&
          team_score(s.result, before.to_move) == 1.0) sample_short_branches(before, rules, seed);
    }
  }
}
}
