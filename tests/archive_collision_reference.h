#pragma once
#include "../sim/decision_branches.h"

namespace jq::archive_probe {

inline void verify_collision_objective() {
  Rules rules;
  expect_value(known_collision_objective(Marshal,Mine,rules)==-60,"marshal loses against mine");
  expect_value(known_collision_objective(Engineer,Mine,rules)==60,"engineer clears mine");
  expect_value(known_collision_objective(Marshal,Bomb,rules)==0 &&
               known_collision_objective(Bomb,Marshal,rules)==0 &&
               known_collision_objective(Engineer,Engineer,rules)==0,"mutual removals");
  expect_value(known_collision_objective(Marshal,Flag,rules)==10000 &&
               known_collision_objective(Bomb,Flag,rules)==0,"flag under archive rules");
  rules.bomb_captures_flag=true;
  rules.bomb_mine_both_removed=false;
  rules.mine_survives=false;
  expect_value(known_collision_objective(Bomb,Flag,rules)==10000 &&
               known_collision_objective(Bomb,Mine,rules)==-60 &&
               known_collision_objective(Marshal,Mine,rules)==0,"alternate configured collisions");
}

inline int collision_score(const Observation& observation, Move move, const Rules& rules) {
  return score_move(observation,move,&rules).total();
}

inline Result collision_rollout(State state, Move first, const Rules& rules,
                                const ShortWeights& weights, uint64_t seed) {
  const int learner=state.to_move;
  auto streams=review_streams(seed);
  step(state,first,rules);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    auto& rng=streams[state.to_move];
    if (!review_heuristic(state.to_move,learner,2,true)) {
      step(state,review_move(d,learner,2,true,weights,rng),rules);
      continue;
    }
    Move best{};
    int value=-100000,ties=0;
    for (auto move:d.actions) {
      const int score=collision_score(d.observation,move,rules);
      if (score>value) { best=move; value=score; ties=1; }
      else if (score==value && rng.below(++ties)==0) best=move;
    }
    expect_value(ties>0,"missing collision-reference move");
    step(state,best,rules);
  }
  return state.result;
}

}
