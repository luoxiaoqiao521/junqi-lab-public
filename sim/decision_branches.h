#pragma once
#include <iomanip>
#include "draw_branch.h"
#include "seat_episode.h"
#include "../strategy/short_policy.h"

namespace jq {
inline bool review_heuristic(int seat, int learner, int variant, bool learned) {
  if (seat == learner) return !learned;
  return (variant == 1 && seat == (learner + 2) % 4) ||
         (variant == 2 && team_of(seat) != team_of(learner));
}

inline Move review_move(const SeatDecision& d, int learner, int variant, bool learned,
                        const ShortWeights& weights, Rng& rng) {
  const int seat = d.observation.viewer;
  if (seat == learner && learned) return short_move(d.observation,d.actions,weights,rng);
  if (!review_heuristic(seat,learner,variant,learned))
    return d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
  Move move;
  expect_value(heuristic_move(d.observation,rng,move),"missing heuristic move");
  return move;
}

inline std::array<Rng,4> review_streams(uint64_t seed) {
  return {Rng(mix_seed(seed,10)),Rng(mix_seed(seed,11)),Rng(mix_seed(seed,12)),Rng(mix_seed(seed,13))};
}

inline std::vector<double> review_reference(const SeatDecision& d) {
  std::vector<int> scores;
  for (auto move : d.actions) scores.push_back(score_move(d.observation,move).total());
  const auto best = *std::max_element(scores.begin(),scores.end());
  const double tied = static_cast<double>(std::count(scores.begin(),scores.end(),best));
  std::vector<double> probabilities;
  for (auto score : scores) probabilities.push_back(score==best ? 1.0/tied : 0.0);
  return probabilities;
}

inline std::vector<double> review_learned(const SeatDecision& d, const ShortWeights& weights) {
  auto probabilities=short_action_masses(d.observation,d.actions,weights);
  double total=0;
  for (double value : probabilities) total+=value;
  for (double& value : probabilities) value/=total;
  return probabilities;
}

struct ReviewPoint {
  State state;
  size_t prefix_length;
  int turn;
};

inline Result review_rollout(State state, Move first, const Rules& rules, int variant,
                             bool learned, const ShortWeights& weights, uint64_t seed,
                             int* elapsed=nullptr) {
  const int learner=state.to_move;
  const int starting_ply=state.ply;
  auto streams=review_streams(seed);
  step(state,first,rules);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,variant,learned,weights,streams[state.to_move]);
    step(state,action,rules);
  }
  if (elapsed) *elapsed=state.ply-starting_ply;
  return state.result;
}

inline int review_outcome(Result result, int learner) {
  const auto score=team_score(result,learner);
  return score ? static_cast<int>(*score*2) : -1;
}

inline void review_response_cells(const std::vector<State>& worlds, Move promoted, Move displaced,
                                  const Rules& rules, int variant, const ShortWeights& weights,
                                  uint64_t seed) {
  bool comma=false;
  const int learner=worlds[0].to_move;
  for (size_t world=0; world<worlds.size(); ++world)
    for (int response : {variant,variant==2 ? 0 : 2})
      for (int continuation : {0,1}) {
        std::cout << (comma ? "," : "") << "{\"world\":" << world << ",\"response\":" << response
                  << ",\"continuation\":" << continuation << ",\"pairs\":[";
        comma=true;
        for (uint64_t sample=0; sample<12; ++sample) {
          const auto paired=mix_seed(seed,sample+20);
          const auto a=review_rollout(worlds[world],promoted,rules,response,continuation==1,weights,paired);
          const auto b=review_rollout(worlds[world],displaced,rules,response,continuation==1,weights,paired);
          std::cout << (sample ? "," : "") << '[' << review_outcome(a,learner)
                    << ',' << review_outcome(b,learner) << ']';
        }
        std::cout << "]}";
      }
}

inline void review_decision(const State& initial, const ReviewPoint& point,
                            const std::vector<Move>& history, const Rules& rules,
                            int variant, int source, int final_score, uint64_t seed,
                            const ShortWeights& weights) {
  const auto d=seat_decision(point.state);
  const auto before=review_reference(d), after=review_learned(d,weights);
  size_t plus=0, minus=0;
  double variation=0;
  for (size_t i=0; i<before.size(); ++i) {
    const double delta=after[i]-before[i];
    if (delta>after[plus]-before[plus]) plus=i;
    if (delta<after[minus]-before[minus]) minus=i;
    variation+=std::abs(delta)/2;
  }
  std::cout << std::setprecision(17) << "{\"source\":" << source << ",\"score\":" << final_score
            << ",\"turn\":" << point.turn << ",\"ply\":" << point.state.ply
            << ",\"tv\":" << variation << ",\"actions\":[";
  for (size_t i=0; i<d.actions.size(); ++i) {
    const auto move=d.actions[i];
    std::cout << (i ? "," : "") << '[' << int(move.from) << ',' << int(move.to)
              << ',' << before[i] << ',' << after[i] << ']';
  }
  std::cout << "],\"plus\":" << plus << ",\"minus\":" << minus << ",\"cells\":[";
  if (variation>1e-12) {
    const std::vector<Move> prefix(history.begin(),history.begin()+point.prefix_length);
    const auto worlds=draw_worlds(initial,point.state,rules,prefix);
    for (const auto& world : worlds) {
      const auto check=seat_decision(world);
      expect_value(check.actions==d.actions && review_reference(check)==before &&
                   review_learned(check,weights)==after,"equivalent public history changed the policy");
    }
    review_response_cells(worlds,d.actions[plus],d.actions[minus],rules,variant,weights,
                          mix_seed(seed,static_cast<uint64_t>(point.state.ply)));
  }
  std::cout << "]}\n";
}

inline void decision_branches(const Rules& rules, uint64_t seed, int learner, int variant,
                              const ShortWeights& weights) {
  Rng layouts_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layouts_rng);
  const int first=rules.random_first_seat ? int(layouts_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  for (int source : {0,1}) {
    State state=initial;
    auto streams=review_streams(seed);
    std::vector<Move> history;
    std::vector<ReviewPoint> points;
    int turns=0;
    while (state.result==Result::Ongoing) {
      const auto d=seat_decision(state);
      if (state.to_move==learner) {
        ++turns;
        if (turns==12 || turns==40) points.push_back({state,history.size(),turns});
      }
      const auto move=review_move(d,learner,variant,source==1,weights,streams[state.to_move]);
      history.push_back(move);
      step(state,move,rules);
    }
    const int score=review_outcome(state.result,learner);
    std::cout << "{\"source\":" << source << ",\"score\":" << score
              << ",\"game_plies\":" << state.ply << ",\"first\":" << first << "}\n";
    for (const auto& point : points)
      review_decision(initial,point,history,rules,variant,source,score,seed,weights);
  }
}
}
