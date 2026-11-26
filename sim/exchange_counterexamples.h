#pragma once
#include "exchange_counterplay.h"

namespace jq {
// Legal deployment puts the flag in an HQ. An eliminated seat leaves no piece.
// Therefore the only remaining HQ occupied by its owner must hold the flag.
inline int sole_enemy_hq(const Observation& view,int enemy,const Rules& rules) {
  if (!rules.flag_eliminates || !participates(view.match,enemy) ||
      view.team(enemy)==view.team(view.viewer)) return kNone;
  const auto& board=Board::get();
  const int a=board.seat_node(enemy,5,1), b=board.seat_node(enemy,5,3);
  const bool owns_a=view.seat_at(a)==enemy, owns_b=view.seat_at(b)==enemy;
  if (owns_a==owns_b) return kNone;
  return owns_a ? a : b;
}

struct FlagSamples {
  std::array<int,7> counts{};
  bool paired=false;
  int first_win_share=-1;
};

inline void sample_flag_choice(FlagSamples& samples,const State& initial,const State& state,
                               const SeatDecision& d,const MovedSquares& moved,const Rules& rules,
                               const ShortWeights& weights,const std::vector<Move>& history,uint64_t seed) {
  const auto& view=d.observation;
  auto& [decisions,inference_turns,attack_turns,win_turns,ambiguous_turns,paired_turns,win_pairs]=samples.counts;
  ++decisions;
  const int a=sole_enemy_hq(view,(view.viewer+1)%4,rules);
  const int b=sole_enemy_hq(view,(view.viewer+3)%4,rules);
  inference_turns+=a!=kNone || b!=kNone;
  for (int node : {a,b})
    if (node!=kNone) expect_value(state.rank_at[node]==Flag,"public inference does not hold");
  std::optional<Move> attack;
  bool ambiguous=false;
  for (const auto move : d.actions) {
    const auto& to=Board::get().node(move.to);
    if (move.to==a || move.to==b) {
      if (!attack) attack=move;
    } else if (to.kind==NodeKind::HQ && view.seat_at(move.to)==to.seat &&
               team_of(to.seat)!=team_of(view.viewer) && sole_enemy_hq(view,to.seat,rules)==kNone) {
      ambiguous=true;
    }
  }
  ambiguous_turns+=ambiguous;
  if (!attack) return;
  State after=state;
  const int enemy=view.seat_at(attack->to);
  step(after,*attack,rules);
  expect_value(after.out[enemy],"inferred flag attack did not eliminate its owner");
  const bool win=review_outcome(after.result,view.viewer)==2;
  ++attack_turns;
  win_turns+=win;
  if (win && samples.first_win_share==-1) {
    const auto probabilities=review_learned(d,weights);
    const auto choice=std::find(d.actions.begin(),d.actions.end(),*attack);
    samples.first_win_share=static_cast<int>(std::lround(10000*probabilities[choice-d.actions.begin()]));
  }
  const auto pair=exchange_offer(d,moved,rules,true,attack);
  if (!pair) return;
  ++paired_turns;
  win_pairs+=win;
  if (samples.paired) return;
  samples.paired=true;
  std::cout << "{\"immediate_win\":" << win << ",\"pair\":";
  print_exchange_case(initial,state,*pair,moved,rules,history,weights,seed,false);
  std::cout << '}';
}

inline void exchange_counterexamples(const Rules& rules,uint64_t seed,int learner,int variant,
                                     const ShortWeights& weights) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  MovedSquares moved{};
  std::vector<Move> history;
  FlagSamples samples;
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant << ",\"case\":";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner)
      sample_flag_choice(samples,initial,state,d,moved,rules,weights,history,seed);
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    history.push_back(action);
    const auto event=step(state,action,rules);
    observe_movement(moved,event,observe(state,learner));
  }
  if (!samples.paired) std::cout << "null";
  std::cout << ",\"counts\":[" << samples.counts[0];
  for (size_t i=1; i<samples.counts.size(); ++i) std::cout << ',' << samples.counts[i];
  std::cout << "],\"first_win_share\":" << samples.first_win_share
            << ",\"source_score\":" << review_outcome(state.result,learner) << "}\n";
}
}
