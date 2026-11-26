#pragma once
#include "mine_evidence.h"

namespace jq {
inline void cover_flag_entrances(int8_t (&layout)[kPiecesPerSeat],int learner) {
  const auto first=std::begin(layout);
  const int flag_slot=int(std::find(first,std::end(layout),Flag)-first);
  const auto& road=Board::get().road(layout_slot_node(learner,flag_slot));
  const auto neighbor=[&](int slot) {
    return std::find(road.begin(),road.end(),layout_slot_node(learner,slot))!=road.end();
  };
  for (int slot=0; slot<kPiecesPerSeat; ++slot) {
    if (!neighbor(slot) || layout[slot]==Mine) continue;
    for (int other=0; other<kPiecesPerSeat; ++other) {
      if (layout[other]==Mine && !neighbor(other)) { std::swap(layout[slot],layout[other]); break; }
    }
  }
  expect_value(valid_layout(layout),"invalid closed layout");
}

inline bool open_one_flag_entrance(int8_t (&layout)[kPiecesPerSeat],int learner) {
  const auto first=std::begin(layout);
  const int flag_slot=int(std::find(first,std::end(layout),Flag)-first);
  const auto& road=Board::get().road(layout_slot_node(learner,flag_slot));
  for (int a=0; a<kPiecesPerSeat; ++a) {
    if (layout[a]!=Mine) continue;
    for (int b=0; b<kPiecesPerSeat; ++b) {
      const auto node=layout_slot_node(learner,b);
      if (!is_army(layout[b]) || !deployment_rank_allowed(node,Mine) ||
          Board::get().node(node).kind==NodeKind::HQ) continue;
      std::swap(layout[a],layout[b]);
      expect_value(std::find(road.begin(),road.end(),node)==road.end() && valid_layout(layout),
                   "opening the route changed the formation contract");
      return true;
    }
  }
  return false;
}

inline int opening_ready_count(const SeatDecision& d) {
  std::array<bool,kNodes> access{};
  for (const auto move : d.actions)
    if (Board::get().node(move.to).kind==NodeKind::Center) access[move.from]=true;
  return static_cast<int>(std::count(access.begin(),access.end(),true));
}

inline std::array<int,4> formation_continuation(State state,const Rules& rules,int learner,
                                                int variant,const ShortWeights& weights,uint64_t seed) {
  auto streams=review_streams(seed);
  const auto threat=(learner+(rules.ccw ? 1 : 3))%4;
  int own_flag=kNone;
  for (int node=0; node<kNodes; ++node)
    if (state.seat_at[node]==learner && state.rank_at[node]==Flag) own_flag=node;
  int first_ready=-1;
  int first_center=-1;
  bool taken=false;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const int actor=d.observation.viewer;
    if (actor==learner && first_ready < 0) first_ready=opening_ready_count(d);
    const auto move=review_move(d,learner,actor==threat ? variant : 0,true,weights,streams[actor]);
    if (actor==learner && first_center<0 && Board::get().node(move.to).kind==NodeKind::Center)
      first_center=state.ply+1;
    const auto event=step(state,move,rules);
    taken|=event.move.to==own_flag &&
      (event.outcome==Outcome::AttackerWins || event.outcome==Outcome::BothRemoved);
  }
  return {review_outcome(state.result,learner),first_ready,first_center,int(taken)};
}

inline void print_formation_metrics(const std::array<int,4>& metrics) {
  std::cout << '[' << metrics[0] << ',' << metrics[1] << ',' << metrics[2] << ',' << metrics[3] << ']';
}

inline void formation_one_swap(const Rules& rules,uint64_t seed,int learner,int variant,
                               const ShortWeights& weights) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  cover_flag_entrances(layouts[learner],learner);
  State closed, open;
  init_state(closed,layouts,first,rules);
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant;
  if (!open_one_flag_entrance(layouts[learner],learner)) {
    std::cout << ",\"pair\":null}\n";
    return;
  }
  init_state(open,layouts,first,rules);
  expect_value(opening_mine_neighbors(closed)[learner]==3 && opening_mine_neighbors(open)[learner]==2,
               "formation counts are not an exact one-mine difference");
  for (int seat=0; seat<kSeats; ++seat)
    if (seat!=learner) expect_value(same_seat_view(closed,open,seat),"formation leaked another seat's ranks");
  std::cout << ",\"pair\":[";
  print_formation_metrics(formation_continuation(closed,rules,learner,variant,weights,seed));
  std::cout << ',';
  print_formation_metrics(formation_continuation(open,rules,learner,variant,weights,seed));
  std::cout << "]}\n";
}
}
