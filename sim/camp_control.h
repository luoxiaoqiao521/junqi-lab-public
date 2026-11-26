#pragma once
#include "credit_window.h"

namespace jq {
struct CampChoice {
  Move seal, press;
  int target, backup_camp;
};

inline bool publicly_mobile(const Observation& view, const MovedSquares& moved, int node) {
  return view.seat_at(node)>=0 && Board::get().node(node).kind!=NodeKind::HQ &&
         (moved[node] || view.rank_at(node)>=Bomb);
}

inline int next_empty_camp(const Observation& view, int target, int excluded, int vacated=kNone) {
  const auto& board=Board::get();
  for (const auto node : board.road(target))
    if (node!=excluded && (view.seat_at(node)==kNone || node==vacated) &&
        board.node(node).kind==NodeKind::Camp)
      return node;
  return kNone;
}

inline std::optional<Move> camp_press(const SeatDecision& decision, Move seal, int target) {
  const auto& board=Board::get();
  for (const auto action : decision.actions)
    if (action.from==seal.from && decision.observation.seat_at(action.to)==kNone &&
        board.node(action.to).rail && distance(action.to,target)<distance(seal.from,target))
      return action;
  return std::nullopt;
}

inline std::array<std::optional<CampChoice>,2> camp_choices(
    const SeatDecision& d, const MovedSquares& moved) {
  const auto& board=Board::get();
  const auto& view=d.observation;
  std::array<std::optional<CampChoice>,2> found;
  for (const auto seal : d.actions) {
    if (board.node(seal.to).kind!=NodeKind::Camp || view.seat_at(seal.to)!=kNone) continue;
    for (const auto target : board.road(seal.to)) {
      if (team_of(view.seat_at(target))==team_of(view.viewer) ||
          !publicly_mobile(view,moved,target) || board.node(target).kind==NodeKind::Camp) continue;
      const int backup=next_empty_camp(view,target,seal.to,seal.from);
      auto& entry=found[backup!=kNone];
      if (entry) continue;
      if (const auto press=camp_press(d,seal,target)) entry=CampChoice{seal,*press,target,backup};
      if (found[0] && found[1]) return found;
    }
  }
  return found;
}

struct CampPoint {
  State state;
  MovedSquares moved;
  CampChoice choice;
};

struct CampTrace {
  State opening, terminal;
  std::vector<Move> history;
  std::array<std::optional<CampPoint>,2> points;
  int decisions=0;
};

inline CampTrace camp_trace(const Rules& rules, uint64_t seed, int learner,
                            int variant, const ShortWeights& weights) {
  CampTrace trace;
  Rng layouts_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layouts_rng);
  const auto first=rules.random_first_seat ? int(layouts_rng.below(kSeats)) : 0;
  init_state(trace.opening,layouts,first,rules);
  auto state=trace.opening;
  auto streams=review_streams(seed);
  MovedSquares moved{};
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner) {
      ++trace.decisions;
      if (!trace.points[0] || !trace.points[1]) {
        const auto choices=camp_choices(d,moved);
        for (size_t kind=0; kind<choices.size(); ++kind)
          if (choices[kind] && !trace.points[kind])
            trace.points[kind]=CampPoint{state,moved,*choices[kind]};
      }
    }
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    trace.history.push_back(action);
    const auto event=step(state,action,rules);
    observe_movement(moved,event,observe(state,learner));
  }
  trace.terminal=state;
  return trace;
}

inline void print_camp_move(Move action) {
  std::cout << '[' << int(action.from) << ',' << int(action.to) << ']';
}

inline void print_camp_point(const CampPoint& point) {
  const auto view=observe(point.state,point.state.to_move);
  const auto& c=point.choice;
  std::cout << "{\"ply\":" << view.ply << ",\"target\":" << c.target
            << ",\"target_seat\":" << view.seat_at(c.target)
            << ",\"seen_moved\":" << (point.moved[c.target] ? "true" : "false")
            << ",\"backup\":" << c.backup_camp << ",\"seal\":";
  print_camp_move(c.seal);
  std::cout << ",\"press\":";
  print_camp_move(c.press);
  std::cout << ",\"same_five\":"
            << (short_features(view,c.seal)==short_features(view,c.press) ? "true" : "false") << '}';
}

inline void scan_camp_control(const Rules& rules, uint64_t seed, int learner,
                               int variant, const ShortWeights& weights) {
  const auto trace=camp_trace(rules,seed,learner,variant,weights);
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant
            << ",\"score\":" << review_outcome(trace.terminal.result,learner)
            << ",\"decisions\":" << trace.decisions << ",\"points\":[";
  for (size_t i=0; i<trace.points.size(); ++i) {
    if (i) std::cout << ',';
    if (trace.points[i]) print_camp_point(*trace.points[i]);
    else std::cout << "null";
  }
  std::cout << "]}\n";
}
}
