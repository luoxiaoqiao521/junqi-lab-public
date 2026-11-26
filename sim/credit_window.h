#pragma once
#include "decision_branches.h"

namespace jq {
struct CreditPoint {
  State state;
  Move played;
};

inline std::vector<std::array<int,3>> live_swap_proposals(const State& initial, const State& at) {
  const auto view=observe(at,at.to_move);
  std::vector<std::array<int,3>> pairs;
  for (int seat : {(at.to_move+1)%4,(at.to_move+3)%4}) {
    std::array<bool,25> hidden_live{}, visible_live{};
    for (int slot=0; slot<25; ++slot) {
      const auto id=initial.id_at[layout_slot_node(seat,slot)];
      if (!at.alive[id]) continue;
      hidden_live[slot]=view.rank_at(at.node_of[id])==Unknown;
      visible_live[slot]=!hidden_live[slot];
    }
    for (int a=0; a<25; ++a)
      for (int b=a+1; b<25; ++b)
        if ((hidden_live[a] || hidden_live[b]) && !visible_live[a] && !visible_live[b])
          pairs.push_back({seat,a,b});
  }
  return pairs;
}

inline std::optional<State> additional_live_world(const State& initial, const State& at,
                                                  const Rules& rules,
                                                  const std::vector<Move>& prefix) {
  auto proposals=live_swap_proposals(initial,at);
  Rng rng(98143);
  for (int attempts=0; attempts<96 && !proposals.empty(); ++attempts) {
    const auto index=rng.below(static_cast<uint32_t>(proposals.size()));
    const auto [seat,a,b]=proposals[index];
    proposals[index]=proposals.back();
    proposals.pop_back();
    State alternative;
    if (swap_in_layout(initial,alternative,seat,a,b) &&
        replay_alternative(initial,alternative,rules,prefix,at.to_move))
      return alternative;
  }
  return std::nullopt;
}

inline bool live_hidden_change(const State& original, const State& alternative) {
  for (int node=0; node<kNodes; ++node)
    if (original.seat_at[node]>=0 && original.rank_at[node]!=alternative.rank_at[node])
      return true;
  return false;
}

inline std::vector<State> credit_worlds(const State& initial, const State& current,
                                        const SeatDecision& d, const Rules& rules,
                                        const std::vector<Move>& prefix,
                                        const ShortWeights& weights, bool prefer_live) {
  auto alternative=std::optional<State>{};
  for (const auto& candidate : draw_worlds(initial,current,rules,prefix)) {
    if (!live_hidden_change(current,candidate)) continue;
    alternative=candidate;
    break;
  }
  if (!alternative && prefer_live) alternative=additional_live_world(initial,current,rules,prefix);
  if (!alternative) return {current};
  const auto checked=seat_decision(*alternative);
  expect_value(live_hidden_change(current,*alternative) && checked.actions==d.actions &&
               review_learned(checked,weights)==review_learned(d,weights),
               "compatible history changed the legal actions or policy");
  return {current,*alternative};
}

inline std::vector<Move> credit_options(const SeatDecision& d, Move played, bool equal) {
  std::vector<Move> choices;
  const auto actual=short_features(d.observation,played);
  for (Move action : d.actions) {
    if (action==played) continue;
    if ((short_features(d.observation,action)==actual)==equal) choices.push_back(action);
  }
  return choices;
}

inline void print_credit_action(const Observation& view, Move action) {
  const auto features=short_features(view,action,true);
  const auto legacy=score_move(view,action);
  std::cout << '[' << int(action.from) << ',' << int(action.to);
  for (double value : features) std::cout << ',' << value;
  std::cout << ',' << legacy.objective << ',' << legacy.position << ',' << legacy.exposure << ']';
}

inline void print_credit_pairs(const std::vector<State>& worlds, Move played, Move alternative,
                               const Rules& rules, const ShortWeights& weights, uint64_t seed) {
  bool comma=false;
  for (const auto& world : worlds) {
    for (int response : {0,2}) {
      std::cout << (comma ? ",[" : "[");
      comma=true;
      for (uint64_t sample=0; sample<8; ++sample) {
        const auto common=mix_seed(seed,sample+20);
        int a_plies=0, b_plies=0;
        const auto a=review_rollout(world,played,rules,response,true,weights,common,&a_plies);
        const auto b=review_rollout(world,alternative,rules,response,true,weights,common,&b_plies);
        std::cout << (sample ? ",[" : "[") << review_outcome(a,world.to_move) << ','
                  << review_outcome(b,world.to_move) << ',' << a_plies << ',' << b_plies << ']';
      }
      std::cout << ']';
    }
  }
}

inline void print_credit_window(const State& initial, const CreditPoint& point, int terminal,
                                const std::vector<Move>& history, const Rules& rules,
                                const ShortWeights& weights, uint64_t seed, bool reseed,
                                bool prefer_live) {
  const auto d=seat_decision(point.state);
  const std::vector<Move> prefix(history.begin(),history.begin()+point.state.ply);
  const auto worlds=credit_worlds(initial,point.state,d,rules,prefix,weights,prefer_live);
  std::cout << "{\"ply\":" << point.state.ply << ",\"remaining\":" << terminal-point.state.ply
            << ",\"worlds\":" << worlds.size() << ",\"played\":";
  print_credit_action(d.observation,point.played);
  std::cout << ",\"alternatives\":[";
  const auto common=mix_seed(seed,static_cast<uint64_t>(point.state.ply));
  for (bool equal : {true,false}) {
    if (!equal) std::cout << ',';
    const auto choices=credit_options(d,point.played,equal);
    if (choices.empty()) {
      std::cout << "null";
      continue;
    }
    Rng rng(common);
    const auto alternative=choices.at(rng.below(static_cast<uint32_t>(choices.size())));
    std::cout << "{\"action\":";
    print_credit_action(d.observation,alternative);
    std::cout << ",\"cells\":[";
    const auto continuation_seed=reseed ? mix_seed(common,1001) : common;
    print_credit_pairs(worlds,point.played,alternative,rules,weights,continuation_seed);
    std::cout << "]}";
  }
  std::cout << "]}";
}

inline const CreditPoint* credit_anchor(const std::vector<CreditPoint>& points, int terminal,
                                       int lower, int upper, int target) {
  const CreditPoint* best=nullptr;
  for (const auto& point : points) {
    const int remaining=terminal-point.state.ply;
    if (remaining<lower || remaining>upper) continue;
    if (!best || std::abs(remaining-target)<std::abs(terminal-best->state.ply-target)) best=&point;
  }
  return best;
}

inline void credit_windows(const Rules& rules, uint64_t seed, int learner, int variant,
                           const ShortWeights& weights, bool reseed, bool prefer_live) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  std::vector<Move> history;
  std::vector<CreditPoint> points;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    if (state.to_move==learner && d.actions.size()>1) points.push_back({state,action});
    history.push_back(action);
    step(state,action,rules);
  }
  std::cout << std::setprecision(17) << "{\"seed\":" << seed << ",\"seat\":" << learner
            << ",\"variant\":" << variant << ",\"score\":" << review_outcome(state.result,learner)
            << ",\"plies\":" << state.ply << ",\"windows\":[";
  for (const auto bounds : {std::array<int,3>{4,16,12},std::array<int,3>{17,48,32}}) {
    if (bounds[0]!=4) std::cout << ',';
    const auto* point=credit_anchor(points,state.ply,bounds[0],bounds[1],bounds[2]);
    if (point && state.result!=Result::Truncated)
      print_credit_window(initial,*point,state.ply,history,rules,weights,seed,reseed,prefer_live);
    else std::cout << "null";
  }
  std::cout << "]}\n";
}
}
