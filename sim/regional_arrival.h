#pragma once
#include "camp_control.h"

namespace jq {
inline Observation project_quiet(Observation view, Move move) {
  view.ranks[move.to]=view.ranks[move.from];
  view.seats[move.to]=view.seats[move.from];
  view.ranks[move.from]=Unknown;
  view.seats[move.from]=kNone;
  return view;
}

inline bool reaches_gate(const Observation& view,int from,int gate) {
  if (view.seat_at(from)<0 || view.seat_at(gate)!=kNone) return false;
  std::vector<Move> legal;
  gen_piece_moves(view,view.seat_at(from),from,legal);
  return std::any_of(legal.begin(),legal.end(),[=](Move move){return move.to==gate;});
}

struct GateOffer {
  Move arrival, closer;
  int gate;
};

inline std::optional<GateOffer> gate_offer(const SeatDecision& d) {
  const auto& view=d.observation;
  const auto& board=Board::get();
  for (int gate=0; gate<kNodes; ++gate) {
    if (board.node(gate).kind!=NodeKind::Center || view.seat_at(gate)!=kNone) continue;
    for (auto a : d.actions) {
      if (a.to==gate || view.seat_at(a.to)!=kNone || !board.node(a.to).rail ||
          !reaches_gate(project_quiet(view,a),a.to,gate)) continue;
      const int distance_a=distance(a.to,gate);
      for (auto b : d.actions) {
        if (a.from!=b.from || a==b || b.to==gate || view.seat_at(b.to)!=kNone ||
            distance(b.to,gate)>distance_a || short_features(view,a)!=short_features(view,b)) continue;
        if (!reaches_gate(project_quiet(view,b),b.to,gate)) return GateOffer{a,b,gate};
      }
    }
  }
  return std::nullopt;
}

inline std::array<std::array<int,2>,4> regional_ready(Observation view,const OpeningMarks& marks,int gate) {
  std::array<std::array<int,2>,4> ready{};
  constexpr unsigned stationary=(1u << Flag) | (1u << Mine);
  constexpr unsigned engineer=1u << Engineer;
  for (int from=0; from<kNodes; ++from) {
    const int seat=view.seat_at(from);
    if (seat<0) continue;
    if (view.rank_at(from)!=Unknown) {
      if (reaches_gate(view,from,gate)) { ++ready[seat][0]; ++ready[seat][1]; }
      continue;
    }
    const auto possible=marks.ranks[from];
    const bool other_possible=possible & ~(stationary | engineer);
    const bool engineer_possible=possible & engineer;
    view.ranks[from]=Platoon;
    const bool other_reaches=other_possible && reaches_gate(view,from,gate);
    view.ranks[from]=Engineer;
    const bool engineer_reaches=engineer_possible && reaches_gate(view,from,gate);
    view.ranks[from]=Unknown;
    if (!(possible & stationary) && possible && (!other_possible || other_reaches) &&
        (!engineer_possible || engineer_reaches)) ++ready[seat][0];
    if (other_reaches || engineer_reaches) ++ready[seat][1];
  }
  return ready;
}

inline std::array<int,3> regional_trial(State state,const GateOffer& offer,Move first,
                                        const Rules& rules,int response,const ShortWeights& weights,
                                        uint64_t seed) {
  const int learner=state.to_move;
  auto streams=review_streams(seed);
  step(state,first,rules);
  while (state.result==Result::Ongoing && state.to_move!=learner && !state.out[learner]) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,response,true,weights,streams[state.to_move]);
    step(state,action,rules);
  }
  const bool revisit=state.result==Result::Ongoing && state.to_move==learner;
  bool arrived=false;
  if (revisit) {
    const auto d=seat_decision(state);
    const Move arrival{first.to,static_cast<uint8_t>(offer.gate)};
    if (d.observation.seat_at(offer.gate)==kNone &&
        std::find(d.actions.begin(),d.actions.end(),arrival)!=d.actions.end()) {
      step(state,arrival,rules);
      arrived=true;
    }
  }
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,response,true,weights,streams[state.to_move]);
    step(state,action,rules);
  }
  return {review_outcome(state.result,learner),int(revisit),int(arrived)};
}

inline void print_ready(const std::array<std::array<int,2>,4>& counts) {
  std::cout << '[';
  for (size_t seat=0; seat<counts.size(); ++seat)
    std::cout << (seat ? ",[" : "[") << counts[seat][0] << ',' << counts[seat][1] << ']';
  std::cout << ']';
}

inline void print_regional_case(const State& initial,const State& state,const GateOffer& offer,
                                const OpeningMarks& marks,const std::vector<Move>& history,
                                const Rules& rules,const ShortWeights& weights,uint64_t seed) {
  const auto d=seat_decision(state);
  const auto worlds=credit_worlds(initial,state,d,rules,history,weights,true);
  std::cout << "{\"ply\":" << state.ply << ",\"gate\":" << offer.gate << ",\"arrival\":";
  print_camp_move(offer.arrival);
  std::cout << ",\"closer\":";
  print_camp_move(offer.closer);
  std::cout << ",\"ready_arrival\":";
  print_ready(regional_ready(project_quiet(d.observation,offer.arrival),marks,offer.gate));
  std::cout << ",\"ready_closer\":";
  print_ready(regional_ready(project_quiet(d.observation,offer.closer),marks,offer.gate));
  std::cout << ",\"worlds\":" << worlds.size() << ",\"cells\":[";
  for (size_t world=0; world<worlds.size(); ++world)
    for (int response : {0,2}) {
      std::cout << (world || response ? ",[" : "[");
      for (uint64_t sample=0; sample<8; ++sample) {
        const auto common=mix_seed(mix_seed(seed,static_cast<uint64_t>(state.ply)),sample+481);
        const auto a=regional_trial(worlds[world],offer,offer.arrival,rules,response,weights,common);
        const auto b=regional_trial(worlds[world],offer,offer.closer,rules,response,weights,common);
        std::cout << (sample ? ",[" : "[") << a[0] << ',' << a[1] << ',' << a[2] << ','
                  << b[0] << ',' << b[1] << ',' << b[2] << ']';
      }
      std::cout << ']';
    }
  std::cout << "]}";
}

inline void regional_arrival(const Rules& rules,uint64_t seed,int learner,int variant,
                              const ShortWeights& weights) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  OpeningMarks marks(observe(state,learner),true);
  std::vector<Move> history;
  int decisions=0;
  bool found=false;
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant << ",\"case\":";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner) {
      ++decisions;
      if (!found && decisions<=12) {
        if (const auto offer=gate_offer(d)) {
          print_regional_case(initial,state,*offer,marks,history,rules,weights,seed);
          found=true;
        }
      }
    }
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    history.push_back(action);
    step_with_public_ranks(state,rules,learner,action,&marks);
  }
  if (!found) std::cout << "null";
  std::cout << ",\"decisions\":" << decisions << ",\"source_score\":"
            << review_outcome(state.result,learner) << "}\n";
}
}
