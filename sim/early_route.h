#pragma once
#include "regional_arrival.h"

namespace jq {
struct EarlyRouteOffer {
  Move opens;
  std::array<int,2> partner_ready;
};

inline std::optional<EarlyRouteOffer> early_route_offer(const SeatDecision& d,const OpeningMarks& marks) {
  const auto& board=Board::get();
  const auto& view=d.observation;
  const int partner=(view.viewer+2)%4;
  for (auto move : d.actions) {
    if (board.node(move.from).kind!=NodeKind::Center || view.seat_at(move.to)!=kNone) continue;
    const auto ready=regional_ready(project_quiet(view,move),marks,move.from)[partner];
    if (ready[1]) return EarlyRouteOffer{move,ready};
  }
  return std::nullopt;
}

inline std::array<int,5> early_route_trial(State state,Move first,int gate,const Rules& rules,
                                         int response,const ShortWeights& weights,uint64_t seed,
                                         bool accept=false) {
  const int learner=state.to_move, ally=(learner+2)%4;
  auto streams=review_streams(seed);
  int ally_turn=0, allowed=0, used=0, enemy_entered=0;
  bool first_rotation=true;
  step(state,first,rules);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const int seat=state.to_move;
    auto move=review_move(d,learner,response,true,weights,streams[seat]);
    if (seat==learner) first_rotation=false;
    if (first_rotation && seat==ally) {
      ally_turn=1;
      if (d.observation.seat_at(gate)==kNone) {
        const auto route=std::find_if(d.actions.begin(),d.actions.end(),[=](Move a){return a.to==gate;});
        allowed=route!=d.actions.end();
        if (accept && allowed && move.to!=gate) move=*route;
        used=move.to==gate;
      }
      first_rotation=false;
    }
    if (first_rotation && team_of(seat)!=team_of(learner) && move.to==gate) enemy_entered=1;
    step(state,move,rules);
  }
  return {review_outcome(state.result,learner),ally_turn,allowed,used,enemy_entered};
}

inline void print_early_trials(const State& world,const std::vector<Move>& moves,int gate,
                              const Rules& rules,int response,const ShortWeights& weights,
                              uint64_t seed,bool followthrough) {
  const auto arms=followthrough ? 4 : moves.size();
  for (size_t a=0;a<arms;++a) {
    if (a) std::cout << ',';
    const auto first=moves[followthrough ? a%2 : a];
    const auto result=early_route_trial(world,first,gate,rules,response,weights,seed,followthrough && a>=2);
    std::cout << '[' << result[0] << ',' << result[1] << ',' << result[2] << ',' << result[3]
              << ',' << result[4] << ']';
  }
}

inline void early_route_forks(const State& initial,const State& state,const EarlyRouteOffer& offer,
                              Move played,const std::vector<Move>& history,const Rules& rules,
                              const ShortWeights& weights,uint64_t seed,int decision,bool followthrough) {
  const auto d=seat_decision(state);
  const int gate=offer.opens.from;
  std::vector<Move> moves{played,offer.opens};
  if (!followthrough) {
    std::vector<Move> outside;
    for (const auto move : d.actions)
      if (!(move==played) && move.from!=gate && d.observation.seat_at(move.to)==kNone)
        outside.push_back(move);
    if (outside.empty()) { std::cout << "null"; return; }
    Rng rng(mix_seed(seed,1792));
    moves.push_back(outside[rng.below(static_cast<uint32_t>(outside.size()))]);
  }
  const auto worlds=credit_worlds(initial,state,d,rules,history,weights,true);
  std::cout << "{\"ply\":" << state.ply << ",\"decision\":" << decision << ",\"gate\":" << gate
            << ",\"partner_ready\":[" << offer.partner_ready[0] << ',' << offer.partner_ready[1]
            << "],\"worlds\":" << worlds.size() << ",\"moves\":[";
  for (size_t arm=0;arm<moves.size();++arm) {
    if (arm) std::cout << ',';
    print_camp_move(moves[arm]);
  }
  std::cout << "],\"cells\":[";
  for (size_t i=0;i<worlds.size();++i) for (int response : {0,2}) {
    std::cout << (i || response ? ",[" : "[");
    for (uint64_t j=0;j<10;++j) {
      std::cout << (j ? ",[" : "[");
      const auto common=mix_seed(mix_seed(seed,state.ply),(followthrough ? 48000 : 41000)+j);
      print_early_trials(worlds[i],moves,gate,rules,response,weights,common,followthrough);
      std::cout << ']';
    }
    std::cout << ']';
  }
  std::cout << "]}";
}

inline void early_route_probe(const Rules& rules,uint64_t seed,int learner,int variant,
                              const ShortWeights& weights,bool followthrough=false) {
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
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant << ",\"window\":";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto move=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    if (state.to_move==learner) {
      ++decisions;
      if (!found && decisions<=16 && decisions%4==0) {
        if (const auto offer=early_route_offer(d,marks)) {
          early_route_forks(initial,state,*offer,move,history,rules,weights,seed,decisions,followthrough);
          found=true;
        }
      }
    }
    history.push_back(move);
    step_with_public_ranks(state,rules,learner,move,&marks);
  }
  if (!found) std::cout << "null";
  std::cout << ",\"first\":" << first << ",\"decisions\":" << decisions
            << ",\"score\":" << review_outcome(state.result,learner) << "}\n";
}
}
