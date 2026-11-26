#pragma once
#include "flag_interception.h"

namespace jq {
struct UrgencyChoice {
  FlagThreat offer;
  int group;
};

inline std::optional<UrgencyChoice> flag_urgency_choice(const SeatDecision& d,
                                                        const MovedSquares& moved, const Rules& rules) {
  const auto& view=d.observation;
  const int early=(view.viewer+(rules.ccw ? 1 : 3))%4, late=(view.viewer+(rules.ccw ? 3 : 1))%4;
  const auto advance=quiet_advance(d);
  if (!advance) return std::nullopt;
  for (int flag=0; flag<kNodes; ++flag) {
    if (view.seat_at(flag)!=view.viewer || view.rank_at(flag)!=Flag) continue;
    const auto& roads=Board::get().road(flag);
    const auto choice=[&](int actor,int group) -> std::optional<UrgencyChoice> {
      for (const auto target : roads) {
        if (view.seat_at(target)!=actor || !publicly_mobile(view,moved,target)) continue;
        for (const auto move : d.actions)
          if (move.to==target) return UrgencyChoice{{flag,target,move,*advance},group};
      }
      return std::nullopt;
    };
    const bool immediate=std::any_of(roads.begin(),roads.end(),[&](int target) {
      return view.seat_at(target)==early && publicly_mobile(view,moved,target);
    });
    return immediate ? choice(early,0) : choice(late,1);
  }
  return std::nullopt;
}

inline std::array<int,5> urgency_delta(const Observation& view, const FlagThreat& offer) {
  const auto a=short_features(view,offer.intercept), b=short_features(view,offer.advance);
  return {int(a[0]-b[0]),int(a[1]-b[1]),int(a[2]-b[2]),
          view.rank_at(offer.intercept.from)-view.rank_at(offer.advance.from),int(a[4]-b[4])};
}

inline std::array<int,5> urgency_trial(State state, const FlagThreat& offer, Move first,
                                      const Rules& rules, const ShortWeights& weights,
                                      uint64_t seed, bool pressure) {
  const int learner=state.to_move, actor=state.seat_at[offer.threat];
  auto streams=review_streams(seed);
  step(state,first,rules);
  bool lost=false, pressure_used=false, ally_first=false, threat_turn=false;
  while (state.result==Result::Ongoing && state.to_move!=learner && !state.out[learner]) {
    const auto d=seat_decision(state);
    const bool first_threat_turn=d.observation.viewer==actor;
    const auto forced=first_threat_turn && pressure ? public_hq_attack(d) : std::nullopt;
    const auto move=forced ? *forced :
      review_move(d,learner,0,true,weights,streams[state.to_move]);
    const auto event=step(state,move,rules);
    lost|=event.move.to==offer.flag &&
      (event.outcome==Outcome::AttackerWins || event.outcome==Outcome::BothRemoved);
    pressure_used|=bool(forced);
    ally_first|=(event.seat==(learner+2)%4);
    if (first_threat_turn) { threat_turn=true; break; }
  }
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto move=review_move(d,learner,0,true,weights,streams[state.to_move]);
    step(state,move,rules);
  }
  return {review_outcome(state.result,learner),int(lost),int(pressure_used),int(ally_first),int(threat_turn)};
}

inline void print_urgency_outcomes(const std::vector<State>& worlds,const FlagThreat& offer,
                                   const Rules& rules,const ShortWeights& weights,uint64_t seed) {
  for (size_t world=0; world<worlds.size(); ++world)
    for (int response : {0,1}) {
      std::cout << (world || response ? ",[" : "[");
      for (uint64_t sample=0; sample<8; ++sample) {
        const auto common=mix_seed(seed,sample+571);
        const auto a=urgency_trial(worlds[world],offer,offer.intercept,rules,weights,common,response==1);
        const auto b=urgency_trial(worlds[world],offer,offer.advance,rules,weights,common,response==1);
        std::cout << (sample ? ",[" : "[");
        bool comma=false;
        for (const auto& values : {a,b})
          for (const int value : values) {
            std::cout << (comma ? "," : "") << value;
            comma=true;
          }
        std::cout << ']';
      }
      std::cout << ']';
    }
}

inline void print_urgency_point(const State& initial,const State& state,const UrgencyChoice& choice,
                                const Rules& rules,const std::vector<Move>& history,const ShortWeights& weights,
                                uint64_t seed,bool pair) {
  const auto d=seat_decision(state);
  const auto& offer=choice.offer;
  std::cout << "{\"group\":" << choice.group << ",\"ply\":" << state.ply << ",\"flag\":" << offer.flag
            << ",\"threat\":" << offer.threat << ",\"intercept\":";
  print_camp_move(offer.intercept);
  std::cout << ",\"advance\":";
  print_camp_move(offer.advance);
  std::cout << ",\"delta\":[";
  const auto difference=urgency_delta(d.observation,offer);
  for (size_t i=0; i<difference.size(); ++i) std::cout << (i ? "," : "") << difference[i];
  std::cout << ']';
  if (pair) {
    const auto worlds=credit_worlds(initial,state,d,rules,history,weights,true);
    std::cout << ",\"worlds\":" << worlds.size() << ",\"cells\":[";
    print_urgency_outcomes(worlds,offer,rules,weights,mix_seed(seed,static_cast<uint64_t>(state.ply)));
    std::cout << ']';
  }
  std::cout << '}';
}

inline void flag_urgency(const Rules& rules,uint64_t seed,int learner,int variant,
                         const ShortWeights& weights,int selected=-1) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  std::vector<Move> history;
  MovedSquares moved{};
  std::array<bool,2> found{};
  bool comma=false;
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant << ",\"points\":[";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner && (!found[0] || !found[1])) {
      const auto choice=flag_urgency_choice(d,moved,rules);
      if (choice && !found[choice->group] && (selected==-1 || selected==choice->group)) {
        if (comma) std::cout << ',';
        comma=true;
        found[choice->group]=true;
        print_urgency_point(initial,state,*choice,rules,history,weights,seed,selected!=-1);
        if (selected!=-1) break;
      }
    }
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    history.push_back(action);
    const auto event=step(state,action,rules);
    observe_movement(moved,event,observe(state,learner));
  }
  std::cout << "]}\n";
}
}
