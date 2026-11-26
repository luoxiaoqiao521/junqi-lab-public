#pragma once
#include "camp_control.h"

namespace jq {
struct FlagThreat {
  int flag, threat;
  Move intercept, advance;
};

inline std::optional<Move> quiet_advance(const SeatDecision& d) {
  for (const auto action : d.actions)
    if (d.observation.seat_at(action.to)==kNone &&
        enemy_hq_distance(d.observation.viewer,action.to)<enemy_hq_distance(d.observation.viewer,action.from))
      return action;
  return std::nullopt;
}

inline std::array<std::optional<FlagThreat>,2> public_flag_threats(
    const SeatDecision& d, const MovedSquares& moved, const Rules& rules) {
  std::array<std::optional<FlagThreat>,2> choices;
  const auto& view=d.observation;
  const auto next_enemy=(view.viewer+(rules.ccw ? 1 : 3))%4;
  const auto advance=quiet_advance(d);
  if (!advance) return choices;
  for (int flag=0; flag<kNodes; ++flag) {
    const auto owner=view.seat_at(flag);
    if (view.rank_at(flag)!=Flag || team_of(owner)!=team_of(view.viewer)) continue;
    auto& result=choices[owner!=view.viewer];
    if (result) continue;
    for (const auto node : Board::get().road(flag)) {
      if (view.seat_at(node)!=next_enemy || !publicly_mobile(view,moved,node)) continue;
      const auto intercept=std::find_if(d.actions.begin(),d.actions.end(),
                                       [=](Move action){return action.to==node;});
      if (intercept==d.actions.end()) continue;
      result=FlagThreat{flag,node,*intercept,*advance};
      break;
    }
  }
  return choices;
}

inline std::optional<Move> public_hq_attack(const SeatDecision& d) {
  std::optional<Move> occupied_hq;
  for (const auto action : d.actions) {
    const auto owner=d.observation.seat_at(action.to);
    if (owner<0 || team_of(owner)==team_of(d.observation.viewer) ||
        Board::get().node(action.to).kind!=NodeKind::HQ) continue;
    if (d.observation.rank_at(action.to)==Flag) return action;
    if (!occupied_hq) occupied_hq=action;
  }
  return occupied_hq;
}

inline std::array<int,4> flag_continuation(State state, Move first, int flag,
                                           const Rules& rules, const ShortWeights& weights,
                                           uint64_t seed, bool prefer_hq) {
  const auto learner=state.to_move;
  const auto enemy=(learner+(rules.ccw ? 1 : 3))%4;
  auto streams=review_streams(seed);
  step(state,first,rules);
  bool flag_lost=false, enemy_saw_flag=false, priority_applied=false;
  if (state.result==Result::Ongoing && state.to_move==enemy) {
    const auto d=seat_decision(state);
    enemy_saw_flag=d.observation.rank_at(flag)==Flag;
    const auto capture=prefer_hq ? public_hq_attack(d) : std::nullopt;
    priority_applied=bool(capture);
    const auto action=capture ? *capture :
      review_move(d,learner,0,true,weights,streams[state.to_move]);
    const auto event=step(state,action,rules);
    flag_lost=event.move.to==flag &&
      (event.outcome==Outcome::AttackerWins || event.outcome==Outcome::BothRemoved);
  }
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,0,true,weights,streams[state.to_move]);
    step(state,action,rules);
  }
  return {review_outcome(state.result,learner),int(flag_lost),int(enemy_saw_flag),int(priority_applied)};
}

inline void print_flag_rollout(const std::array<int,4>& branch) {
  std::cout << '[' << branch[0] << ',' << branch[1] << ',' << branch[2] << ',' << branch[3] << ']';
}

inline void report_flag_pairs(const State& initial, const State& point, const Rules& rules,
                              const std::vector<Move>& prefix, const FlagThreat& choice,
                              const ShortWeights& weights, uint64_t seed, size_t kind) {
  const auto d=seat_decision(point);
  const auto worlds=credit_worlds(initial,point,d,rules,prefix,weights,true);
  std::cout << "{\"kind\":" << kind << ",\"ply\":" << point.ply << ",\"flag\":" << choice.flag
            << ",\"threat\":" << choice.threat << ",\"intercept\":";
  print_camp_move(choice.intercept);
  std::cout << ",\"advance\":";
  print_camp_move(choice.advance);
  std::cout << ",\"same_five\":"
            << (short_features(d.observation,choice.intercept)==short_features(d.observation,choice.advance)
                ? "true" : "false") << ",\"worlds\":" << worlds.size() << ",\"cells\":[";
  for (size_t world=0; world<worlds.size(); ++world)
    for (int response : {0,1}) {
      std::cout << (world || response ? "," : "") << "{\"world\":" << world
                << ",\"response\":" << response << ",\"pairs\":[";
      for (uint64_t sample=0; sample<8; ++sample) {
        const auto common=mix_seed(mix_seed(seed,static_cast<uint64_t>(point.ply)),sample+310);
        const auto a=flag_continuation(worlds[world],choice.intercept,choice.flag,rules,weights,common,response==1);
        const auto b=flag_continuation(worlds[world],choice.advance,choice.flag,rules,weights,common,response==1);
        std::cout << (sample ? ",[" : "["); print_flag_rollout(a); std::cout << ',';
        print_flag_rollout(b); std::cout << ']';
      }
      std::cout << "]}";
    }
  std::cout << "]}";
}

inline void flag_interception(const Rules& rules,uint64_t seed,int learner,int variant,
                              const ShortWeights& weights) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const auto first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
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
      const auto choices=public_flag_threats(d,moved,rules);
      for (size_t kind=0; kind<choices.size(); ++kind) {
        if (found[kind] || !choices[kind]) continue;
        found[kind]=true;
        if (comma) std::cout << ',';
        comma=true;
        report_flag_pairs(initial,state,rules,history,*choices[kind],weights,seed,kind);
      }
    }
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    history.push_back(action);
    const auto event=step(state,action,rules);
    observe_movement(moved,event,observe(state,learner));
  }
  std::cout << "],\"source_score\":" << review_outcome(state.result,learner) << "}\n";
}
}
