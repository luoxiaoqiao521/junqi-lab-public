#pragma once
#include "../sim/regional_arrival.h"

namespace jq::archive_probe {

inline std::optional<Move> visible_flag_attack(const SeatDecision& d) {
  for (auto move : d.actions)
    if (d.observation.rank_at(move.to)==Flag) return move;
  return std::nullopt;
}

inline bool projected_flag_attack(const Observation& view, Move quiet) {
  SeatDecision advanced{project_quiet(view,quiet),{}};
  gen_moves(advanced.observation,view.viewer,advanced.actions);
  return visible_flag_attack(advanced).has_value();
}

struct FlagCondition {
  int code; // 0 ineligible, 1 both absent, 2 creates an attack, 3 attack already available
  Move candidate;
};

inline FlagCondition screen_flag_condition(const SeatDecision& d, Move back) {
  const auto& o=d.observation;
  if (o.seat_at(back.to)>=0 || std::find(d.actions.begin(),d.actions.end(),back)==d.actions.end())
    return {0,{}};
  bool seen_flag=false;
  for (int n=0;n<kNodes;++n)
    if (o.rank_at(n)==Flag && o.team(o.seat_at(n))!=o.team(o.viewer)) seen_flag=true;
  if (!seen_flag) return {0,{}};
  if (visible_flag_attack(d) || projected_flag_attack(o,back)) return {3,{}};
  std::optional<Move> near,positive;
  for (auto move : d.actions) {
    if (move.from!=back.from || move==back || o.seat_at(move.to)>=0 ||
        short_features(o,move)!=short_features(o,back)) continue;
    auto& chosen=projected_flag_attack(o,move)?positive:near;
    if (!chosen || move.to<chosen->to) chosen=move;
  }
  if (positive) return {2,*positive};
  if (near) return {1,*near};
  return {0,{}};
}

inline void verify_flag_condition() {
  SeatDecision d{};
  auto& o=d.observation;
  std::fill_n(o.seats,kNodes,static_cast<int8_t>(kNone));
  std::fill_n(o.ranks,kNodes,static_cast<int8_t>(Unknown));
  const int flag=Board::get().seat_node(1,5,1);
  o.seats[flag]=1; o.ranks[flag]=Flag;
  for (int from=0;from<kNodes;++from) {
    if (from==flag) continue;
    o.seats[from]=0; o.ranks[from]=Marshal;
    gen_moves(o,0,d.actions);
    for (auto back:d.actions) {
      if (screen_flag_condition(d,back).code!=2) continue;
      o.ranks[flag]=Unknown;
      expect_value(screen_flag_condition(d,back).code==0,"hidden flag triggered the public condition");
      return;
    }
    o.seats[from]=kNone; o.ranks[from]=Unknown;
  }
  expect_value(false,"missing public flag-attack positive control");
}

inline std::pair<int,bool> flag_plan_trial(State state, Move first, const Rules& rules,
                                         const ShortWeights& weights, int variant,
                                         uint64_t seed, bool plan) {
  const auto learner=state.to_move;
  auto streams=review_streams(seed);
  bool next=true, used=false;
  step(state,first,rules);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    auto move=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    if (state.to_move==learner && next) {
      next=false;
      const auto attack=visible_flag_attack(d);
      if (plan && attack) { move=*attack; used=true; }
    }
    step(state,move,rules);
  }
  return {review_outcome(state.result,learner),used};
}

inline void print_flag_condition(const State& state,const SeatDecision& d,Move back,
                                 const Rules& rules,const ShortWeights& weights,
                                 const std::string& key,int ordinal) {
  const auto test=screen_flag_condition(d,back);
  std::cout<<"Q "<<key<<' '<<test.code<<' '<<int(test.candidate.from)<<' '
           <<int(test.candidate.to)<<' '<<int(back.from)<<' '<<int(back.to)<<'\n';
  if (test.code!=1 && test.code!=2) return;
  for (int variant:{0,2}) for (int sample=0;sample<12;++sample) {
    const auto seed=mix_seed(886301+static_cast<uint64_t>(ordinal),static_cast<uint64_t>(sample));
    for (int plan:{0,1}) {
      const auto a=flag_plan_trial(state,test.candidate,rules,weights,variant,seed,plan==1);
      const auto b=flag_plan_trial(state,back,rules,weights,variant,seed,plan==1);
      std::cout<<"F "<<key<<' '<<variant<<' '<<sample<<' '<<plan<<' '<<a.first<<' '
               <<b.first<<' '<<a.second<<' '<<b.second<<'\n';
    }
  }
}

}
