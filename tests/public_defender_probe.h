#pragma once
#include "../sim/public_threat_evidence.h"
#include "single_rules_cases.h"

namespace jq::archive_probe {
struct DefenderEvidence {
  std::array<std::optional<OpeningMarks>,4> path{}, contact{};
  void track(const State& before,const StepInfo& event,const State& after,const Rules& rules) {
    for(int seat=0;seat<4;++seat) {
      const auto current=observe(before,seat), following=observe(after,seat);
      if(!path[seat]) {
        path[seat].emplace(current,true);
        contact[seat].emplace(current,true);
      }
      advance_public_ranks(*path[seat],current,following,event.move,event.outcome);
      advance_contact_ranks(*contact[seat],current,following,event.move,event.outcome,rules);
    }
  }
};

inline void verify_threat_fixture() {
  auto s=single_case_state();
  const auto& board=Board::get();
  const int flag=board.seat_node(0,5,1), landing=board.seat_node(0,4,1);
  const int origin=board.seat_node(0,4,0), guard=board.seat_node(0,3,1);
  expect_value(s.rank_at[flag]==Flag,"fixture flag at wrong coordinate");
  place_single_case_piece(s,1,landing,Mine);
  place_single_case_piece(s,2,guard,Platoon);
  place_single_case_piece(s,26,origin,Engineer);
  s.to_move=1;
  s.ply=0;
  s.no_progress=0;
  const Move approach{static_cast<uint8_t>(origin),static_cast<uint8_t>(landing)};
  const Move intercept{static_cast<uint8_t>(guard),static_cast<uint8_t>(landing)};
  const Rules rules;
  expect_value(is_legal(s,approach),"fixture approach is illegal");
  const auto before=s;
  const auto event=step(s,approach,rules);
  DefenderEvidence evidence;
  evidence.track(before,event,s,rules);
  const auto d=seat_decision(s);
  const auto fallback=std::find_if(d.actions.begin(),d.actions.end(),
                                  [&](Move a){return !(a==intercept);});
  expect_value(fallback!=d.actions.end() && is_legal(s,intercept),"fixture requires two defender options");
  const auto a=visible_flag_threat(d.observation,*evidence.path[0],rules);
  const auto b=visible_flag_threat(d.observation,*evidence.contact[0],rules);
  expect_value(compatible_rank_mask(d.observation,*evidence.contact[0],landing)==(1u<<Engineer),
               "public mine removal did not identify a surviving engineer");
  expect_value(!visible_intercept(d,*evidence.path[0],rules,*fallback,a) &&
               visible_intercept(d,*evidence.contact[0],rules,*fallback,b)==intercept,
               "collision evidence did not distinguish a viable guard");
  auto result=s;
  step(result,intercept,rules);
  expect_value(!result.alive[26],"verified interception did not remove the engineer");
  std::cout<<"F 1\n";
}

inline void evidence_window(const std::string& group,const State& s,const Rules& rules,
                            const DefenderEvidence& evidence) {
  if(!evidence.path[s.to_move]) return;
  const auto d=seat_decision(s);
  int changes=0, forced_slots=0;
  for(int n=0;n<kNodes;++n) {
    if(d.observation.seat_at(n)<0 || d.observation.rank_at(n)!=Unknown) continue;
    const auto a=evidence.path[s.to_move]->ranks[n], b=evidence.contact[s.to_move]->ranks[n];
    expect_value((b&(1u<<s.rank_at[n])) && (a&b)==b,"contact evidence excluded archived identity");
    changes+=a!=b;
    forced_slots+=a!=b && (b&(b-1))==0;
  }
  const auto a=visible_flag_threat(d.observation,*evidence.path[s.to_move],rules);
  const auto b=visible_flag_threat(d.observation,*evidence.contact[s.to_move],rules);
  std::cout<<"E "<<group<<' '<<s.ply+1<<' '<<changes<<' '<<forced_slots
           <<' '<<a.origins<<' '<<a.forced<<' '<<b.origins<<' '<<b.forced<<'\n';
}

inline int next_enemy_flag_attack(const State& s,int defender) {
  if(s.result!=Result::Ongoing || team_of(s.to_move)==team_of(defender)) return 0;
  std::vector<Move> moves;
  legal_moves(s,moves);
  return std::any_of(moves.begin(),moves.end(),[&](Move a){return s.rank_at[a.to]==Flag;});
}

inline int finish_defense(State s,Move first,const Rules& rules,const ShortWeights& weights,
                          std::array<Rng,4> streams,int attacker) {
  step(s,first,rules);
  while(s.result==Result::Ongoing) {
    const auto d=seat_decision(s);
    const auto action=review_move(d,attacker,2,true,weights,streams[s.to_move]);
    step(s,action,rules);
  }
  return review_outcome(s.result,attacker);
}

inline void probe_defense(const std::string& group,const State& root,const Rules& rules,
                          const ShortWeights& weights,const DefenderEvidence& original,Move start,int arm) {
  const int attacker=root.to_move;
  Rules trial=rules;
  trial.truncation_cap=std::min(trial.truncation_cap,root.ply+240);
  for(int seed=0;seed<8;++seed) {
    auto streams=review_streams(mix_seed(220471+root.ply,seed));
    auto s=root;
    auto evidence=original;
    Move last=start;
    while(s.result==Result::Ongoing && team_of(s.to_move)==team_of(attacker)) {
      const auto before=s;
      const auto info=step(s,last,trial);
      evidence.track(before,info,s,trial);
      if(s.result==Result::Ongoing && team_of(s.to_move)==team_of(attacker)) {
        const auto ally=seat_decision(s);
        last=review_move(ally,attacker,2,true,weights,streams[s.to_move]);
      }
    }
    std::cout<<"R "<<group<<' '<<root.ply+1<<' '<<seed<<' '<<arm;
    if(s.result!=Result::Ongoing) {
      const auto outcome=review_outcome(s.result,attacker);
      for(int profile=0;profile<3;++profile) std::cout<<' '<<outcome<<" 0 0";
    } else {
      const auto d=seat_decision(s);
      const int side=s.to_move;
      const auto fallback=review_move(d,attacker,2,true,weights,streams[side]);
      const auto path=visible_flag_threat(d.observation,*evidence.path[side],trial);
      const auto contact=visible_flag_threat(d.observation,*evidence.contact[side],trial);
      const std::array<Move,3> choices{fallback,
        visible_intercept(d,*evidence.path[side],trial,fallback,path).value_or(fallback),
        visible_intercept(d,*evidence.contact[side],trial,fallback,contact).value_or(fallback)};
      std::array<int,3> outcome{};
      for(size_t i=0;i<choices.size();++i) {
        auto after=s;
        step(after,choices[i],trial);
        if(i>0 && choices[i]==choices[0]) outcome[i]=outcome[0];
        else if(i==2 && choices[i]==choices[1]) outcome[i]=outcome[1];
        else outcome[i]=finish_defense(s,choices[i],trial,weights,streams,attacker);
        std::cout<<' '<<outcome[i]<<' '<<!(choices[i]==fallback)<<' '<<next_enemy_flag_attack(after,side);
      }
    }
    std::cout<<'\n';
  }
}
}
