#include "../sim/replay_deadline.h"
#include "layout.h"
#include <iostream>
#include <limits>
#include <string>

using namespace jq;

struct Intervention {
  State state;
  std::array<Rng,4> streams;
  Move shortlist, expanded;
  int kind;
};

struct Coverage {
  std::array<int,3> turns{}, win{}, threat{}, later{};
};

static Move respond(const SeatDecision& d,int learner,int table,const ShortWeights& parent,
                    const ShortWeights& historical,Rng& rng) {
  const int who=d.observation.viewer;
  if(who==learner) return short_move(d.observation,d.actions,parent,rng);
  if(table && who==(learner+2)%4) return short_move(d.observation,d.actions,historical,rng);
  return review_move(d,learner,table ? 2 : 0,false,parent,rng);
}

static int danger(State s,Move move,const Rules& rules) {
  const int team=team_of(s.to_move);
  step(s,move,rules);
  if(s.result!=Result::Ongoing || team_of(s.to_move)==team) return 0;
  std::vector<Move> replies;
  legal_moves(s,replies);
  for(const auto reply:replies)
    if(s.rank_at[reply.to]==Flag) return 1;
  return 0;
}

static std::pair<size_t,size_t> choices(const SeatDecision& d,const MovedSquares& moved,Move fallback,
                                       const Rules& rules,const ShortWeights& parent) {
  const auto base=static_cast<size_t>(std::find(d.actions.begin(),d.actions.end(),fallback)-d.actions.begin());
  auto preferences=review_learned(d,parent);
  preferences[base]=std::numeric_limits<double>::infinity();
  std::vector<FlagDeadline> limits;
  for(auto a:d.actions) limits.push_back(replay_flag_deadline(d.observation,moved,a,rules));
  const size_t expanded=replay_deadline_choice(limits,preferences);
  // The narrow arm checks the frozen draw and the two highest-ranked remaining moves.
  auto shortlist=limits;
  std::fill(shortlist.begin(),shortlist.end(),FlagDeadline{});
  shortlist[base]=limits[base];
  auto leftover=preferences;
  leftover[base]=-1.;
  for(int j=0;j<2 && j+1<static_cast<int>(d.actions.size());++j) {
    const auto next=static_cast<size_t>(std::max_element(leftover.begin(),leftover.end())-leftover.begin());
    shortlist[next]=limits[next];
    leftover[next]=-1.;
  }
  // Treat omitted actions as the fallback's deadline so they can never benefit from omission.
  for(size_t i=0;i<shortlist.size();++i) if(leftover[i]>=0.) shortlist[i]=limits[base];
  return {replay_deadline_choice(shortlist,preferences),expanded};
}

static int finish(State state,Move action,std::array<Rng,4> rng,const Rules& rules,
                  int learner,int table,const ShortWeights& parent,const ShortWeights& historical) {
  step(state,action,rules);
  while(state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    action=respond(d,learner,table,parent,historical,rng[state.to_move]);
    step(state,action,rules);
  }
  return review_outcome(state.result,learner);
}

static void report(const Coverage& coverage) {
  for(const auto values:{coverage.turns,coverage.win,coverage.threat,coverage.later})
    for(int value:values) std::cout<<' '<<value;
}

static void game(State s,const Rules& rules,const ShortWeights& parent,const ShortWeights& historical,
                 const std::string& key,int learner,int table,uint64_t seed,bool compare) {
  const auto before=s;
  auto streams=review_streams(seed);
  MovedSquares moved{};
  Coverage coverage;
  std::optional<Intervention> save;
  int possible=0,narrow_missed=0,lost_warning=0,verified=0;
  while(s.result==Result::Ongoing) {
    const int seat=s.to_move;
    const auto d=seat_decision(s);
    const auto action=respond(d,learner,table,parent,historical,streams[seat]);
    if(seat==learner) {
      const auto phase=s.ply<40?0:s.ply<120?1:2;
      const auto base=replay_flag_deadline(d.observation,moved,action,rules);
      const bool win=std::any_of(d.actions.begin(),d.actions.end(),[&](Move move) {
        return d.observation.rank_at(move.to)==Flag &&
               replay_flag_deadline(d.observation,moved,move,rules).team_win;
      });
      ++coverage.turns[phase];
      coverage.win[phase]+=win;
      coverage.threat[phase]+=base.enemy_turn==1;
      coverage.later[phase]+=base.enemy_turn>1 && base.enemy_turn<kSeats;
      if(!save && (win || (base.enemy_turn==1 && !base.uncertain_finish))) {
        ++possible;
        const auto pair=choices(d,moved,action,rules,parent);
        const auto expanded=d.actions[pair.second],narrow=d.actions[pair.first];
        if(!(expanded==action)) {
          narrow_missed=narrow==action;
          lost_warning=danger(s,action,rules)-danger(s,expanded,rules);
          State check=s;step(check,expanded,rules);
          if(win) {
            expect_value(team_score(check.result,seat)==1.,"public terminal move did not win");
            verified=1;
          }
          save=Intervention{s,streams,narrow,expanded,win?1:2};
        }
      }
    }
    const auto event=step(s,action,rules);
    observe_movement(moved,event,observe(s,learner));
  }
  const int baseline=review_outcome(s.result,learner);
  std::cout<<key<<' '<<learner<<' '<<table<<' '<<int(before.to_move)<<' '<<seed<<' '
           <<baseline<<' '<<s.ply<<' '<<(save?save->kind:0)<<' '<<possible<<' '
           <<narrow_missed<<' '<<lost_warning<<' '<<verified;
  report(coverage);
  if(compare) {
    int small=baseline,expanded=baseline;
    if(save) {
      expanded=finish(save->state,save->expanded,save->streams,rules,learner,table,parent,historical);
      if(save->shortlist==save->expanded) small=expanded;
      else if(!narrow_missed) small=finish(save->state,save->shortlist,save->streams,rules,
                                          learner,table,parent,historical);
    }
    std::cout<<' '<<small<<' '<<expanded;
  }
  std::cout<<'\n';
}

int main(int argc,char** argv) {
  int layouts_count;
  ShortWeights parent{},historical{};
  const bool compare=argc==2 && std::string(argv[1])=="compare";
  if(!(std::cin>>layouts_count)||layouts_count<1) return 2;
  for(auto weights:{&parent,&historical})
    for(size_t i=0;i<5;++i) if(!(std::cin>>(*weights)[i])||!std::isfinite((*weights)[i])) return 2;
  Rules rules;
  rules.ccw=false; rules.marshal_reveals_flag=true; rules.no_progress_limit=71;
  rules.check_involved_no_move=true;
  for(int i=0;i<layouts_count;++i) {
    std::string key;
    uint64_t seed;
    if(!(std::cin>>key>>seed)) return 2;
    int8_t layouts[4][25]{};
    if(key.rfind("generated-",0)==0) {
      Rng lr(mix_seed(seed,48));
      for(auto& seat:layouts) generate_layout(seat,lr);
    } else {
      for(auto& seat:layouts) for(auto& rank:seat) {
        int n;
        if(!(std::cin>>n)||n<0||n>=RankCount) return 2;
        rank=static_cast<int8_t>(n);
      }
    }
    for(const auto& seat:layouts) if(!valid_layout(seat)) return 3;
    for(int learner=0;learner<4;++learner) {
      State s;
      init_state(s,layouts,(learner+static_cast<int>(seed%4))%4,rules);
      for(int table=0;table<2;++table)
        game(s,rules,parent,historical,key,learner,table,mix_seed(seed,learner),compare);
    }
  }
}
