#include "single_rules_cases.h"
#include "../sim/exchange_counterexamples.h"
#include "../sim/public_threat_evidence.h"
#include "layout.h"
#include <iostream>

using namespace jq;

static Move play_turn(const SeatDecision& d,int learner,int table,const Rules& rules,
                      const ShortWeights& weights,Rng& stream) {
  const int actor=d.observation.viewer;
  const int enemy=next_seat(learner,rules);
  if(actor==learner || (table==2 && actor==enemy))
    return short_move(d.observation,d.actions,weights,stream);
  Move move;
  if(table==2 || (table==1 && actor==enemy)) {
    expect_value(heuristic_move(d.observation,stream,move),"heuristic without a legal action");
    return move;
  }
  expect_value(!d.actions.empty(),"random player without a legal action");
  return d.actions[stream.below(static_cast<uint32_t>(d.actions.size()))];
}

static void finish(State& state,int learner,int table,const Rules& rules,
                   const ShortWeights& weights,uint64_t seed) {
  auto streams=review_streams(seed);
  while(state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=play_turn(d,learner,table,rules,weights,streams[state.to_move]);
    step(state,action,rules);
  }
}

static int finish_with_partner(State state,int seat,int table,const Rules& rules,
                               const ShortWeights& weights,const ShortWeights& historical,
                               uint64_t seed,int& divergence) {
  auto streams=review_streams(seed);
  const int partner=(seat+2)%4;
  auto original_partner=streams[partner];
  while(state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto d=seat_decision(state);
    Move action;
    if(actor==partner) {
      action=short_move(d.observation,d.actions,historical,streams[actor]);
      if(!divergence && !(action==play_turn(d,seat,table,rules,weights,original_partner)))
        divergence=state.ply+1;
    } else action=play_turn(d,seat,table,rules,weights,streams[actor]);
    step(state,action,rules);
  }
  return review_outcome(state.result,seat);
}

static State position(int seat,int enemy,int flag_col,bool hidden) {
  State s{};
  std::fill(std::begin(s.id_at),std::end(s.id_at),int8_t(kNone));
  std::fill(std::begin(s.seat_at),std::end(s.seat_at),int8_t(kNone));
  std::fill(std::begin(s.rank_at),std::end(s.rank_at),int8_t(Unknown));
  const auto& board=Board::get();
  for(int player=0;player<kSeats;++player) s.out[player]=player!=seat && player!=enemy;
  const auto put=[&](int owner,int member,int area,int row,int col,int rank) {
    place_single_case_piece(s,owner*kPiecesPerSeat+member,board.seat_node(area,row,col),rank);
  };
  put(seat,0,seat,5,1,Flag);
  put(enemy,0,enemy,5,flag_col,Flag);
  if(hidden) put(enemy,3,enemy,5,4-flag_col,Platoon);
  put(seat,1,enemy,4,1,Engineer);
  put(seat,2,enemy,4,3,Engineer);
  put(enemy,1,seat,4,1,Engineer);
  put(enemy,2,enemy,0,0,Corps);
  s.to_move=static_cast<int8_t>(seat);
  return s;
}

static std::array<Move,2> hq_actions(int enemy) {
  const auto& b=Board::get();
  return {Move{uint8_t(b.seat_node(enemy,4,1)),uint8_t(b.seat_node(enemy,5,1))},
          Move{uint8_t(b.seat_node(enemy,4,3)),uint8_t(b.seat_node(enemy,5,3))}};
}

static void control_games(const Rules& rules,const ShortWeights& weights) {
  for(int seat=0;seat<kSeats;++seat) {
    const int enemy=next_seat(seat,rules);
    const auto moves=hq_actions(enemy);
    const auto hidden_left=position(seat,enemy,1,true);
    const auto hidden_right=position(seat,enemy,3,true);
    const auto lhs=seat_decision(hidden_left),rhs=seat_decision(hidden_right);
    expect_value(same_seat_view(hidden_left,hidden_right,seat) && lhs.actions==rhs.actions,
                 "hidden HQ swap changed player information");
    for(int hidden=0;hidden<2;++hidden) for(int world=0;world<2;++world) {
      const auto state=position(seat,enemy,1+2*world,hidden!=0);
      const auto view=observe(state,seat);
      expect_value(sole_enemy_hq(view,enemy,rules)==(hidden?kNone:moves[world].to),
                   "HQ evidence does not support the visible label");
      for(int table=0;table<3;++table) for(int run=0;run<24;++run) {
        const auto seed=mix_seed(839270+seat,run);
        std::cout<<"T "<<seat<<' '<<hidden<<' '<<world<<' '<<table<<' '<<run;
        for(int arm=0;arm<2;++arm) {
          auto trial=state;
          expect_value(is_legal(trial,moves[arm]),"tactical arm is illegal");
          step(trial,moves[arm],rules);
          if(arm==world) expect_value(team_score(trial.result,seat)==1.,"flag arm is not terminal");
          else expect_value(trial.result==Result::Ongoing,"decoy arm is accidentally terminal");
          finish(trial,seat,table,rules,weights,seed);
          std::cout<<' '<<review_outcome(trial.result,seat)<<' '<<trial.ply;
        }
        std::cout<<'\n';
      }
    }
  }
}

struct LateCounts {
  int turns=0, quiet=0, reverse=0, contact=0, safe=0, winning=0;
  int learner_quiet=0, learner_has_contact=0;
};

struct Offer {
  Move move;
  int kind;
};

static int opponent_mask(const Observation& o) {
  int enemies=0;
  for(int n=0;n<kNodes;++n) {
    const int owner=o.seat_at(n);
    if(owner>=0 && o.team(owner)!=o.team(o.viewer)) enemies|=1<<owner;
  }
  return enemies;
}

static std::optional<Offer> public_offer(const SeatDecision& d,const OpeningMarks& marks,
                                         const Rules& rules) {
  const auto& o=d.observation;
  const int foes=opponent_mask(o);
  std::optional<Offer> best;
  int best_score=0;
  for(const auto a:d.actions) {
    const int enemy=o.seat_at(a.to);
    if(enemy<0) continue;
    if(foes==(1<<enemy) && a.to==sole_enemy_hq(o,enemy,rules)) return Offer{a,2};
    if(Board::get().node(a.to).kind==NodeKind::HQ) continue;
    const auto outcomes=collision_outcome_mask(o.rank_at(a.from),compatible_rank_mask(o,marks,a.to),rules);
    if(outcomes!=(1u<<static_cast<unsigned>(Outcome::AttackerWins))) continue;
    const int score=score_move(o,a).total();
    if(!best || score>best_score) {
      best=Offer{a,1};
      best_score=score;
    }
  }
  return best;
}

static void compare_offer(const State& at,Move actual,Offer test,int block,int table,
                          const Rules& rules,const ShortWeights& weights,uint64_t seed,const ShortWeights* partner) {
  const int learner=at.to_move;
  std::cout<<"O "<<block<<' '<<table<<' '<<at.ply+1<<' '<<at.no_progress<<' '<<test.kind<<' '
           <<int(actual.from)<<' '<<int(actual.to)<<' '<<int(test.move.from)<<' '<<int(test.move.to)<<'\n';
  for(int run=0;run<12;++run) {
    std::cout<<"P "<<block<<' '<<table<<' '<<run;
    std::array<int,2> outcomes{},splits{};
    const std::array<Move,2> actions{actual,test.move};
    const auto continuation=mix_seed(seed,420+run);
    for(size_t k=0;k<actions.size();++k) {
      const auto action=actions[k];
      auto after=at;
      step(after,action,rules);
      if(k==1 && test.kind==2)
        expect_value(team_score(after.result,learner)==1.,"public finish lacked a team win");
      auto s=after;
      finish(s,learner,table,rules,weights,continuation);
      std::cout<<' '<<review_outcome(s.result,learner);
      if(partner) outcomes[k]=finish_with_partner(after,learner,table,rules,weights,
                                                  *partner,continuation,splits[k]);
    }
    std::cout<<'\n';
    if(partner) std::cout<<"X "<<block<<' '<<table<<' '<<run<<' '<<outcomes[0]<<' '
                         <<outcomes[1]<<' '<<splits[0]<<' '<<splits[1]<<'\n';
  }
}

static void survey(State& state,int block,int learner,int table,int clock_low,const Rules& rules,
                   const ShortWeights& weights,uint64_t seed,const ShortWeights* partner) {
  auto streams=review_streams(seed);
  std::vector<OpeningMarks> marks;
  for(int seat=0;seat<kSeats;++seat) marks.emplace_back(observe(state,seat),true);
  std::array<std::optional<Move>,4> last;
  LateCounts counts;
  bool tested=false;
  while(state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto d=seat_decision(state);
    const auto action=play_turn(d,learner,table,rules,weights,streams[actor]);
    const bool quiet=d.observation.seat_at(action.to)<0;
    if(state.no_progress>=clock_low && state.no_progress<clock_low+10) {
      ++counts.turns;
      counts.quiet+=quiet;
      counts.reverse+=quiet && last[actor] && action.from==last[actor]->to && action.to==last[actor]->from;
      const bool contact=std::any_of(d.actions.begin(),d.actions.end(),
          [&](Move a){return d.observation.seat_at(a.to)>=0;});
      counts.contact+=quiet && contact;
      counts.learner_quiet+=quiet && actor==learner;
      counts.learner_has_contact+=quiet && contact && actor==learner;
      const auto offer=quiet && contact ? public_offer(d,marks[actor],rules) : std::nullopt;
      counts.safe+=offer && offer->kind==1;
      counts.winning+=offer && offer->kind==2;
      if(!tested && actor==learner && offer) {
        compare_offer(state,action,*offer,block,table,rules,weights,seed,partner);
        tested=true;
      }
    }
    last[actor]=quiet ? std::optional<Move>{action} : std::nullopt;
    const auto before=state;
    const auto event=step(state,action,rules);
    for(int seat=0;seat<kSeats;++seat)
      advance_contact_ranks(marks[seat],observe(before,seat),observe(state,seat),action,event.outcome,rules);
  }
  std::cout<<"L "<<block<<' '<<table<<' '<<counts.turns<<' '<<counts.quiet<<' '<<counts.reverse<<' '
           <<counts.contact<<' '<<counts.safe<<' '<<counts.winning<<' '<<counts.learner_quiet<<' '
           <<counts.learner_has_contact<<'\n';
}
static bool read_layouts(int8_t (&layout)[kSeats][kPiecesPerSeat],int source,uint64_t seed) {
  Rng rng(mix_seed(seed,18));
  for(auto& row:layout) {
    if(source==0) generate_layout(row,rng);
    else for(auto& rank:row) {
      int r;
      if(!(std::cin>>r)||r<Flag||r>=RankCount) return false;
      rank=static_cast<int8_t>(r);
    }
    if(!valid_layout(row)) return false;
  }
  return true;
}

#include "selfplay_cycle_entry.h"

int main(int argc,char** argv) {
  try {
    const std::string mode=argc==3 ? argv[2] : "";
    const bool carryover=mode=="carryover";
    const bool actor_mode=mode=="cycle-actor";
    const bool local=mode=="cycle-local" || actor_mode;
    const bool cycle=mode=="cycle" || local;
    const bool stall=mode=="stall" || mode=="stall-mid" || carryover;
    if(argc!=2 && !stall && !cycle) return 2;
    if(cycle) check_public_repeat();
    if(local) check_local_cycle();
    Rules rules;
    std::string err;
    if(!parse_rules(read_text(argv[1]),rules,err)) throw std::runtime_error(err);
    expect_value(rules.eliminate_exhausted_movers && rules.flag_eliminates,"wrong selfplay rules");
    rules.truncation_cap=1600;
    std::cout<<"R "<<rules_hash(rules)<<'\n';
    int count;
    ShortWeights weights{};
    if(!(std::cin>>count)||count<1) return 2;
    for(size_t i=0;i<5;++i) if(!(std::cin>>weights[i])||!std::isfinite(weights[i])) return 2;
    ShortWeights partner{};
    if(carryover) for(size_t i=0;i<5;++i)
      if(!(std::cin>>partner[i])||!std::isfinite(partner[i])) return 2;
    for(int block=0;block<count;++block) {
      uint64_t seed;
      int source;
      if(!(std::cin>>source>>seed)||(source!=0 && source!=1)) return 2;
      int8_t layouts[kSeats][kPiecesPerSeat]{};
      if(!read_layouts(layouts,source,seed)) return 2;
      const int seat=block%4,first=(block/4)%4;
      for(int table=carryover?2:0;table<3;++table) {
        State state{};
        init_state(state,layouts,first,rules);
        if(cycle) {
          survey_cycles(state,block,seat,table,rules,weights,seed,local,actor_mode);
          if(block%96==0) {
            State check{};
            init_state(check,layouts,first,rules);
            finish(check,seat,table,rules,weights,seed);
            expect_value(state_hash(state)==state_hash(check),"cycle measurement changed base play");
          }
        } else if(stall) {
          const int clock_low=mode=="stall-mid" || carryover ? 30 : rules.no_progress_limit-10;
          survey(state,block,seat,table,clock_low,rules,weights,seed,carryover ? &partner : nullptr);
        }
        else finish(state,seat,table,rules,weights,seed);
        std::cout<<"G "<<block<<' '<<source<<' '<<seat<<' '<<first<<' '<<seed<<' '<<table<<' '
                 <<review_outcome(state.result,seat)<<' '<<state.ply<<'\n';
      }
    }
    if(!stall && !cycle) control_games(rules,weights);
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 3;
  }
}
