#include "../sim/regional_arrival.h"
#include "archive_replay_prefix.h"
#include "public_defender_probe.h"
#include <bitset>
#include <iomanip>
#include <set>

using namespace jq;
using namespace jq::archive_probe;

struct Route { Move start, next; };

static std::bitset<kNodes> target_gates(const Observation& view) {
  const auto& board=Board::get();
  std::bitset<kNodes> gates;
  for(int seat=0;seat<4;++seat) {
    if(view.team(seat)==view.team(view.viewer)) continue;
    int known=kNone;
    for(int col:{1,3}) {
      const int hq=board.seat_node(seat,5,col);
      if(view.rank_at(hq)==Flag) known=hq;
    }
    for(int col:{1,3}) {
      const int hq=board.seat_node(seat,5,col);
      if(view.seat_at(hq)!=seat || (known!=kNone && hq!=known) ||
         (view.rank_at(hq)!=Unknown && view.rank_at(hq)!=Flag)) continue;
      gates.set(hq);
      gates.set(board.seat_node(seat,4,col));
    }
  }
  return gates;
}

static std::vector<Route> routes(const SeatDecision& d) {
  const auto gates=target_gates(d.observation);
  std::vector<Route> output;
  if(gates.none()) return output;
  std::array<std::bitset<kNodes>,kNodes> old;
  for(const auto action:d.actions) if(gates[action.to]) old[action.from].set(action.to);
  for(const auto action:d.actions) {
    if(d.observation.seat_at(action.to)!=kNone || gates[action.to]) continue;
    const auto after=project_quiet(d.observation,action);
    std::vector<Move> legal;
    gen_piece_moves(after,after.viewer,action.to,legal);
    for(const auto follow:legal)
      if(gates[follow.to] && !old[action.from][follow.to]) {
        output.push_back({action,follow});
        break;
      }
  }
  return output;
}

struct ReplyPoint { State state; EndReason end_reason; int learner_retired; };

static ReplyPoint after_reply(State state,Move start,const Rules& rules,const ShortWeights& weights,
                              int response,uint64_t seed) {
  const int learner=state.to_move;
  auto streams=review_streams(seed);
  auto event=step(state,start,rules);
  int retired=state.out[learner] ? state.ply : 0;
  while(state.result==Result::Ongoing && state.to_move!=learner) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,response,true,weights,streams[state.to_move]);
    event=step(state,action,rules);
    if(!retired && state.out[learner]) retired=state.ply;
  }
  return {state,event.end_reason,retired};
}

static const char* follow_status(const ReplyPoint& point,const Route& route,int learner,
                                 bool legal,bool has_fallback) {
  const auto& state=point.state;
  if(state.result!=Result::Ongoing) {
    if(point.learner_retired && point.learner_retired<state.ply) return "learner_retired_before_terminal";
    if(state.result==Result::Truncated) return "early_cutoff";
    if(state.result==Result::Draw) return "early_draw";
    return review_outcome(state.result,learner)==2 ? "early_win" : "early_loss";
  }
  if(legal) return has_fallback ? "both_actions_available" : "no_alternative_action";
  if(state.seat_at[route.next.from]!=learner) return "route_piece_removed";
  const int target=state.seat_at[route.next.to];
  if(target>=0 && team_of(target)==team_of(learner)) return "target_occupied_by_team";
  return "route_no_longer_reachable";
}

static const char* follow_target(const State& state,int learner,Move follow,bool legal) {
  if(!legal) return "unavailable";
  const auto view=observe(state,learner);
  if(!target_gates(view)[follow.to]) return "outside_current_gate";
  if(view.rank_at(follow.to)==Flag) return "revealed_flag_contact";
  return view.seat_at(follow.to)<0 ? "quiet_gate_arrival" : "other_contact_at_gate";
}

static void sample(const std::string& group,const State& s,const Rules& rules,const ShortWeights& weights,
                   const Route& proposed,Move alternative,bool audit) {
  const int learner=s.to_move;
  Rules trial=rules;
  trial.truncation_cap=std::min(trial.truncation_cap,s.ply+240);
  for(int response:{0,2}) for(int i=0;i<16;++i) {
    const auto seed=mix_seed(515499+static_cast<uint64_t>(s.ply),i);
    const auto actual=review_rollout(s,proposed.start,trial,response,true,weights,seed);
    const auto other=review_rollout(s,alternative,trial,response,true,weights,seed);
    const auto later=after_reply(s,proposed.start,trial,weights,response,seed);
    const bool legal=later.state.result==Result::Ongoing && is_legal(later.state,proposed.next);
    int forward=-2, switch_score=-2;
    if(legal) {
      const auto d=seat_decision(later.state);
      const auto weights_for_moves=review_learned(d,weights);
      double largest=-1;
      Move alternate=proposed.next;
      for(size_t j=0;j<d.actions.size();++j) if(!(d.actions[j]==proposed.next) && weights_for_moves[j]>largest) {
        alternate=d.actions[j];
        largest=weights_for_moves[j];
      }
      if(!(alternate==proposed.next)) {
        forward=review_outcome(review_rollout(later.state,proposed.next,trial,response,true,weights,seed),learner);
        switch_score=review_outcome(review_rollout(later.state,alternate,trial,response,true,weights,seed),learner);
      }
    }
    std::cout<<"P "<<group<<' '<<s.ply+1<<' '<<response<<' '<<i<<' '
             <<review_outcome(actual,learner)<<' '<<review_outcome(other,learner)<<' '
             <<legal<<' '<<forward<<' '<<switch_score<<'\n';
    if(audit) std::cout<<"T "<<group<<' '<<s.ply+1<<' '<<response<<' '<<i<<' '
      <<follow_status(later,proposed,learner,legal,forward!=-2)<<' '
      <<follow_target(later.state,learner,proposed.next,legal)<<' '
      <<int(later.end_reason)<<' '<<later.learner_retired<<' '<<later.state.ply<<' '
      <<review_outcome(later.state.result,learner)<<'\n';
  }
}

static void audit_ranking(const std::string& group,const SeatDecision& d,const std::vector<double>& values,
                          const std::vector<Route>& options,double best) {
  if(options.empty()) return;
  double mass=0;
  int higher=0, tied=0, non_route_ties=0, best_stable=static_cast<int>(d.actions.size());
  for(size_t i=0;i<d.actions.size();++i) {
    const bool route=std::any_of(options.begin(),options.end(),[&](const Route& r){return r.start==d.actions[i];});
    if(route) mass+=values[i];
    if(values[i]>best) ++higher;
    if(values[i]==best) {
      if(route) best_stable=std::min(best_stable,tied);
      else ++non_route_ties;
      ++tied;
    }
  }
  std::cout<<"K "<<group<<' '<<d.observation.ply+1<<' '<<higher<<' '<<tied<<' '
           <<non_route_ties<<' '<<higher+best_stable<<' '<<std::setprecision(17)<<mass<<'\n';
}

static bool inspect(const std::string& group,const State& state,const Rules& rules,const ShortWeights& weights,
                    Move human,bool can_sample,const DefenderEvidence* evidence,bool audit) {
  const auto d=seat_decision(state);
  const auto values=review_learned(d,weights);
  const auto options=routes(d);
  size_t choose=d.actions.size();
  const auto first=static_cast<size_t>(std::max_element(values.begin(),values.end())-values.begin());
  int rank=0;
  bool chosen=false;
  if(!options.empty()) {
    double best=-1;
    for(size_t i=0;i<options.size();++i) {
      const auto action=options[i].start;
      chosen=chosen || action==human;
      const auto index=static_cast<size_t>(std::find(d.actions.begin(),d.actions.end(),action)-d.actions.begin());
      if(values[index]>best) { choose=i; best=values[index]; }
    }
    rank=static_cast<int>(std::count_if(values.begin(),values.end(),[&](double value){return value>best;}));
    if(audit) audit_ranking(group,d,values,options,best);
  }
  const bool assessed=can_sample && !options.empty() && !(options[choose].start==d.actions[first]);
  std::cout<<"D "<<group<<' '<<state.ply+1<<' '<<d.actions.size()<<' '<<options.size()<<' '
           <<chosen<<' '<<rank<<' '<<assessed<<'\n';
  if(evidence) {
    evidence_window(group,state,rules,*evidence);
    if(assessed) {
      probe_defense(group,state,rules,weights,*evidence,options[choose].start,0);
      probe_defense(group,state,rules,weights,*evidence,d.actions[first],1);
    }
  } else if(assessed) sample(group,state,rules,weights,options[choose],d.actions[first],audit);
  return assessed;
}

int main(int argc,char** argv) {
  const bool defenders=argc==2 && std::string(argv[1])=="defend";
  const bool audit=argc==2 && std::string(argv[1])=="audit-timing";
  if(defenders) verify_threat_fixture();
  int count;
  ShortWeights weights{};
  if(!(std::cin>>count)||count<1) return 2;
  for(int i=0;i<5;++i) if(!(std::cin>>weights[i])||!std::isfinite(weights[i])) return 2;
  std::set<std::string> sampled;
  std::array<int,2> budget{12,12};
  for(int i=0;i<count;++i) {
    std::string group;
    int qq,ply,n;
    if(!(std::cin>>group>>qq>>ply>>n)||n<0||qq<0||qq>1) return 2;
    int8_t layouts[4][25]{};
    if(!read_layouts(layouts)) return 2;
    std::vector<Row> rows(static_cast<size_t>(n));
    for(auto& row:rows) if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return 2;
    Rules rules;
    rules.ccw=false; rules.marshal_reveals_flag=qq!=0; rules.check_involved_no_move=qq!=0;
    rules.no_progress_limit=qq ? 71 : 0;
    State state{};
    MovedSquares marks{};
    Move human;
    DefenderEvidence evidence;
    ReplayObserver track;
    if(defenders) track=[&](const State& before,const StepInfo& event,const State& after) {
      evidence.track(before,event,after,rules);
    };
    if(!archive_prefix(state,marks,human,rules,layouts,rows,true,track) || state.ply+1!=ply) {
      if(audit && archive_prefix(state,marks,human,rules,layouts,rows,false) && state.ply+1==ply)
        std::cout<<"U "<<group<<' '<<ply<<'\n';
      std::cout<<"X "<<group<<' '<<ply<<' '<<(n==0)<<'\n';
      continue;
    }
    if(inspect(group,state,rules,weights,human,budget[qq]>0 && !sampled.count(group),
               defenders ? &evidence : nullptr,audit)) {
      --budget[qq];
      sampled.insert(group);
    }
  }
}
