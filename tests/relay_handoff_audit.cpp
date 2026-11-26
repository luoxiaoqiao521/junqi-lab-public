#pragma warning(push)
#pragma warning(disable:4715)
#define main legacy_deadline_opening_main
#include "deadline_opening.cpp"
#undef main
#pragma warning(pop)

namespace {
bool contact(const State& state, int receiver, Move move, int source_id, int target_id) {
  if (state.result!=Result::Ongoing || state.out[receiver] || state.id_at[move.from]!=source_id ||
      state.id_at[move.to]!=target_id || state.seat_at[move.from]!=receiver ||
      state.seat_at[move.to]<0 || team_of(state.seat_at[move.to])==team_of(receiver)) return false;
  std::vector<Move> legal;
  gen_piece_moves(observe(state,receiver,IdentityVisibility::AllIdentities),receiver,move.from,legal);
  return std::find(legal.begin(),legal.end(),move)!=legal.end();
}

Move frozen_move(const State& state,const Rules& rules,int receiver,int table,Rng& rng) {
  const auto view=observe(state,state.to_move,IdentityVisibility::AllIdentities);
  const bool changed=table==1 ? state.to_move==receiver : table==2 && team_of(state.to_move)!=team_of(receiver);
  if (!changed) return route_baseline(state,rules,rng);
  Move move{};
  single_case_check(DecisionScorer(view,rules,DecisionKind::CollisionHeuristic).choose(rng,move) &&
                    is_legal(state,move),"comparison policy returned an illegal action");
  return move;
}

struct TeamResult {int score, steps; uint64_t hash;};
TeamResult finish(State state,const Rules& rules,int receiver,int table,Move first,std::array<Rng,kSeats> rng) {
  const auto before=state.ply;
  step(state,first,rules);
  while (state.result==Result::Ongoing && state.ply-before<360) {
    const auto action=frozen_move(state,rules,receiver,table,rng[state.to_move]);
    step(state,action,rules);
  }
  const auto score=team_score(state.result,receiver);
  return {score?int(*score*2):-1,int(state.ply-before),state_hash(state)};
}

void emit_pairs(const std::string& key,const State& root,Move target,const Rules& rules,const ShortWeights& peer) {
  const int actor=root.to_move;
  const auto decision=compare_visibility(root,IdentityVisibility::AllIdentities);
  const OpeningMarks marks(decision.observation,false);
  const PublicReplyHistory history(root,false);
  for (int table:{0,1,2}) for (int rep:{0,1,2,3}) {
    const uint64_t seed=mix_seed(public_rollout_seed(decision.observation,marks,decision.observation.quiet_clock),
                                 317021+rep);
    auto rng=review_streams(seed);
    const auto ordinary=frozen_move(root,rules,actor,table,rng[actor]);
    const auto list=budget_candidates(decision,ordinary,rules,4);
    const bool nominated=std::find(list.begin(),list.end(),target)!=list.end();
    const auto baseline=finish(root,rules,actor,table,ordinary,rng);
    const auto forced=finish(root,rules,actor,table,target,rng);
    const auto estimate=[&](Move move) {
      ResponseValue value;
      ResponseAudit effort{6,0,0,false,true};
      for (int i:{0,1,2,3})
        response_trial(root,move,marks,history,rules,peer,i%2,mix_seed(seed,i),value,false,
                       RoutePriority::Old,true,&effort);
      single_case_check(effort.plies<=24,"paired teacher exceeded its action budget");
      return std::pair<double,int>{value.value,effort.plies};
    };
    const auto left=estimate(ordinary),right=estimate(target);
    if (ordinary==target)
      single_case_check(baseline.hash==forced.hash && baseline.score==forced.score &&
                        left==right,"same-action continuation or estimate changed");
    std::cout<<"P "<<key<<' '<<table<<' '<<rep<<' '<<early_action_key(ordinary)<<' '
             <<early_action_key(target)<<' '<<nominated<<' '<<baseline.score<<' '<<forced.score<<' '
             <<baseline.steps<<' '<<forced.steps<<' '<<baseline.hash<<' '<<forced.hash<<' '
             <<std::setprecision(17)<<left.first<<' '<<right.first<<' '<<left.second<<' '<<right.second<<'\n';
  }
}

struct HandoffWindow {
  const ShortWeights& peer;
  std::string key;
  int open_ply, handoff, receiver;
  Move opening, target;
  int source_id=-1,target_id=-1;
  int examined=0,stays=0,blocked=0,changed=0,skipped=0;
  bool created=false,reported=false;

  void examine_open(const State& before,Move human,const Rules& rules) {
    single_case_check(human==opening && receiver==(before.to_move+2)%kSeats,"fixed opening mismatch");
    source_id=before.id_at[target.from];
    target_id=before.id_at[target.to];
    single_case_check(before.seat_at[target.from]==receiver && target_id>=0,"missing fixed route pieces");
    auto after=before;
    const auto event=step(after,human,rules);
    created=event.outcome==Outcome::Move && !contact(before,receiver,target,source_id,target_id) &&
            contact(after,receiver,target,source_id,target_id);
    if (!created || after.to_move==receiver || after.result!=Result::Ongoing) return;
    single_case_check(team_of(after.to_move)!=team_of(receiver),"unexpected intermediate acting seat");
    std::vector<Move> replies;
    legal_moves(after,replies);
    for (const auto reply:replies) {
      auto next=after;
      step(next,reply,rules);
      ++examined;
      if (next.result!=Result::Ongoing || next.to_move!=receiver) ++skipped;
      else if (next.id_at[target.from]!=source_id || next.id_at[target.to]!=target_id) ++changed;
      else if (contact(next,receiver,target,source_id,target_id)) ++stays;
      else ++blocked;
    }
  }

  void inspect(const State& root,Move human,const Rules& rules) {
    const int ply=root.ply+1;
    if (ply==open_ply) examine_open(root,human,rules);
    if (ply!=handoff) return;
    reported=true;
    const bool reachable=created && root.to_move==receiver && contact(root,receiver,target,source_id,target_id);
    const bool target_changed=target_id!=root.id_at[target.to];
    int outcome=-1;
    if (reachable) {
      auto branch=root;
      const auto event=step(branch,target,rules);
      outcome=int(event.outcome);
    }
    std::cout<<"H "<<key<<' '<<created<<' '<<reachable<<' '<<(reachable && human==target)<<' '
             <<target_changed<<' '<<outcome<<' '<<examined<<' '<<stays<<' '<<blocked<<' '
             <<changed<<' '<<skipped<<' '<<state_hash(root)<<'\n';
    if (reachable) emit_pairs(key,root,target,rules,peer);
  }
};
}

int main(int argc,char** argv) {
  try {
    single_case_check(argc==2,"expected frozen training rules");
    const auto config=read_text(argv[1]);
    Rules rules;
    std::string error;
    single_case_check(parse_rules(config,rules,error),"invalid research rules");
    ShortWeights peer{};
    for (auto& weight:peer)
      single_case_check(bool(std::cin>>weight) && std::isfinite(weight),"invalid frozen teacher weights");
    run_team_relay(rules);
    std::string key,tape;
    int ccw,open_ply,handoff,receiver,from,to,src,dst;
    while (std::cin>>key>>ccw>>open_ply>>handoff>>receiver>>from>>to>>src>>dst>>tape) {
      single_case_check(open_ply>0 && handoff==open_ply+2 && handoff<=280 &&
                        receiver>=0 && receiver<4 && std::min({from,to,src,dst})>=0 &&
                        std::max({from,to,src,dst})<kNodes,"invalid declared route window");
      Rules archive;
      single_case_check(parse_rules(archive_review_rules(config,ccw!=0),archive,error),"bad archive profile");
      auto local=rules;
      local.ccw=ccw!=0;
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      HandoffWindow window{peer,key,open_ply,handoff,receiver,{uint8_t(from),uint8_t(to)},{uint8_t(src),uint8_t(dst)}};
      const auto confirmed=scan_jgs_pressure(window,ArchiveRows(decode_archive_tape(tape)),archive,local,handoff);
      std::cout<<"A "<<key<<' '<<confirmed.agrees<<' '<<window.reported<<' '<<confirmed.reason<<'\n';
    }
    single_case_check(std::cin.eof(),"incomplete declared windows");
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
