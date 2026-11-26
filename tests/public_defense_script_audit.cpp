#pragma warning(push)
#pragma warning(disable:4715)
#define main legacy_defense_audit_main
#include "deadline_opening.cpp"
#undef main
#pragma warning(pop)

#define main legacy_practice_window_main
#include "practice_flag_save_window.cpp"
#undef main

namespace {
struct DefenseRequest {
  int ply, actor, from, to, flag, depth;
};

struct DefenseCase {
  DarkOpeningPath path;
  SeatDecision decision;
  OpeningMarks evidence;
  Move actual, proposed;
  int flag, depth;

  DefenseCase(DarkOpeningPath prefix,Move recorded,const DefenseRequest& in)
      : path(std::move(prefix)),decision(seat_decision(path.game.state)),
        evidence(path.game.reply_history.for_actor(decision.observation,path.game.marks[in.actor])),
        actual(recorded),proposed{uint8_t(in.from),uint8_t(in.to)},flag(in.flag),depth(in.depth) {}
};

DefenseRequest read_request() {
  DefenseRequest request{};
  std::string verb,extra;
  expect_value(bool(std::cin>>verb>>request.ply>>request.actor>>request.from>>request.to>>request.flag>>request.depth)
               && verb=="assess-defense","bad request");
  expect_value(request.ply>=1 && request.ply<=800 && request.actor>=0 && request.actor<kSeats &&
               std::min({request.from,request.to,request.flag})>=0 &&
               std::max({request.from,request.to,request.flag})<kNodes &&
               request.depth>=1 && request.depth<=4 && !(std::cin>>extra) && std::cin.eof(),"request out of bounds");
  return request;
}

DefenseCase restore_case(const char* file,const DefenseRequest& request,Rules& rules) {
  const auto body=read_text(file);
  const auto header=parse_json(body.substr(0,body.find('\n')));
  const auto start=replay_initial(header,rules);
  expect_value(header.at("rules").at("id").str()=="flag-practice-1","unsupported trace profile");
  DarkOpeningPath prefix(start,request.actor);
  std::optional<Move> actual;
  replay(file,false,[&](const State& before,const StepInfo& event,const State&) {
    if (before.ply+1<request.ply) prefix.play(event.move,request.actor,rules);
    if (before.ply+1==request.ply) {
      expect_value(before.to_move==request.actor && state_hash(prefix.game.state)==state_hash(before),"stale decision");
      actual=event.move;
    }
  });
  expect_value(actual.has_value(),"decision not found");
  DefenseCase root(std::move(prefix),*actual,request);
  const auto& v=root.decision.observation;
  const int owner=v.seat_at(request.flag);
  expect_value(owner>=0 && v.team(owner)==v.team(v.viewer) && v.rank_at(request.flag)==Flag,
               "flag is not visible and friendly");
  expect_value(std::find(root.decision.actions.begin(),root.decision.actions.end(),root.proposed)!=
               root.decision.actions.end(),"illegal submitted action");
  return root;
}

Defense local_label(State state,Move move,const DefenseCase& root,const Rules& rules) {
  const int owner=root.decision.observation.seat_at(root.flag);
  step(state,move,rules);
  int left=1800;
  return defend(state,root.flag,owner,root.depth-1,rules,left);
}

void verify_public(const DefenseCase& root,const EarlyPrefix& other,const Rules& rules) {
  const auto d=seat_decision(other.state);
  const int actor=d.observation.viewer;
  const auto evidence=other.reply_history.for_actor(d.observation,other.marks[actor]);
  expect_value(same_observation(d.observation,root.decision.observation) && d.actions==root.decision.actions &&
               budget_candidates(d,root.actual,rules,4)==budget_candidates(root.decision,root.actual,rules,4),
               "public input or order changed");
  for (const auto move:{root.actual,root.proposed})
    expect_value(student_features(d,evidence,move,&rules)==student_features(root.decision,root.evidence,move,&rules),
                 "44 input values changed");
}

void emit_result(const DefenseCase& root,const State& state,const Rules& rules,int label,const char* tag) {
  expect_value(is_legal(state,root.actual) && is_legal(state,root.proposed),"fixed comparison became illegal");
  auto after=state;
  const auto result=step(after,root.proposed,rules);
  if (label) expect_value(int(result.outcome)==label,"witness missed requested collision");
  std::cout<<tag<<' '<<int(result.outcome)<<' '<<int(local_label(state,root.actual,root,rules))<<' '
           <<int(local_label(state,root.proposed,root,rules))<<' '<<state_hash(state)<<'\n';
}

struct TerminalResult { int score, plies; uint64_t hash; };

TerminalResult finish_defense_script(State state,Move first,int viewer,int desk,const Rules& rules,
                                     std::array<Rng,kSeats> streams) {
  const int start=state.ply;
  step(state,first,rules);
  while (state.result==Result::Ongoing && state.ply-start<360) {
    const int actor=state.to_move;
    const bool alternate_enemy=desk==1 && team_of(actor)!=team_of(viewer);
    const auto policy=alternate_enemy?DecisionKind::CollisionHeuristic:DecisionKind::FlagRoute;
    Move action{};
    expect_value(DecisionScorer(observe(state,actor),rules,policy).choose(streams[actor],action) &&
                 is_legal(state,action),"terminal continuation produced an illegal action");
    step(state,action,rules);
  }
  const auto score=team_score(state.result,viewer,state.match);
  return {score?int(*score*2):-1,state.ply-start,state_hash(state)};
}

void terminal_pairs(const DefenseCase& root,const State& state,const Rules& rules) {
  const auto& view=root.decision.observation;
  const auto salt=public_rollout_seed(view,root.evidence,view.quiet_clock);
  for (int desk:{0,1}) for (int trial:{0,1,2,3}) {
    const auto streams=review_streams(mix_seed(salt,40881+trial));
    const auto recorded=finish_defense_script(state,root.actual,view.viewer,desk,rules,streams);
    const auto proposal=finish_defense_script(state,root.proposed,view.viewer,desk,rules,streams);
    if (root.actual==root.proposed)
      expect_value(recorded.score==proposal.score && recorded.hash==proposal.hash && recorded.plies==proposal.plies,
                   "same decision did not produce the same game");
    std::cout<<"T "<<state_hash(state)<<' '<<desk<<' '<<trial<<' '<<recorded.score<<' '
             <<proposal.score<<' '<<recorded.plies<<' '<<proposal.plies<<' '
             <<recorded.hash<<' '<<proposal.hash<<'\n';
  }
}

void witness(const DefenseCase& root,const Rules& rules,int label,bool terminal) {
  const auto& view=root.decision.observation;
  const auto to=root.proposed.to;
  const auto support=compatible_rank_mask(view,root.evidence,to);
  unsigned ranks=0;
  for (int rank=0;rank<RankCount;++rank)
    if ((support&(1u<<rank)) && int(resolve(view.rank_at(root.proposed.from),rank,rules))==label) ranks|=1u<<rank;
  if (!ranks) {
    std::cout<<"S "<<label<<" 0 0 0 0 0 0\n";
    return;
  }
  std::array<unsigned,kNodes> pins{};
  pins[to]=ranks;
  int accepted=0, unresolved=0, failed=0, contradicted=0;
  unsigned nodes=0;
  const auto seed=public_rollout_seed(view,root.evidence,view.quiet_clock);
  std::vector<uint64_t> hashes;
  for (int i=0;i<5 && accepted<2;++i) {
    WitnessOpening layout{};
    Rng rng(mix_seed(seed,311007+label*31+i));
    CaptureSearch work{2048,&rng,&layout};
    const auto& game=root.path.game;
    const auto status=game.reply_history.solve_candidate(view,game.marks[view.viewer],root.path.observations.front(),pins,work);
    nodes+=2048-work.remaining;
    if (status==CaptureCompatibility::Contradicted) { ++contradicted; break; }
    if (status==CaptureCompatibility::Unresolved) { ++unresolved; continue; }
    const auto alternate=replay_dark_opening(root.path,view.viewer,rules,layout);
    if (!alternate) { ++failed; continue; }
    const auto hash=state_hash(alternate->state);
    if (std::find(hashes.begin(),hashes.end(),hash)!=hashes.end()) continue;
    hashes.push_back(hash);
    verify_public(root,*alternate,rules);
    emit_result(root,alternate->state,rules,label,"W");
    if (terminal) terminal_pairs(root,alternate->state,rules);
    ++accepted;
  }
  std::cout<<"S "<<label<<' '<<ranks<<' '<<accepted<<' '<<unresolved<<' '<<failed<<' '
           <<contradicted<<' '<<nodes<<'\n';
}
}

int main(int argc,char** argv) {
  try {
    const bool terminal=argc==3 && std::string(argv[2])=="terminal";
    expect_value(argc==2 || terminal,"expected one fixed replay");
    const auto request=read_request();
    Rules rules;
    const auto root=restore_case(argv[1],request,rules);
    const auto choices=budget_candidates(root.decision,root.actual,rules,4);
    std::cout<<"D "<<request.ply<<' '<<request.actor<<' '<<early_action_key(root.actual)<<' '
             <<early_action_key(root.proposed)<<' '<<root.decision.actions.size()<<' '
             <<(std::find(choices.begin(),choices.end(),root.proposed)!=choices.end())<<'\n';
    emit_result(root,root.path.game.state,rules,0,"O");
    if (terminal) terminal_pairs(root,root.path.game.state,rules);
    const auto& view=root.decision.observation;
    if (view.seat_at(root.proposed.to)>=0 && view.rank_at(root.proposed.to)==Unknown)
      for (int label:{1,2,3}) witness(root,rules,label,terminal);
    return 0;
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
  }
}
