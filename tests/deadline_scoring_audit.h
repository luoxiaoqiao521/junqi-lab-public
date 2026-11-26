#pragma once

struct ScoringRoot {
  State state;
  std::array<Rng,kSeats> streams;
  Move fallback, reset, inherited;
  bool split;
};

static int scoring_result(const State& state,int viewer) {
  const auto score=team_score(state.result,viewer);
  return score?int(*score*2):-1;
}

static Move rival_probe(const State& state,Move fallback,const Rules& rules,int learner,
                        uint64_t seed,int& search_work) {
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  auto choice=fallback;
  int best=3;
  for (auto reply:budget_candidates(d,fallback,rules,6)) {
    auto trial=state;
    auto streams=review_streams(seed);
    step(trial,reply,rules);
    const bool exact=scoring_result(trial,learner)==0;
    while (trial.result==Result::Ongoing) {
      const int actor=trial.to_move;
      step(trial,full_study_fallback(trial,rules,learner,2,streams[actor]),rules);
    }
    search_work+=trial.ply-state.ply;
    const auto value=scoring_result(trial,learner);
    if (exact) return reply;
    if (value>=0 && value<best) {
      best=value;
      choice=reply;
    }
  }
  return choice;
}

static std::pair<ScoringRoot,State> scoring_root(uint64_t seed,int table,int cut,const Rules& rules,
                                                const StudentNetwork& reset,const StudentNetwork& inherited) {
  auto state=opening(seed,rules);
  auto streams=review_streams(seed);
  const int learner=int(seed%4);
  std::optional<ScoringRoot> root,first;
  while (state.result==Result::Ongoing) {
    const auto fallback=full_study_fallback(state,rules,learner,table,streams[state.to_move]);
    auto a=fallback;
    if (state.to_move==learner) {
      int work=0;
      a=full_student_turn(state,fallback,rules,reset,work);
      if (!root) {
        const auto b=full_student_turn(state,fallback,rules,inherited,work);
        if (cut>0 && !first) first=ScoringRoot{state,streams,fallback,a,b,!(a==b)};
        if ((cut==0 && !(a==b)) || (cut>0 && state.ply+1>=cut))
          root=ScoringRoot{state,streams,fallback,a,b,!(a==b)};
      }
    }
    step(state,a,rules);
  }
  if (!root && first) root=first;
  if (!root) throw std::runtime_error("requested scoring root was never reached for seed "+std::to_string(seed));
  return {*root,state};
}

static int scoring_empirical(const ScoringRoot& root,Move first,const Rules& rules,int table,
                             std::array<Rng,kSeats> streams,const StudentNetwork* student=nullptr,
                             int* work_done=nullptr,uint64_t* final_hash=nullptr,uint64_t pressure_seed=0) {
  auto state=root.state;
  const int learner=state.to_move;
  step(state,first,rules);
  int search_work=0;
  while (state.result==Result::Ongoing) {
    auto move=full_study_fallback(state,rules,learner,table==4?2:table,streams[state.to_move]);
    if (pressure_seed && team_of(learner)!=team_of(state.to_move)) {
      move=rival_probe(state,move,rules,learner,pressure_seed,search_work);
      pressure_seed=0;
    }
    if (student && state.to_move==learner) {
      int work=0;
      move=full_student_turn(state,move,rules,*student,work);
    }
    step(state,move,rules);
  }
  if (work_done) *work_done=state.ply-root.state.ply+search_work;
  if (final_hash) *final_hash=state_hash(state);
  return scoring_result(state,learner);
}

static void emit_scoring_leaf_terms(const ScoringRoot& root,uint64_t seed,const Rules& rules,const ShortWeights& peer) {
  const auto& state=root.state;
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  const OpeningMarks marks(d.observation,false);
  const PublicReplyHistory history(state,false);
  for (auto action:budget_candidates(d,root.fallback,rules,4)) for (int sample=0;sample<4;++sample) {
    ResponseBranch branch(state,marks,history,mix_seed(seed,9041+sample));
    ResponseAudit work{6,0,0,false,true};
    branch.play(action,rules,true);
    continue_response(branch,rules,peer,sample%2,6,true,&work);
    ResponseValue score;
    record_leaf(score,branch.sample,branch.viewer,branch.root_ply,false,rules,RoutePriority::Old);
    const auto terms=branch.sample.result==Result::Ongoing?leaf_terms(branch.sample,team_of(branch.viewer)):
                       LeafValueTerms{};
    const double speed=score.wins?score.value-5.:0.;
    std::cout<<"L "<<seed<<' '<<early_action_key(action)<<' '<<sample<<' '<<state_hash(branch.sample)
             <<' '<<int(branch.sample.result)<<' '<<std::setprecision(17)<<score.value<<' '<<terms.movable
             <<' '<<terms.distance<<' '<<terms.adjacency<<' '<<speed<<' '<<1+work.plies<<'\n';
  }
}

static void emit_matched_reply_budget(const ScoringRoot& root,uint64_t seed,const Rules& rules,
                                      const ShortWeights& peer) {
  const auto& state=root.state;
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  const OpeningMarks marks(d.observation,false);
  const PublicReplyHistory history(state,false);
  for (const auto action:budget_candidates(d,root.fallback,rules,4)) for (int depth:{3,6,12}) {
    ResponseValue value;
    double without_distance=0.;
    int used=0,terminals=0;
    for (int i=0;i<24/depth;++i) {
      ResponseBranch branch(state,marks,history,mix_seed(seed,9041+i));
      ResponseAudit limit{depth,0,0,false,true};
      branch.play(action,rules,true);
      continue_response(branch,rules,peer,i%2,depth,true,&limit);
      record_leaf(value,branch.sample,branch.viewer,branch.root_ply,false,rules,RoutePriority::Old);
      used+=1+limit.plies;
      terminals+=branch.sample.result!=Result::Ongoing;
      if (branch.sample.result==Result::Ongoing)
        without_distance+=leaf_terms(branch.sample,team_of(branch.viewer)).distance;
    }
    const double mean=value.value/(24/depth);
    std::cout<<"N "<<seed<<' '<<early_action_key(action)<<' '<<depth<<' '<<used<<' '<<terminals<<' '
             <<std::setprecision(17)<<mean<<' '<<mean-without_distance/(24/depth)<<'\n';
  }
}

static std::vector<Move> scoring_ablation(const SeatDecision& decision,const ScoringRoot& root,
                                         const Rules& rules,int omit) {
  const DecisionScorer path(decision.observation,rules,DecisionKind::FlagRoute);
  std::vector<ScoredMove> order;
  for (auto action:decision.actions) {
    const auto score=path.score(action);
    const int terms[3]{score.objective,score.position,score.exposure};
    order.push_back({action,score.total()-terms[omit]});
  }
  std::stable_sort(order.begin(),order.end(),[](const auto& a,const auto& b){return a.score>b.score;});
  std::vector<Move> shortlist{root.fallback};
  for (auto row:order) {
    if (shortlist.size()==4) break;
    if (!(row.move==root.fallback)) shortlist.push_back(row.move);
  }
  return shortlist;
}

static void emit_scoring_root(const ScoringRoot& root,const State& end,uint64_t seed,int table,const Rules& rules,
                              const ShortWeights& peer,const StudentNetwork& reset,const StudentNetwork& inherited) {
  const auto& state=root.state;
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  const DecisionScorer path(d.observation,rules,DecisionKind::FlagRoute);
  const auto short_list=budget_candidates(d,root.fallback,rules,4);
  const OpeningMarks marks(d.observation,false);
  uint64_t old_hash=0,new_hash=0;
  const auto old_outcome=scoring_empirical(root,root.reset,rules,table,root.streams,&reset,nullptr,&old_hash);
  const auto new_outcome=scoring_empirical(root,root.inherited,rules,table,root.streams,&inherited,nullptr,&new_hash);
  single_case_check(old_outcome==scoring_result(end,state.to_move) && old_hash==state_hash(end),
                    "replayed actual reset suffix drifted");
  std::cout<<"R "<<seed<<' '<<state.ply+1<<' '<<state_hash(state)<<' '<<int(state.to_move)<<' '
           <<early_action_key(root.fallback)<<' '<<early_action_key(root.reset)<<' '<<early_action_key(root.inherited)
           <<' '<<old_outcome<<' '<<new_outcome<<' '<<old_hash<<' '<<new_hash<<' '<<d.actions.size()<<'\n';
  auto fallback_state=state;
  step(fallback_state,root.fallback,rules);
  const bool won=scoring_result(fallback_state,state.to_move)==2;
  for (int omit=0;omit<3;++omit) {
    int work=0;
    const auto list=scoring_ablation(d,root,rules,omit);
    const auto changed=won?root.fallback:window_ranking(d,list,root.fallback,rules,inherited,true,work).action;
    std::cout<<"X "<<seed<<' '<<omit<<' '<<early_action_key(changed);
    for (auto move:list) std::cout<<' '<<early_action_key(move);
    std::cout<<'\n';
  }
  const PublicReplyHistory history(state,false);
  for (auto action:d.actions) {
    auto next=state;
    const auto event=step(next,action,rules);
    const auto s=path.score(action);
    const auto pos=std::find(short_list.begin(),short_list.end(),action);
    const int rank=pos==short_list.end()?-1:int(pos-short_list.begin());
    int work=0;
    const auto x=window_input(d,marks,action,rules,true,work);
    const auto a=reset.score(x,false);
    const auto b=inherited.score(x,false);
    ResponseValue value;
    ResponseAudit meter{6,0,0,false,true};
    if (rank>=0) for (int i=0;i<4;++i)
      response_trial(state,action,marks,history,rules,peer,i%2,mix_seed(seed,9041+i),value,
                     false,RoutePriority::Old,true,&meter);
    std::cout<<"A "<<seed<<' '<<early_action_key(action)<<' '<<rank<<' '<<s.objective<<' '<<s.position<<' '
             <<s.exposure<<' '<<scoring_result(next,state.to_move)<<' '<<int(event.outcome)<<' '<<int(next.to_move)<<' '
             <<std::setprecision(17)<<a<<' '<<b<<' '<<value.value<<' '<<meter.plies<<'\n';
    std::cout<<"F "<<seed<<' '<<early_action_key(action);
    for (const auto* model:{&reset,&inherited}) {
      const auto z=model->logits(x);
      for (int i=0;i<3;++i) std::cout<<' '<<z[i];
    }
    for (auto feature:x) std::cout<<' '<<feature;
    std::cout<<'\n';
  }
}

static void run_scoring_audit(const Rules& rules) {
  ShortWeights peer{};
  StudentNetwork reset,inherited;
  for (auto& v:peer) single_case_check(bool(std::cin>>v) && std::isfinite(v),"invalid scoring peer");
  if (active_full_research())
    std::cerr<<"JUNQI_ACTIVE_PEER "<<fnv1a(peer.data(),sizeof(double)*peer.size())<<'\n';
  single_case_check(reset.load(std::cin,true,true) && inherited.load(std::cin,true,true),"invalid scoring students");
  uint64_t seed;
  int table,cut,mode,action,stream;
  while (std::cin>>seed>>table>>cut>>mode) {
    single_case_check(table>=0 && table<=2 && cut>=0 && cut<=100 && mode>=0 && mode<=4,"invalid scoring request");
    const auto [root,end]=scoring_root(seed,table,cut,rules,reset,inherited);
    if (mode==0) {
      emit_scoring_root(root,end,seed,table,rules,peer,reset,inherited);
      continue;
    }
    if (mode==3) {
      emit_scoring_leaf_terms(root,seed,rules,peer);
      continue;
    }
    if (mode==4) {
      emit_matched_reply_budget(root,seed,rules,peer);
      continue;
    }
    int count;
    single_case_check(bool(std::cin>>count) && count>=1 && count<=kNodes*kNodes,"invalid branch count");
    for (int i=0;i<count;++i) {
      int desk;
      single_case_check(bool(std::cin>>action>>desk>>stream) && desk>=0 && desk<=4 && stream>=0,
                        "invalid scoring continuation");
      const auto move=archived_action(action);
      single_case_check(is_legal(root.state,move),"scoring continuation began with illegal action");
      const auto tape=review_streams(mix_seed(seed,uint64_t(stream)));
      int plies=0;
      const auto outcome=scoring_empirical(root,move,rules,desk,tape,mode==2?&reset:nullptr,&plies,
                                           nullptr,desk==4?mix_seed(seed,uint64_t(stream)+500000):0);
      std::cout<<"B "<<seed<<' '<<state_hash(root.state)<<' '<<action<<' '<<desk<<' '<<stream<<' '<<outcome<<' '<<plies<<'\n';
    }
  }
  single_case_check(std::cin.eof(),"incomplete scoring manifest");
}
