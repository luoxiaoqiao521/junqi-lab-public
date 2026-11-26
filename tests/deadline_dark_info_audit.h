#pragma once

static Move dark_actor_move(const EarlyPrefix& game,const SeatDecision& d,const Rules& rules,
                            int learner,int table,Rng& rng) {
  const int actor=d.observation.viewer;
  const bool special_ally=(table&1) && actor==(learner+2)%4;
  const bool special_enemy=(table&2) && team_of(actor)!=team_of(learner);
  const DecisionScorer scorer(d.observation,rules,special_ally||special_enemy?
                               DecisionKind::CollisionHeuristic:DecisionKind::FlagRoute);
  Move choice;
  single_case_check(scorer.choose(rng,choice),"dark response has no legal action");
  if (special_enemy)
    return plan(d,game.marks[actor],choice,game.reply_history.last_moves()[actor],rules);
  return choice;
}

static void dark_continue(EarlyPrefix& game,const Rules& rules,int learner,int table,int stop,
                          std::array<Rng,kSeats>& streams) {
  while (game.state.result==Result::Ongoing && game.state.ply<stop) {
    const auto d=seat_decision(game.state);
    const auto answer=dark_actor_move(game,d,rules,learner,table,streams[d.observation.viewer]);
    game.play(answer,learner,rules);
  }
}

static double dark_leaf_estimate(const State& state,int learner) {
  const int outcome=review_outcome(state.result,learner);
  return outcome>=0?.5*outcome:.5+.25*std::tanh(leaf_value(state,team_of(learner)));
}

static int dark_rollout(EarlyPrefix game,Move first,const Rules& rules,int table,uint64_t seed,
                        int cap,int& work,double* estimate=nullptr) {
  const int actor=game.state.to_move;
  const int start=game.state.ply;
  auto streams=review_streams(seed);
  game.play(first,actor,rules);
  dark_continue(game,rules,actor,table,start+cap,streams);
  work+=game.state.ply-start;
  const int outcome=review_outcome(game.state.result,actor);
  if (estimate) *estimate=dark_leaf_estimate(game.state,actor);
  return outcome;
}

static bool equal_dark_marks(const OpeningMarks& first,const OpeningMarks& second) {
  if (first.ranks!=second.ranks || first.moved!=second.moved) return false;
  for (int s=0;s<kSeats;++s)
    if (first.removed[s].size!=second.removed[s].size || first.removed[s].masks!=second.removed[s].masks) return false;
  return true;
}

static bool equal_dark_actor(const EarlyPrefix& a,const EarlyPrefix& b,int actor) {
  const auto view=observe(a.state,actor),other=observe(b.state,actor);
  if (!same_observation(view,other) || a.reply_history.last_moves()!=b.reply_history.last_moves() ||
      !equal_dark_marks(a.marks[actor],b.marks[actor])) return false;
  return equal_dark_marks(a.reply_history.for_actor(view,a.marks[actor]),
                          b.reply_history.for_actor(other,b.marks[actor]));
}

static void dark_future_controls(EarlyPrefix a,EarlyPrefix b,const Rules& rules,const std::string& key,
                                 uint64_t seed,char tag='I') {
  std::array<int,kSeats> matches{},private_changes{};
  int legal_candidates=0;
  const int learner=a.state.to_move;
  auto ar=review_streams(seed),br=ar;
  for (int i=0;i<12 && a.state.result==Result::Ongoing && b.state.result==Result::Ongoing;++i) {
    const int actor=a.state.to_move;
    if (actor!=b.state.to_move) break;
    const auto d=seat_decision(a.state),e=seat_decision(b.state);
    if (equal_dark_actor(a,b,actor)) {
      for (int table=0;table<4;++table) {
        auto rr=ar[actor],ss=br[actor];
        single_case_check(d.actions==e.actions && dark_actor_move(a,d,rules,learner,table,rr)==
                          dark_actor_move(b,e,rules,learner,table,ss),"future policy depends on another seat's hidden rank");
      }
      ++matches[actor];
      legal_candidates+=int(d.actions.size());
    } else ++private_changes[actor];
    const auto quiet=std::find_if(d.actions.begin(),d.actions.end(),[&](Move m){
      return d.observation.seat_at(m.to)==kNone && std::find(e.actions.begin(),e.actions.end(),m)!=e.actions.end();
    });
    if (quiet==d.actions.end()) break;
    a.play(*quiet,learner,rules);
    b.play(*quiet,learner,rules);
  }
  std::cout<<tag<<' '<<key<<' '<<legal_candidates;
  for (int n:matches) std::cout<<' '<<n;
  for (int n:private_changes) std::cout<<' '<<n;
  std::cout<<'\n';
}

static void dark_isolation_fixtures(const Rules& rules) {
  for (int owner=0;owner<kSeats;++owner) {
    auto state=opening(290100+owner,rules), alternate=state;
    int division=kNone,regiment=kNone;
    for (int id=owner*25;id<(owner+1)*25;++id) {
      if (state.rank_of[id]==Division) division=id;
      if (state.rank_of[id]==Regiment) regiment=id;
    }
    single_case_check(division>=0 && regiment>=0,"missing test ranks");
    std::swap(alternate.rank_of[division],alternate.rank_of[regiment]);
    std::swap(alternate.rank_at[state.node_of[division]],alternate.rank_at[state.node_of[regiment]]);
    dark_future_controls(EarlyPrefix(state,291000),EarlyPrefix(alternate,291000),rules,
                         std::to_string(owner),291000+owner,'C');
  }
}

static std::vector<EarlyPrefix> dark_world_prefixes(const CoverageRoot& root,const CoverageTrial& trial,
                                                   const StudentNetwork& student) {
  std::vector<EarlyPrefix> games;
  const auto& view=root.decision.observation;
  for (const auto& world:trial.worlds) {
    single_case_check(same_capture_evidence(root,world,student),"reconstructed world changed actor evidence or model inputs");
    int8_t layout[kSeats][25];
    for (int s=0;s<kSeats;++s) for (int p=0;p<25;++p)
      layout[s][p]=world.rank_of[s*kPiecesPerSeat+p];
    games.push_back(coverage_prefix(root.raw,layout,root.rules,view.viewer,view.ply).game);
  }
  return games;
}

static Move dark_joint_pick(const std::string& key,const CoverageRoot& root,const std::vector<EarlyPrefix>& worlds,
                            const std::vector<Move>& candidates,Move fallback,uint64_t seed) {
  double value=-1.;
  Move picked=fallback;
  auto order=candidates;
  if (const auto where=std::find(order.begin(),order.end(),fallback); where!=order.end()) std::iter_swap(order.begin(),where);
  for (auto move:order) {
    int known=0,score=0,steps=0;
    for (size_t i=0;i<worlds.size();++i) for (int table:{0,1}) {
      const auto result=dark_rollout(worlds[i],move,root.rules,table,mix_seed(seed,i*19+table),160,steps);
      known+=result>=0;
      score+=result>=0?result:0;
    }
    const int sampled=int(worlds.size())*2;
    const int upper=score+(sampled-known)*2;
    if (sampled && double(score+upper)/(sampled*4)>value) {
      value=double(score+upper)/(sampled*4);
      picked=move;
    }
    std::cout<<"H "<<key<<' '<<early_action_key(move)<<' '<<sampled<<' '<<known<<' '<<score<<' '<<upper<<' '<<steps<<'\n';
  }
  return picked;
}

static void dark_audit_root(const ArchiveRows& raw,const Rules& rules,const std::string& key,int viewer,int cut,
                            uint64_t seed,const StudentNetwork& student) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto d=seat_decision(prefix.game.state);
  const auto& actor=prefix.game.marks[viewer];
  std::vector<CaptureEvidence> hits;
  const auto history=prefix.game.reply_history.for_actor(d.observation,actor,&hits);
  const CoverageRoot root{raw,rules,prefix,d,history};
  const auto salt=public_rollout_seed(d.observation,history,d.observation.quiet_clock);
  auto rng=Rng(mix_seed(seed,salt));
  Move fallback;
  single_case_check(DecisionScorer(d.observation,rules,DecisionKind::FlagRoute).choose(rng,fallback),"missing path action");
  const auto candidates=budget_candidates(d,fallback,rules,4);
  const auto a=student_pick(d,OpeningMarks(d.observation,false),fallback,candidates,student,rules,false);
  const auto b=student_pick(d,history,fallback,candidates,student,rules,false);
  int changes=0;
  for (auto action:d.actions)
    changes+=student_features(d,OpeningMarks(d.observation,false),action,&rules)!=student_features(d,history,action,&rules);
  const auto focus=directed_contact(d,history,rules,0);
  const auto trial=run_directed_panel(root,mix_seed(salt,seed),2,focus).trial;
  const auto worlds=dark_world_prefixes(root,trial,student);
  const auto c=dark_joint_pick(key,root,worlds,candidates,b,mix_seed(seed,salt+131));
  const auto ally=compare_visibility(prefix.game.state,IdentityVisibility::AllyAndPublic);
  const auto full=compare_visibility(prefix.game.state,IdentityVisibility::AllIdentities);
  const auto visible_pick=[&](const SeatDecision& view) {
    single_case_check(view.actions==d.actions,"diagnostic disclosure changed the legal set");
    return student_pick(view,OpeningMarks(view.observation,false),fallback,candidates,student,rules,false);
  };
  const auto disclosed=visible_pick(ally),oracle=visible_pick(full);
  bool guarded=false;
  try { prefix.game.reply_history.for_actor(full.observation,actor); }
  catch (const std::runtime_error&) { guarded=true; }
  single_case_check(guarded,"all-visible view bypassed the dark-history guard");
  std::cout<<"D "<<key<<' '<<viewer<<' '<<cut<<' '<<d.actions.size()<<' '<<candidates.size()<<' '
           <<hits.size()<<' '<<changes<<' '<<worlds.size()<<' '<<trial.proposals<<' '<<trial.nodes<<' '
           <<trial.replay_failed<<' '<<trial.unresolved<<' '<<directed_sides(trial,focus);
  for (auto move:{fallback,a,b,c,disclosed,oracle}) std::cout<<' '<<early_action_key(move);
  std::cout<<'\n';
  for (const auto& world:worlds) dark_future_controls(prefix.game,world,rules,key,seed);
  std::vector<Move> unique;
  for (auto move:{fallback,a,b,c,disclosed,oracle})
    if (std::find(unique.begin(),unique.end(),move)==unique.end()) unique.push_back(move);
  for (auto move:unique) for (int table=0;table<4;++table) for (int i=0;i<4;++i) {
    int plies=0;
    const auto result=dark_rollout(prefix.game,move,rules,table,mix_seed(seed,80319+i),400,plies);
    std::cout<<"Y "<<key<<' '<<early_action_key(move)<<' '<<table<<' '<<i<<' '<<result<<' '<<plies<<'\n';
  }
}

static void run_dark_info_audit(const char* rules_path) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid dark information student");
  std::string key,hex,error;
  Rules fixture_rules;
  single_case_check(parse_rules(read_text(rules_path),fixture_rules,error),"invalid isolation fixture rules");
  dark_isolation_fixtures(fixture_rules);
  int viewer,cut,direction;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>direction>>seed>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && (direction==0 || direction==1) &&
                      parse_rules(archive_review_rules(read_text(rules_path),bool(direction)),rules,error),
                      "invalid dark archive audit input");
    dark_audit_root(ArchiveRows(decode_archive_tape(hex)),rules,key,viewer,cut,seed,student);
  }
  single_case_check(std::cin.eof(),"incomplete dark archive request");
}
