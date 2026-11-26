#pragma once

// An actual first enemy gets one extra, observation-only reply policy for the stress arm.
static Move dark_pressure_reply(const SeatDecision& decision,const OpeningMarks& evidence,Move last,
                               int learner,Move first,Move natural,const Rules& rules) {
  const auto& view=decision.observation;
  if (view.seat_at(first.to)==learner) {
    const auto possible=compatible_rank_mask(view,evidence,first.to);
    const DecisionScorer score(view,rules,DecisionKind::CollisionHeuristic);
    double best=-1e9;
    Move chosen=natural;
    for (const auto move:decision.actions) if (move.to==first.to && possible) {
      const auto outcomes=collision_outcome_mask(view.rank_at(move.from),possible,rules);
      if (outcomes==outcome_bit(Outcome::DefenderWins)) continue;
      const auto value=score.score(move).total();
      if (value>best) { best=value; chosen=move; }
    }
    if (!(chosen==natural)) return chosen;
  }
  return plan(decision,evidence,natural,last,rules);
}

struct DarkReplyLine {
  EarlyPrefix game;
  std::array<Rng,kSeats> streams;
  int enemy=kNone,natural=-1,reply=-1,ally=-1;
  bool received_enemy=false;
};

static DarkReplyLine dark_reply_checkpoint(const EarlyPrefix& world,Move action,const Rules& rules,
    int table,uint64_t seed,bool pressure,int depth) {
  const int learner=world.state.to_move;
  DarkReplyLine line{world,review_streams(seed)};
  line.game.play(action,learner,rules);
  while (line.game.state.result==Result::Ongoing && line.game.state.ply<world.state.ply+depth) {
    const auto decision=seat_decision(line.game.state);
    const int actor=decision.observation.viewer;
    const auto natural=dark_actor_move(line.game,decision,rules,learner,table,line.streams[actor]);
    auto move=natural;
    if (!line.received_enemy && team_of(actor)!=team_of(learner)) {
      line.received_enemy=true;
      line.enemy=actor;
      if (pressure) {
        const auto& history=line.game.reply_history;
        move=dark_pressure_reply(decision,history.for_actor(decision.observation,line.game.marks[actor]),
                                 history.last_moves()[actor],learner,action,natural,rules);
      }
      line.natural=early_action_key(natural);
      line.reply=early_action_key(move);
    }
    if (actor==(learner+2)%kSeats && line.ally<0) line.ally=early_action_key(move);
    line.game.play(move,learner,rules);
  }
  return line;
}

static void emit_dark_reply_cells(const std::string& key,const EarlyPrefix& world,const Rules& rules,
                                  Move action,int side,int table,uint64_t seed) {
  const int learner=world.state.to_move;
  const int start=world.state.ply;
  const auto natural=dark_reply_checkpoint(world,action,rules,table,seed,false,48);
  const auto stress=dark_reply_checkpoint(world,action,rules,table,seed,true,48);
  single_case_check(natural.enemy==stress.enemy && natural.natural==stress.natural,
                    "first response fork changed its pre-reply observation");
  for (int arm=0;arm<2;++arm) {
    const auto& line=arm?stress:natural;
    const auto& checkpoint=line.game;
    std::cout<<"C "<<key<<' '<<side<<' '<<table<<' '<<arm<<' '<<early_action_key(action)<<' '
             <<line.enemy<<' '<<line.natural<<' '<<line.reply<<' '<<line.ally<<' '<<checkpoint.state.ply-start
             <<' '<<review_outcome(checkpoint.state.result,learner)<<' '<<std::setprecision(17)
             <<dark_leaf_estimate(checkpoint.state,learner)<<' '<<checkpoint.trace;
    for (int i=0;i<2;++i) {
      auto continuation=checkpoint;
      auto streams=review_streams(mix_seed(seed,88101+i));
      dark_continue(continuation,rules,learner,table,start+400,streams);
      std::cout<<' '<<review_outcome(continuation.state.result,learner)<<' '<<continuation.state.ply-checkpoint.state.ply;
    }
    std::cout<<'\n';
  }
}

static std::array<Move,2> reply_candidates(const SeatDecision& decision,const OpeningMarks& evidence,
    const Rules& rules,const StudentNetwork& student,uint64_t seed,Move focal,int old_key,int new_key) {
  if (old_key>=0 && new_key>=0) {
    const auto& legal=decision.actions;
    const auto find=[&](int code){ return std::find_if(legal.begin(),legal.end(),[&](Move m){return early_action_key(m)==code;}); };
    const auto a=find(old_key),b=find(new_key);
    single_case_check(a!=legal.end() && b!=legal.end() && a!=b,"prior decision was not restored");
    return {*a,*b};
  }
  Rng rng(seed);
  Move fallback;
  single_case_check(DecisionScorer(decision.observation,rules,DecisionKind::FlagRoute).choose(rng,fallback),"reply root has no fallback");
  const auto list=budget_candidates(decision,fallback,rules,4);
  const auto anchor=student_pick(decision,evidence,fallback,list,student,rules,false);
  if (!(anchor==focal)) return {anchor,focal};
  for (const auto m:list) if (!(m==anchor)) return {anchor,m};
  for (const auto m:decision.actions) if (!(m==anchor)) return {anchor,m};
  return {anchor,anchor};
}

static void dark_reply_leaf_root(const ArchiveRows& raw,const Rules& rules,const StudentNetwork& student,
    const std::string& key,int viewer,int cut,uint64_t seed,int old_key,int new_key,bool course) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto decision=seat_decision(prefix.game.state);
  const auto& view=decision.observation;
  const auto evidence=prefix.game.reply_history.for_actor(view,prefix.game.marks[viewer]);
  const auto focus=directed_contact(decision,evidence,rules,0);
  if (focus.kind!=0) { std::cout<<"M "<<key<<" no-focal-contact\n"; return; }
  const auto salt=mix_seed(seed,public_rollout_seed(view,evidence,view.quiet_clock));
  const auto actions=reply_candidates(decision,evidence,rules,student,salt,focus.move,old_key,new_key);
  if (actions[0]==actions[1]) { std::cout<<"M "<<key<<" single-action\n"; return; }
  const CoverageRoot root{raw,rules,prefix,decision,evidence};
  const auto began=std::chrono::steady_clock::now();
  const auto no=bomb_side(root,focus.move.to,false,salt,1),yes=bomb_side(root,focus.move.to,true,salt,1);
  const auto worlds=DarkBudgetWorlds{dark_world_prefixes(root,no,student),dark_world_prefixes(root,yes,student)};
  const auto micros=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-began).count();
  if (worlds[0].empty() || worlds[1].empty()) {
    std::cout<<"M "<<key<<" missing-side "<<no.contradicted<<' '<<yes.contradicted<<' '
             <<no.unresolved<<' '<<yes.unresolved<<' '<<no.replay_failed<<' '<<yes.replay_failed<<'\n'; return;
  }
  const auto possible=compatible_rank_mask(view,evidence,focus.move.to);
  int slots=0;
  for (int rank=Flag;rank<RankCount;++rank) if (possible&(1u<<rank)) slots+=kRankCount[rank];
  std::cout<<"D "<<key<<' '<<early_action_key(actions[0])<<' '<<early_action_key(actions[1])<<' '
           <<slots<<' '<<no.nodes+yes.nodes<<' '<<micros<<'\n';
  if (course) for (const auto move:actions) {
    std::cout<<"X "<<key<<' '<<early_action_key(move)<<std::setprecision(17);
    for (const auto value:student_features(decision,evidence,move,&rules)) std::cout<<' '<<value;
    std::cout<<'\n';
  }
  dark_future_controls(worlds[0].front(),worlds[1].front(),rules,key,salt);
  for (int table=0;table<4;++table) for (auto action:actions) {
    for (int side=0;side<2;++side)
      emit_dark_reply_cells(key,worlds[side].front(),rules,action,side,table,mix_seed(salt,3020+table));
    for (int i=0;i<2;++i) {
      int steps=0;
      const auto score=dark_rollout(prefix.game,action,rules,table,mix_seed(seed,95110+i),400,steps);
      std::cout<<"Y "<<key<<' '<<table<<' '<<early_action_key(action)<<' '<<i<<' '<<score<<' '<<steps<<'\n';
    }
  }
}

static void run_dark_reply_leaf(const char* rules_path,bool course) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid reply/leaf student");
  std::string key,hex,error;
  int viewer,cut,direction,old_key,new_key;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>direction>>seed>>old_key>>new_key>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && (direction==0 || direction==1) &&
                      parse_rules(archive_review_rules(read_text(rules_path),bool(direction)),rules,error),"invalid reply/leaf archive");
    dark_reply_leaf_root(ArchiveRows(decode_archive_tape(hex)),rules,student,key,viewer,cut,seed,old_key,new_key,course);
  }
  single_case_check(std::cin.eof(),"incomplete reply/leaf archive request");
}
