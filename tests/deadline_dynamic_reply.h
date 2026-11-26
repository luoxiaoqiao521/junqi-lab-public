#pragma once

static Move study_baseline(const State& state,const Rules& rules,int ally,int table,Rng& rng) {
  if (table!=1 || state.to_move!=ally) return route_baseline(state,rules,rng);
  const auto view=observe(state,ally,IdentityVisibility::AllIdentities);
  Move move;
  single_case_check(heuristic_move(view,rng,move),"frozen partner had no legal move");
  return move;
}

struct FrozenReplyPlayers {
  ShortWeights peer{};
  StudentNetwork board,window;

  void load() {
    for (double& w:peer) single_case_check(bool(std::cin>>w) && std::isfinite(w),"missing frozen reply policy");
    if (active_full_research())
      std::cerr<<"JUNQI_ACTIVE_PEER "<<fnv1a(peer.data(),peer.size()*sizeof(double))<<'\n';
    single_case_check(board.load(std::cin,true) && window.load(std::cin,true,true),"invalid frozen students");
  }
};

static void study_frozen_candidates(const std::string& key,const State& root,Move fallback,Move witness,
                                    const Rules& rules,const FrozenReplyPlayers& models,uint64_t seed) {
  const auto d=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto narrow=budget_candidates(d,fallback,rules,4);
  const auto wide=budget_candidates(d,fallback,rules,8);
  const auto all=budget_candidates(d,fallback,rules,d.actions.size());
  const auto& view=d.observation;
  const OpeningMarks marks(view,false);
  int work=0;
  const auto s44=window_ranking(d,narrow,fallback,rules,models.board,false,work).action;
  const auto s52=window_ranking(d,narrow,fallback,rules,models.window,true,work).action;
  const auto wide52=window_ranking(d,wide,fallback,rules,models.window,true,work).action;
  const auto full52=window_ranking(d,all,fallback,rules,models.window,true,work).action;
  const auto teacher=budget_choice(root,rules,models.peer,fallback,BudgetArm::Extend,mix_seed(seed,root.ply+3301)).action;
  std::cout<<"R "<<key<<' '<<root.ply+1<<' '<<root.to_move+0<<' '<<state_hash(root)<<' '
           <<early_action_key(fallback)<<' '<<early_action_key(witness)<<' '<<early_action_key(s44)<<' '
           <<early_action_key(s52)<<' '<<early_action_key(wide52)<<' '<<early_action_key(full52)<<' '
           <<early_action_key(teacher)<<' '<<narrow.size()<<' '<<wide.size()<<' '<<all.size()<<'\n';
  std::cout<<std::setprecision(17);
  for (size_t i=0;i<all.size();++i) {
    const auto a=all[i];
    std::cout<<"S "<<key<<' '<<i<<' '<<early_action_key(a)<<' '
             <<window_score(d,marks,a,rules,models.board,false,work)<<' '
             <<window_score(d,marks,a,rules,models.window,true,work)<<'\n';
  }
  std::vector<Move> choices;
  for (const auto action:{fallback,witness,s44,s52,wide52,full52,teacher})
    if (std::find(choices.begin(),choices.end(),action)==choices.end()) choices.push_back(action);
  for (size_t i=0;i<choices.size();++i)
    for (int table:{0,1}) for (int stream=0;stream<8;++stream) {
      const auto sub_seed=mix_seed(seed,17678+stream);
      route_continuation(root,choices[i],rules,root.to_move,table,sub_seed,key,root.ply+1,int(i),stream);
    }
}

struct FrozenArchiveCut {
  const FrozenReplyPlayers& models;
  const std::string& key;
  uint64_t seed;
  int cut, original, witness;
  bool seen=false;

  void inspect(const State& root,Move,const Rules& rules) {
    if (root.ply+1!=cut) return;
    Rng rng(42);
    const auto old=route_baseline(root,rules,rng);
    const auto alt=archived_action(witness);
    single_case_check(old==archived_action(original) && is_legal(root,alt),"frozen archive actions drifted");
    seen=true;
    study_frozen_candidates(key,root,old,alt,rules,models,seed);
  }
};

static void study_frozen_seed(const std::string& key,uint64_t seed,int cut,int table,int original,int witness,
                             const Rules& rules,const FrozenReplyPlayers& models) {
  auto root=opening(seed,rules);
  auto rng=review_streams(seed);
  while (root.result==Result::Ongoing && root.ply<cut-1) {
    const auto choice=study_baseline(root,rules,(int(seed%4)+2)%4,table,rng[root.to_move]);
    step(root,choice,rules);
  }
  single_case_check(root.result==Result::Ongoing && root.to_move==int(seed%4),"frozen opening prefix ended early");
  const auto old=route_baseline(root,rules,rng[root.to_move]);
  const auto alt=archived_action(witness);
  single_case_check(old==archived_action(original) && is_legal(root,alt),"frozen opening actions drifted");
  study_frozen_candidates(key,root,old,alt,rules,models,seed);
}

static void run_frozen_reply_audit(const char* path,const Rules& rules) {
  FrozenReplyPlayers models;
  models.load();
  const auto config=read_text(path);
  std::string tag,key,tape,error;
  int order,cut,original,witness;
  uint64_t seed;
  while (std::cin>>tag>>key>>order>>cut>>seed>>original>>witness) {
    single_case_check((order==0 || order==1) && cut>=24 && cut<=224,"invalid frozen audit manifest");
    if (tag=="S") {
      study_frozen_seed(key,seed,cut,order,original,witness,rules,models);
      continue;
    }
    FrozenArchiveCut scan{models,key,seed,cut,original,witness};
    auto local=rules;
    local.ccw=order!=0;
    RuleComparison checked;
    if (tag=="H") checked=scan_jqh_pressure(scan,local,cut);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid frozen audit archive");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      checked=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cut);
    }
    single_case_check(scan.seen && checked.agrees==cut,"failed to restore the archived cut");
  }
  single_case_check(std::cin.eof(),"incomplete frozen audit payload");
}

enum class ReplySchedule { Path, Fixed, Dynamic, FixedHybrid, DynamicHybrid, Wide, WideIfClose };

struct ReplyScheduleTrace {
  int scanned=0, used=0, late=0, last=-32, changed=0, searches=0, search_steps=0, feature_steps=0;
  int first=0, old_move=kNone, new_move=kNone, extended=0, override_student=0;
  int widened=0, tail_selected=0;
  int64_t micros=0;
  uint64_t root_hash=0;
  bool windows[2]{};
};

static Move scheduled_reply(const State& state,Move fallback,const Rules& rules,const FrozenReplyPlayers& models,
                           ReplySchedule arm,ReplyScheduleTrace& trace,uint64_t seed) {
  if (arm==ReplySchedule::Path || trace.used==2) return fallback;
  const bool dynamic=arm==ReplySchedule::Dynamic || arm==ReplySchedule::DynamicHybrid;
  const bool hybrid=arm==ReplySchedule::FixedHybrid || arm==ReplySchedule::DynamicHybrid;
  const int phase=state.ply>=111?1:0;
  if (dynamic ? state.ply<19 || state.ply-trace.last<32 : state.ply<31 || trace.windows[phase])
    return fallback;
  ++trace.scanned;
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  const auto actions=budget_candidates(d,fallback,rules,4);
  int no_work=0;
  const auto cheap=window_ranking(d,actions,fallback,rules,models.board,false,no_work);
  // Dynamic timing tests whether the fast student's disagreement can locate a useful intervention.
  if (dynamic && (cheap.action==fallback || cheap.best-cheap.fallback<0.025)) return fallback;
  ++trace.used;
  trace.windows[phase]=true;
  trace.late+=state.ply>=100;
  trace.last=state.ply;
  if (actions.size()==1) return fallback;
  auto next=state;
  step(next,fallback,rules);
  if (team_score(next.result,state.to_move)==1.0) return fallback;
  const auto student=window_ranking(d,actions,fallback,rules,models.window,true,trace.feature_steps);
  const bool widen=arm==ReplySchedule::Wide ||
                   (arm==ReplySchedule::WideIfClose && student.best-student.second<=0.01);
  if (widen && d.actions.size()>actions.size()) {
    const auto extended=budget_candidates(d,fallback,rules,8);
    single_case_check(std::equal(actions.begin(),actions.end(),extended.begin()),"eight-move prefix drifted");
    const std::vector<Move> tail(extended.begin()+actions.size(),extended.end());
    const auto extra=window_ranking(d,tail,fallback,rules,models.window,true,trace.feature_steps);
    ++trace.widened;
    if (extra.best>student.best+1e-12) {
      ++trace.tail_selected;
      return extra.action;
    }
  }
  if (!hybrid || student.action==cheap.action || student.best-student.second>0.08) return student.action;
  const auto extension=budget_choice(state,rules,models.peer,fallback,BudgetArm::Extend,mix_seed(seed,state.ply+3301));
  ++trace.searches;
  trace.search_steps+=extension.work.plies;
  trace.extended+=extension.work.extended;
  trace.override_student+=!(extension.action==student.action);
  return extension.action;
}

static void run_scheduled_reply_game(uint64_t seed,int table,ReplySchedule arm,const Rules& rules,
                                     const FrozenReplyPlayers& models,bool show_width=false) {
  auto root=opening(seed,rules);
  auto rng=review_streams(seed);
  const int viewer=int(seed%4), ally=(viewer+2)%4;
  ReplyScheduleTrace trace;
  while (root.result==Result::Ongoing) {
    const auto fallback=study_baseline(root,rules,ally,table,rng[root.to_move]);
    auto choice=fallback;
    if (root.to_move==viewer && arm!=ReplySchedule::Path) {
      const auto begin=std::chrono::steady_clock::now();
      choice=scheduled_reply(root,fallback,rules,models,arm,trace,seed);
      trace.micros+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
      if (!(choice==fallback)) {
        ++trace.changed;
        if (!trace.first) {
          trace.first=root.ply+1;
          trace.root_hash=state_hash(root);
          trace.old_move=early_action_key(fallback);
          trace.new_move=early_action_key(choice);
        }
      }
    }
    single_case_check(is_legal(root,choice),"scheduled reply was not legal");
    step(root,choice,rules);
  }
  const auto points=team_score(root.result,viewer);
  std::cout<<"G "<<seed<<' '<<table<<' '<<int(arm)<<' '<<root.ply<<' '<<(points?int(*points*2):-1)<<' '
           <<trace.scanned<<' '<<trace.used<<' '<<trace.late<<' '<<trace.changed<<' '<<trace.searches<<' '
           <<trace.search_steps<<' '<<trace.feature_steps<<' '<<trace.micros<<' '<<trace.first<<' '
           <<trace.root_hash<<' '<<trace.old_move<<' '<<trace.new_move<<' '<<state_hash(root)<<' '
           <<trace.extended<<' '<<trace.override_student;
  if (show_width) std::cout<<' '<<trace.widened<<' '<<trace.tail_selected;
  std::cout<<'\n';
}

static void run_scheduled_reply_games(const Rules& rules) {
  uint64_t seed;
  int count;
  int mode;
  single_case_check(bool(std::cin>>seed>>count>>mode) && count>=1 && count<=32 && mode>=0 && mode<=3,
                    "invalid scheduled reply budget");
  FrozenReplyPlayers models;
  models.load();
  for (uint64_t s=seed;s<seed+uint64_t(count);++s) for (int table:{0,1}) {
    if (mode==3) {
      for (const auto arm:{ReplySchedule::Path,ReplySchedule::Fixed,ReplySchedule::Wide,ReplySchedule::WideIfClose})
        run_scheduled_reply_game(s,table,arm,rules,models,true);
      continue;
    }
    run_scheduled_reply_game(s,table,ReplySchedule::Path,rules,models);
    if (mode!=2) {
      run_scheduled_reply_game(s,table,ReplySchedule::Fixed,rules,models);
      run_scheduled_reply_game(s,table,ReplySchedule::Dynamic,rules,models);
    }
    if (mode!=1) {
      run_scheduled_reply_game(s,table,ReplySchedule::FixedHybrid,rules,models);
      run_scheduled_reply_game(s,table,ReplySchedule::DynamicHybrid,rules,models);
    }
  }
}
