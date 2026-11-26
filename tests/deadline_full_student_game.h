#pragma once
#include <unordered_set>

static Move full_study_fallback(const State& state,const Rules& rules,int learner,int table,Rng& rng) {
  if ((table==2 || table==3) && team_of(state.to_move)!=team_of(learner)) {
    const auto view=observe(state,state.to_move,IdentityVisibility::AllIdentities);
    Move choice;
    single_case_check(DecisionScorer(view,rules,DecisionKind::CollisionHeuristic).choose(rng,choice),
                      "challenge policy failed to select a legal move");
    return choice;
  }
  return study_baseline(state,rules,(learner+2)%4,table==3?1:table,rng);
}

struct FullStudentPlan {
  const StudentNetwork* network=nullptr;
  bool continuous=false;
};

static Move full_student_turn(const State& state,Move fallback,const Rules& rules,
                              const StudentNetwork& network,int& work) {
  auto exact=state;
  step(exact,fallback,rules);
  return team_score(exact.result,state.to_move)==1.0?fallback:
         window_student_choice(state,fallback,rules,network,true,work);
}

struct FullStudentMetrics {
  int learner_turns=0, used=0, changed=0, steps=0, first_change=0;
  int engineer_move=0, engineer_exit=0, first_engineer_lost=0, first_enemy=0, first_exit=0;
  int first_exit_seat=kNone, first_exit_reason=0, support_open=0, support_used=0;
  int pending=0;
  int64_t micros=0;
  bool windows[2]{};
};

struct FullGameTrace {
  struct SeatTrace {
    Move last{};
    int id=kNone, turns=0, quiet=0, reversals=0, reversal_with_contact=0;
  };
  std::array<SeatTrace,kSeats> seats{};
  std::unordered_set<uint64_t> positions;
  std::vector<int> reversal_steps;
  int repeats=0, first_contact=0, first_flag=0, first_enemy_flag=0;
  EndReason ending=EndReason::None;
  int64_t decision_micros=0;

  void before(const State& state,Move action) {
    auto h=fnv1a(state.id_at,sizeof(state.id_at));
    h=fnv1a(state.out,sizeof(state.out),h);
    h=fnv1a(state.flag_revealed,sizeof(state.flag_revealed),h);
    repeats+=!positions.insert(fnv1a(&state.to_move,sizeof(state.to_move),h)).second;
    auto& seat=seats[state.to_move];
    const int id=state.id_at[action.from];
    ++seat.turns;
    if (state.seat_at[action.to]==kNone) {
      ++seat.quiet;
      if (seat.id==id && seat.last.from==action.to && seat.last.to==action.from) {
        ++seat.reversals;
        reversal_steps.push_back(state.ply+1);
        std::vector<Move> legal;
        gen_moves(StateView{state},state.to_move,legal);
        seat.reversal_with_contact+=std::any_of(legal.begin(),legal.end(),
                              [&](Move move){return state.seat_at[move.to]!=kNone;});
      }
    }
    seat.id=id;
    seat.last=action;
  }

  void after(const StepInfo& event,int ply,int learner) {
    if (!first_contact && event.outcome!=Outcome::Move) first_contact=ply;
    for (int seat=0;seat<kSeats;++seat) if (event.exit_reason[seat]==ExitReason::FlagRemoved) {
      if (!first_flag) first_flag=ply;
      if (!first_enemy_flag && team_of(seat)!=team_of(learner)) first_enemy_flag=ply;
    }
    ending=event.end_reason;
  }

  void print(const std::string& key,int table,int arm,int plies,int learner) const {
    const auto recent=std::count_if(reversal_steps.begin(),reversal_steps.end(),[&](int ply){return ply>plies-70;});
    std::cout<<"T "<<key<<' '<<table<<' '<<arm<<' '<<repeats<<' '<<first_contact<<' '<<first_flag<<' '
             <<first_enemy_flag<<' '<<int(ending)<<' '<<decision_micros<<' '<<recent;
    for (int i=0;i<kSeats;++i) {
      const auto& seat=seats[(learner+i)%kSeats];
      std::cout<<' '<<seat.turns<<' '<<seat.quiet<<' '<<seat.reversals<<' '<<seat.reversal_with_contact;
    }
    std::cout<<'\n';
  }
};

static int ally_flag_options(const State& state,const Rules& rules,int ally,std::optional<Move> preferred={}) {
  if (state.out[ally] || state.result!=Result::Ongoing) return 0;
  int mask=0;
  for (const int enemy:{(ally+1)%4,(ally+3)%4}) for (int col:{1,3}) {
    const int flag=Board::get().seat_node(enemy,5,col);
    if (state.seat_at[flag]!=enemy) continue;
    const auto access=relay_flag_access(state,ally,flag,rules,preferred);
    if (access && (!preferred || access->move==*preferred)) mask|=1<<enemy;
  }
  return mask;
}

static StepInfo record_study_move(State& state,Move move,const Rules& rules,int learner,FullStudentMetrics& metrics) {
  const int actor=state.to_move;
  const int ally=(learner+2)%4;
  const int from_rank=state.rank_at[move.from];
  const int from_id=state.id_at[move.from], to_id=state.id_at[move.to];
  const bool own_engineer=actor==learner && from_rank==Engineer;
  const bool hit_engineer=state.seat_at[move.to]==learner && state.rank_at[move.to]==Engineer;
  const int before=actor==learner?ally_flag_options(state,rules,ally):0;
  if (actor==ally) {
    if (metrics.pending && (ally_flag_options(state,rules,ally,move)&metrics.pending)) ++metrics.support_used;
    metrics.pending=0;
  }
  const auto event=step(state,move,rules);
  if (own_engineer && !metrics.engineer_move) metrics.engineer_move=state.ply;
  if (own_engineer && !metrics.engineer_exit && Board::get().node(move.to).seat!=learner)
    metrics.engineer_exit=state.ply;
  if (!metrics.first_engineer_lost &&
      ((own_engineer && !state.alive[from_id]) || (hit_engineer && !state.alive[to_id])))
    metrics.first_engineer_lost=state.ply;
  if (actor==learner) {
    const int opened=ally_flag_options(state,rules,ally)&~before;
    if (opened) {
      ++metrics.support_open;
      metrics.pending|=opened;
    }
  }
  for (int seat=0;seat<4;++seat) if (event.exit_reason[seat]!=ExitReason::None) {
    if (!metrics.first_exit) {
      metrics.first_exit=state.ply;
      metrics.first_exit_seat=seat;
      metrics.first_exit_reason=int(event.exit_reason[seat]);
    }
    if (!metrics.first_enemy && team_of(seat)!=team_of(learner)) metrics.first_enemy=state.ply;
  }
  return event;
}

static void run_full_student_episode(State state,const Rules& rules,uint64_t seed,int learner,int table,
                                     int arm,const FullStudentPlan& plan,const std::string& prefix,bool telemetry=false) {
  auto rng=review_streams(seed);
  FullStudentMetrics metrics;
  std::optional<FullGameTrace> trace;
  if (telemetry) trace.emplace();
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto selection_start=telemetry?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    const auto fallback=full_study_fallback(state,rules,learner,table,rng[actor]);
    auto choice=fallback;
    if (actor==learner) {
      ++metrics.learner_turns;
      const int phase=state.ply>=111?1:0;
      if (plan.network && (plan.continuous || (state.ply>=31 && !metrics.windows[phase]))) {
        metrics.windows[phase]=true;
        ++metrics.used;
        const auto begin=std::chrono::steady_clock::now();
        choice=full_student_turn(state,fallback,rules,*plan.network,metrics.steps);
        metrics.micros+=std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now()-begin).count();
        if (!(choice==fallback)) {
          ++metrics.changed;
          if (!metrics.first_change) metrics.first_change=state.ply+1;
        }
      }
    }
    if (trace) trace->decision_micros+=std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now()-selection_start).count();
    single_case_check(is_legal(state,choice),"full study applied an illegal action");
    if (trace) trace->before(state,choice);
    const auto event=record_study_move(state,choice,rules,learner,metrics);
    if (trace) trace->after(event,state.ply,learner);
  }
  const auto outcome=team_score(state.result,learner);
  std::cout<<"G "<<prefix<<' '<<table<<' '<<arm<<' '<<state.ply<<' '<<(outcome?int(*outcome*2):-1)<<' '
           <<metrics.learner_turns<<' '<<metrics.used<<' '<<metrics.changed<<' '<<metrics.steps<<' '
           <<metrics.micros<<' '<<metrics.first_change<<' '<<metrics.engineer_move<<' '<<metrics.engineer_exit<<' '
           <<metrics.first_engineer_lost<<' '<<metrics.first_exit<<' '<<metrics.first_exit_seat<<' '
           <<metrics.first_exit_reason<<' '<<metrics.first_enemy<<' '<<metrics.support_open<<' '
           <<metrics.support_used<<' '<<state_hash(state)<<'\n';
  if (trace) trace->print(prefix,table,arm,state.ply,learner);
}

static void run_evolution_games(const Rules& rules,bool telemetry=false) {
  uint64_t start;
  int count;
  StudentNetwork parent,child;
  single_case_check(bool(std::cin>>start>>count) && count>=1 && count<=24 &&
                    parent.load(std::cin,true,true) && child.load(std::cin,true,true),"invalid full learner panel");
  const std::array<FullStudentPlan,5> plans{{{nullptr,false},{&parent,false},{&child,false},
                                            {&parent,true},{&child,true}}};
  for (uint64_t seed=start;seed<start+uint64_t(count);++seed) for (int table:{0,1,2})
    for (size_t arm=0;arm<plans.size();++arm)
      run_full_student_episode(opening(seed,rules),rules,seed,int(seed%4),table,int(arm),plans[arm],std::to_string(seed),telemetry);
}

static void run_preference_games(const Rules& rules) {
  uint64_t start;
  int count;
  std::array<StudentNetwork,3> models;
  single_case_check(bool(std::cin>>start>>count) && count>=1 && count<=32,"invalid preference game set");
  for (auto& model:models) single_case_check(model.load(std::cin,true,true),"invalid preference model");
  for (uint64_t seed=start;seed<start+uint64_t(count);++seed) for (int table:{0,1,2,3})
    for (size_t model=0;model<models.size();++model)
      run_full_student_episode(opening(seed,rules),rules,seed,int(seed%4),table,int(model),
                               {&models[model],true},std::to_string(seed));
}

struct OrdinaryRoot {
  State state;
  Move fallback;
  bool reached=false;
};

static OrdinaryRoot find_ordinary_root(uint64_t seed,int table,int cut,const Rules& rules,
                                      const FrozenReplyPlayers* historical) {
  auto state=opening(seed,rules);
  auto rng=review_streams(seed);
  const int learner=int(seed%4);
  ReplyScheduleTrace trace;
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    auto action=full_study_fallback(state,rules,learner,table,rng[actor]);
    if (actor==learner && state.ply+1>=cut) return {state,action,true};
    if (actor==learner && historical)
      action=scheduled_reply(state,action,rules,*historical,ReplySchedule::Fixed,trace,seed);
    step(state,action,rules);
  }
  return {state,{},false};
}

static void emit_classroom_game(const std::string& key,OrdinaryRoot root,Move known,Move added,const Rules& rules,
                                const FrozenReplyPlayers& model,uint64_t seed,int table) {
  const auto& state=root.state;
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  const OpeningMarks marks(d.observation,false);
  int cost=0;
  const auto choose=[&](size_t width) {
    const auto actions=budget_candidates(d,root.fallback,rules,width);
    return window_ranking(d,actions,root.fallback,rules,model.window,true,cost).action;
  };
  const auto frontier=budget_choice(state,rules,model.peer,root.fallback,BudgetArm::Wide,mix_seed(seed,778)).action;
  const auto actions=budget_candidates(d,root.fallback,rules,4);
  const auto first=choose(4);
  const auto add=[&](std::vector<Move>& list,Move action) {
    if (std::find(list.begin(),list.end(),action)==list.end()) list.push_back(action);
  };
  std::vector<Move> choices;
  for (Move move:{known,added,root.fallback,first,choose(8),frontier}) add(choices,move);
  if (choices.size()==1 && actions.size()>1) add(choices,actions[1]);
  std::cout<<"R "<<key<<' '<<state.ply+1<<' '<<state_hash(state)<<' '<<early_action_key(known)<<' '
           <<early_action_key(added)<<' '<<early_action_key(first)<<' '<<table<<' '<<choices.size()<<'\n';
  for (size_t i=0;i<choices.size();++i) {
    const auto first_action=choices[i];
    single_case_check(is_legal(state,first_action),"classroom action was not legal");
    auto x=student_features(d,marks,first_action,&rules);
    const auto dynamic=reply_window_features(d.observation,first_action,rules);
    single_case_check(dynamic.has_value(),"classroom lost visible information");
    x.insert(x.end(),dynamic->begin(),dynamic->end());
    std::cout<<"F "<<key<<' '<<early_action_key(first_action)<<std::setprecision(17);
    for (double f:x) std::cout<<' '<<f;
    std::cout<<'\n';
    for (int t:{0,1}) for (int stream=0;stream<6;++stream)
      route_continuation(state,first_action,rules,state.to_move,t,mix_seed(seed,33001+stream),key,state.ply+1,int(i),stream);
  }
}

static void run_ordinary_classroom(const Rules& rules) {
  FrozenReplyPlayers model;
  model.load();
  uint64_t seed, hash;
  int cut,table,original,alternative;
  while (std::cin>>seed>>cut>>table>>hash>>original>>alternative) {
    single_case_check(cut>=1 && cut<=180 && table>=0 && table<=1,"invalid decision classroom cut");
    const auto root=find_ordinary_root(seed,table,cut,rules,hash?&model:nullptr);
    const auto key=std::to_string(seed);
    if (!root.reached) {
      single_case_check(!hash,"lost recorded diagnostic root");
      std::cout<<"M "<<key<<' '<<root.state.ply<<'\n';
      continue;
    }
    auto before=root.fallback, after=root.fallback;
    if (hash) {
      single_case_check(state_hash(root.state)==hash && root.state.ply+1==cut,"classroom root drifted");
      before=archived_action(original);
      after=archived_action(alternative);
    }
    emit_classroom_game(key,root,before,after,rules,model,seed,table);
  }
  single_case_check(std::cin.eof(),"incomplete classroom manifest");
}
