#pragma once

struct CrossSectorFootprint {
  int exits=0, flag_attacks=0;
};

static CrossSectorFootprint cross_sector_footprint(const State& state,int seat,int& generated) {
  if (state.out[seat] || state.result!=Result::Ongoing) return {};
  const auto& board=Board::get();
  std::vector<Move> moves;
  gen_moves(StateView{state},seat,moves);
  generated+=int(moves.size());
  std::array<bool,kNodes> reachable{};
  CrossSectorFootprint footprint;
  for (Move move:moves) {
    if (board.node(move.to).seat!=seat && !reachable[move.to]) {
      ++footprint.exits;
      reachable[move.to]=true;
      if (state.rank_at[move.to]==Flag) ++footprint.flag_attacks;
    }
  }
  return footprint;
}

static std::vector<Move> nominate_quiet_options(const State& root,const Rules& rules,const SeatDecision& d,
                                              const std::vector<Move>& local,const std::string& key,int& generated,
                                              int& probes) {
  auto slots=local;
  slots.resize(std::min(size_t(6),slots.size()));
  const int ally=(root.to_move+2)%4;
  int enemy=next_seat(root.to_move,rules);
  while (root.out[enemy] || team_of(enemy)==team_of(root.to_move)) enemy=next_seat(enemy,rules);
  const auto friend_before=cross_sector_footprint(root,ally,generated);
  const auto enemy_before=cross_sector_footprint(root,enemy,generated);
  struct QuietOption {Move action; int ally_extra, enemy_less, attack_less;};
  std::vector<QuietOption> quiet;
  for (Move action:d.actions) {
    if (root.seat_at[action.to]!=kNone || std::find(slots.begin(),slots.end(),action)!=slots.end()) continue;
    auto future=root;
    ++probes;
    step(future,action,rules);
    if (future.result!=Result::Ongoing) continue;
    const auto friend_after=cross_sector_footprint(future,ally,generated);
    const auto enemy_after=cross_sector_footprint(future,enemy,generated);
    quiet.push_back({action,friend_after.exits-friend_before.exits,enemy_before.exits-enemy_after.exits,
                       enemy_before.flag_attacks-enemy_after.flag_attacks});
  }
  // Reserve one slot for a reduction in the next enemy's options and one for allied access.
  for (bool support:{false,true}) {
    int top=0;
    const QuietOption* best=nullptr;
    for (const auto& q:quiet) {
      const int score=support?q.ally_extra:128*q.attack_less+q.enemy_less;
      if (score>top && q.attack_less>=0 && std::find(slots.begin(),slots.end(),q.action)==slots.end()) {
        top=score;
        best=&q;
      }
    }
    if (best && slots.size()<8) {
      slots.push_back(best->action);
      const int outside=std::find(local.begin(),local.end(),best->action)==local.end();
      std::cout<<"P "<<key<<' '<<int(support)<<' '<<early_action_key(best->action)<<' '<<best->ally_extra<<' '
               <<best->enemy_less<<' '<<best->attack_less<<' '<<outside<<'\n';
    }
  }
  for (Move m:local) if (slots.size()<local.size() && std::find(slots.begin(),slots.end(),m)==slots.end()) slots.push_back(m);
  return slots;
}

static Move quiet_student(const State& state,Move fallback,const Rules& rules,const StudentNetwork& model) {
  auto exact=state;
  step(exact,fallback,rules);
  int work=0;
  return team_score(exact.result,state.to_move)==1.0?fallback:window_student_choice(state,fallback,rules,model,true,work);
}

static void run_quiet_options(const Rules& rules) {
  FrozenReplyPlayers models;
  models.load();
  StudentNetwork child;
  single_case_check(child.load(std::cin,true,true),"quiet-options child missing");
  uint64_t seed;
  int table;
  while (std::cin>>seed>>table) {
    single_case_check(table==1 || table==2,"invalid failure replay context");
    auto root=opening(seed,rules);
    auto rng=review_streams(seed);
    const int learner=int(seed%4);
    Move fallback,a,b;
    bool found=false;
    while (root.result==Result::Ongoing) {
      const int actor=root.to_move;
      fallback=full_study_fallback(root,rules,learner,table,rng[actor]);
      if (actor==learner) {
        a=quiet_student(root,fallback,rules,models.window);
        b=quiet_student(root,fallback,rules,child);
        if (!(a==b)) { found=true; break; }
        step(root,a,rules);
      } else step(root,fallback,rules);
    }
    const std::string key=std::to_string(seed);
    if (!found) {
      std::cout<<"M "<<key<<' '<<root.ply<<'\n';
      continue;
    }
    const auto d=compare_visibility(root,IdentityVisibility::AllIdentities);
    const auto no=budget_candidates(d,fallback,rules,8);
    std::cout<<"R "<<key<<' '<<root.ply+1<<' '<<state_hash(root)<<' '<<table<<' '<<early_action_key(a)<<' '
             <<early_action_key(b)<<' '<<early_action_key(fallback)<<' '<<d.actions.size()<<'\n';
    std::vector<Move> to_review{a,b};
    for (int arm:{0,1,2}) {
      const auto start=std::chrono::steady_clock::now();
      int generated=0,probes=0;
      auto shortlist=no;
      if (arm==0) shortlist.resize(std::min(size_t(4),shortlist.size()));
      if (arm==2) shortlist=nominate_quiet_options(root,rules,d,no,key,generated,probes);
      const OpeningMarks marks(d.observation,false);
      const PublicReplyHistory history(root,false);
      ResponseAudit work{6,0,0,false,true};
      const int samples=arm==0?4:2;
      double highest=-1e9;
      Move best=fallback;
      for (Move action:shortlist) {
        ResponseValue value;
        for (int i=0;i<samples;++i)
          response_trial(root,action,marks,history,rules,models.peer,i%2,mix_seed(seed,17930+i),value,false,
                         RoutePriority::Old,true,&work);
        if (value.value>highest+1e-9) {
          highest=value.value;
          best=action;
        }
      }
      const auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-start).count();
      if (std::find(to_review.begin(),to_review.end(),best)==to_review.end()) to_review.push_back(best);
      std::cout<<"A "<<key<<' '<<arm<<' '<<early_action_key(best)<<' '<<shortlist.size()<<' '<<samples<<' '
               <<work.plies<<' '<<probes<<' '<<generated<<' '<<elapsed<<'\n';
    }
    for (size_t i=0;i<to_review.size();++i) for (int t:{0,1}) for (int stream=0;stream<8;++stream)
      route_continuation(root,to_review[i],rules,learner,t,mix_seed(seed,67180+stream),key,root.ply+1,int(i),stream);
  }
  single_case_check(std::cin.eof(),"incomplete quiet-options manifest");
}
