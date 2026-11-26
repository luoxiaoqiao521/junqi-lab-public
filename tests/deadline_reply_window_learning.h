#pragma once

static int flag_contacts(const State& state,int viewer,const Rules& rules) {
  if (state.result!=Result::Ongoing || team_of(state.to_move)!=team_of(viewer)) return 0;
  std::vector<Move> moves;
  legal_moves(state,moves);
  return static_cast<int>(std::count_if(moves.begin(),moves.end(),[&](Move move) {
    return state.rank_at[move.to]==Flag && resolve(state.rank_at[move.from],Flag,rules)!=Outcome::DefenderWins;
  }));
}

// Record the options after actual replies. These are one-turn opportunities, not a proof of a win.
static std::optional<std::array<double,8>> reply_window_features(const Observation& view,Move first,
                                                                 const Rules& rules,int* work=nullptr) {
  if (!rules.flag_eliminates) return {};
  auto restored=visible_terminal_board(view);
  if (!restored) return {};
  auto next=*restored;
  const int viewer=view.viewer, mover=next.id_at[first.from];
  step(next,first,rules);
  if (work) ++*work;
  std::array<double,8> x{};
  const auto settled=team_score(next.result,viewer,next.match);
  x[0]=settled==1.; x[1]=settled==0.; x[2]=settled==.5;
  if (next.result!=Result::Ongoing) return x;
  if (team_of(next.to_move,next.match)==team_of(viewer,next.match)) {
    x[5]=flag_contacts(next,viewer,rules)>0;
    return x;
  }
  std::vector<Move> replies;
  legal_moves(next,replies);
  if (replies.empty()) return x;
  for (const auto reply:replies) {
    auto after=next;
    step(after,reply,rules);
    if (work) ++*work;
    const bool loses=team_score(after.result,viewer,after.match)==0.0;
    const int contacts=flag_contacts(after,viewer,rules);
    x[3]+=loses;
    x[4]+=next.alive[mover] && !after.alive[mover];
    x[5]+=contacts>0;
    x[6]+=contacts>1;
    x[7]+=(!loses && after.result==Result::Ongoing &&
           team_of(after.to_move,after.match)==team_of(viewer,after.match));
  }
  for (int i=3;i<8;++i) x[i]/=double(replies.size());
  return x;
}

static void emit_window_case(const State& root,Move a,Move b,const Rules& rules,
                             int flag,int trap,int gate,int from,int defender) {
  const auto full=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto input=[&](Move move) {
    const auto features=reply_window_features(full.observation,move,rules);
    single_case_check(features.has_value(),"full observation lost the reply window");
    const auto known=student_features(full,OpeningMarks(full.observation,false),move,&rules);
    for (const double x:known) std::cout<<' '<<x;
    for (const double x:*features) std::cout<<' '<<x;
  };
  const auto aa=full_route_proof(root,a,rules,3), bb=full_route_proof(root,b,rules,3);
  if (aa.lower<=bb.upper && bb.lower<=aa.upper) return;
  std::cout<<"P "<<flag<<' '<<trap<<' '<<gate<<' '<<from<<' '<<defender<<' '
           <<early_action_key(a)<<' '<<early_action_key(b)<<' '<<aa.lower<<' '<<aa.upper<<' '<<bb.lower<<' '<<bb.upper;
  input(a);
  input(b);
  std::cout<<'\n';
}

static State timing_root(int rank,int flag) {
    auto root=selfplay_fixture();
    root.out[2]=root.out[3]=false;
    if (flag!=56) {
      root.id_at[56]=root.seat_at[56]=kNone;
      root.rank_at[56]=Unknown;
      place_single_case_piece(root,25,flag,Flag);
    }
    place_single_case_piece(root,1,57,rank);
    place_single_case_piece(root,26,34,Platoon);
    place_single_case_piece(root,50,86,Flag);
    place_single_case_piece(root,51,115,Engineer);
    place_single_case_piece(root,75,116,Flag);
    place_single_case_piece(root,76,85,Engineer);
    return root;
}

static void reply_window_timing_examples(const Rules& rules) {
  for (const int rank:{Engineer,Bomb}) for (const int flag:{56,58}) {
    auto root=timing_root(rank,flag);
    root.no_progress=rules.no_progress_limit-1;
    const Move take{57,uint8_t(flag)}, quiet{57,52};
    const auto full=compare_visibility(root,IdentityVisibility::AllIdentities);
    std::array<int,2> outcome{};
    for (const bool ccw:{false,true}) {
      auto local=rules;
      local.ccw=ccw;
      const auto capture=full_route_proof(root,take,local,3);
      const auto hold=full_route_proof(root,quiet,local,3);
      single_case_check(capture.lower==capture.upper && hold.lower==1 && hold.upper==1,
                        "seat order timing control did not settle");
      outcome[ccw]=capture.lower;
      emit_window_case(root,take,quiet,local,int(rank==Bomb),flag,52,57,int(ccw));
    }
    single_case_check(outcome[0]+outcome[1]==2 && outcome[0]!=outcome[1] && full.actions.size()>1,
                      "the last flag race must reverse with the real seat order");
  }
}

static void reply_window_skip_examples(const Rules& rules) {
  auto local=rules;
  local.ccw=true;
  for (const int flag:{56,58}) {
    const Move take{57,uint8_t(flag)}, quiet{57,52};
    for (const bool blocked:{false,true}) {
      auto root=timing_root(Engineer,flag);
      root.no_progress=rules.no_progress_limit-1;
      place_single_case_piece(root,52,84,Mine);
      place_single_case_piece(root,53,88,Mine);
      if (blocked) {
        root.id_at[115]=root.seat_at[115]=kNone;
        root.rank_at[115]=Unknown;
        place_single_case_piece(root,51,89,Engineer);
      }
      auto branch=root;
      const auto first=step(branch,take,local);
      single_case_check(branch.result==Result::Ongoing && branch.to_move==(blocked?3:2) &&
                        bool(first.skipped_mask&(1u<<2))==blocked,
                        "guard must distinguish an unavailable ally from a retired ally");
      const auto proof=full_route_proof(root,take,local,3);
      single_case_check(proof.lower==proof.upper && proof.lower==(blocked?0:2),
                        "a blocked ally must reverse the exact race");
      emit_window_case(root,take,quiet,local,2,flag,52,57,int(blocked));
    }
  }
}

static void run_reply_window_examples(const Rules& rules) {
  auto control=selfplay_fixture();
  place_single_case_piece(control,1,4,Brigade);
  place_single_case_piece(control,26,30,Engineer);
  place_single_case_piece(control,27,21,Engineer);
  const Move bait{4,30};
  const auto visible=observe(control,0,IdentityVisibility::AllIdentities);
  const auto masked=observe(control,0,IdentityVisibility::OwnAndPublic);
  const auto window=reply_window_features(visible,bait,rules);
  single_case_check(window && (*window)[3]>0 && !reply_window_features(masked,bait,rules),
                    "next seat or partial-observation control failed");
  std::cout<<"C 1\n"<<std::setprecision(17);
  reply_window_timing_examples(rules);
  reply_window_skip_examples(rules);
  const auto& board=Board::get();
  for (int flag:{56,58}) {
    auto empty=selfplay_fixture();
    if (flag!=56) {
      empty.id_at[56]=empty.seat_at[56]=kNone;
      empty.rank_at[56]=Unknown;
      place_single_case_piece(empty,25,flag,Flag);
    }
    const auto& gates=board.road(flag);
    for (int trap:gates) for (int gate:gates) {
      if (trap==gate) continue;
      for (int from:{45,46,47,48,49,50,51,52,53,54,55,57,59}) {
        if (from==trap || from==gate || from==flag) continue;
        auto position=empty;
        place_single_case_piece(position,1,from,Engineer);
        place_single_case_piece(position,26,trap,Mine);
        const Move a{uint8_t(from),uint8_t(trap)}, b{uint8_t(from),uint8_t(gate)};
        if (!is_legal(position,a) || !is_legal(position,b)) continue;
        for (int n=20;n<69;++n) {
          if (position.seat_at[n]!=kNone || n==gate || !can_leave_node(Marshal,n)) continue;
          auto root=position;
          place_single_case_piece(root,27,n,Marshal);
          if (is_legal(root,a) && is_legal(root,b)) emit_window_case(root,a,b,rules,flag,trap,gate,from,n);
        }
      }
    }
  }
}

static void run_scored_window(int count,const StudentNetwork& model,size_t dimensions) {
  for (int i=0;i<count;++i) {
    double score[2]{};
    for (double& v:score) {
      std::vector<double> features(dimensions);
      for (double& x:features) single_case_check(bool(std::cin>>x) && std::isfinite(x),"missing window inputs");
      v=model.score(std::move(features),false);
    }
    std::cout<<"J "<<(score[1]>score[0]+1e-12)<<' '<<std::setprecision(17)<<score[0]<<' '<<score[1]<<'\n';
  }
}

enum class WindowPlayer { Path, Extension, StudentBoard, StudentWindow };

struct WindowRanking {
  Move action;
  double best, second, fallback;
};

static std::vector<double> window_input(const SeatDecision& decision,const OpeningMarks& marks,Move action,
                                       const Rules& rules,bool window,int& work) {
  auto x=student_features(decision,marks,action,&rules);
  if (window) {
    const auto facts=reply_window_features(decision.observation,action,rules,&work);
    single_case_check(facts.has_value(),"the full student encountered a masked reply window");
    x.insert(x.end(),facts->begin(),facts->end());
  }
  return x;
}

static double window_score(const SeatDecision& decision,const OpeningMarks& marks,Move action,
                           const Rules& rules,const StudentNetwork& network,bool window,int& work) {
  return network.score(window_input(decision,marks,action,rules,window,work),false);
}

static WindowRanking window_ranking(const SeatDecision& decision,const std::vector<Move>& actions,Move fallback,
                                   const Rules& rules,const StudentNetwork& network,bool window,int& work) {
  const auto& view=decision.observation;
  const OpeningMarks marks(view,false);
  WindowRanking rank{fallback,-1e9,-1e9,-1e9};
  for (const auto action:actions) {
    const auto score=window_score(decision,marks,action,rules,network,window,work);
    if (action==fallback) rank.fallback=score;
    if (score>rank.best+1e-12) {
      rank.second=rank.best;
      rank.best=score;
      rank.action=action;
    } else {
      rank.second=std::max(rank.second,score);
    }
  }
  return rank;
}

static Move window_student_choice(const State& root,Move fallback,const Rules& rules,
                                  const StudentNetwork& network,bool window,int& work) {
  const auto decision=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto actions=budget_candidates(decision,fallback,rules,4);
  return window_ranking(decision,actions,fallback,rules,network,window,work).action;
}

static void run_window_game(uint64_t seed,WindowPlayer arm,const Rules& rules,int table,const ShortWeights& peer,
                            const StudentNetwork& board,const StudentNetwork& window) {
  auto root=opening(seed,rules);
  auto rng=review_streams(seed);
  const int viewer=int(seed%4), ally=(viewer+2)%4;
  bool used[2]{};
  int chances=0, changed=0, work=0, extensions=0, improved=0, harmed=0;
  int first_change=0, old_move=kNone, new_move=kNone;
  uint64_t first_hash=0;
  int64_t micros=0;
  while (root.result==Result::Ongoing) {
    const int actor=root.to_move;
    Move fallback;
    if (table==1 && actor==ally) {
      const auto view=observe(root,actor,IdentityVisibility::AllIdentities);
      single_case_check(heuristic_move(view,rng[actor],fallback),"frozen ally had no move");
    } else fallback=route_baseline(root,rules,rng[actor]);
    auto choice=fallback;
    const int phase=root.ply>=111?1:0;
    if (actor==viewer && root.ply>=31 && !used[phase]) {
      used[phase]=true;
      ++chances;
      const auto begin=std::chrono::steady_clock::now();
      if (arm==WindowPlayer::Extension) {
        const auto scored=budget_choice(root,rules,peer,fallback,BudgetArm::Extend,mix_seed(seed,root.ply+3301));
        choice=scored.action;
        work+=scored.work.plies;
        extensions+=scored.work.extended;
      } else if (arm==WindowPlayer::StudentBoard || arm==WindowPlayer::StudentWindow) {
        const bool dynamic=arm==WindowPlayer::StudentWindow;
        choice=window_student_choice(root,fallback,rules,dynamic?window:board,dynamic,work);
      }
      micros+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
      if (!(choice==fallback)) {
        ++changed;
        const auto old_proof=full_route_proof(root,fallback,rules,2), new_proof=full_route_proof(root,choice,rules,2);
        improved+=new_proof.lower>old_proof.upper;
        harmed+=old_proof.lower>new_proof.upper;
        if (!first_change) {
          first_change=root.ply+1;
          first_hash=state_hash(root);
          old_move=early_action_key(fallback);
          new_move=early_action_key(choice);
        }
      }
    }
    single_case_check(is_legal(root,choice),"research game applied an illegal student choice");
    step(root,choice,rules);
  }
  const auto points=team_score(root.result,viewer);
  std::cout<<"G "<<seed<<' '<<viewer<<' '<<table<<' '<<int(arm)<<' '<<root.ply<<' '
           <<(points?int(*points*2):-1)<<' '<<chances<<' '<<changed<<' '<<work<<' '<<extensions<<' '
           <<improved<<' '<<harmed<<' '<<micros<<' '<<first_change<<' '<<first_hash<<' '<<old_move<<' '
           <<new_move<<' '<<state_hash(root)<<'\n';
}

static void run_window_games(const Rules& rules) {
  uint64_t start;
  int count;
  ShortWeights peer{};
  StudentNetwork board,window;
  single_case_check(bool(std::cin>>start>>count) && count>=1 && count<=24,"invalid full game budget");
  for (double& w:peer) single_case_check(bool(std::cin>>w) && std::isfinite(w),"missing full game peer");
  single_case_check(board.load(std::cin,true) && window.load(std::cin,true,true),"incompatible research game weights");
  for (uint64_t seed=start;seed<start+uint64_t(count);++seed) for (int table:{0,1})
    for (const auto arm:{WindowPlayer::Path,WindowPlayer::Extension,WindowPlayer::StudentBoard,WindowPlayer::StudentWindow})
      run_window_game(seed,arm,rules,table,peer,board,window);
}
