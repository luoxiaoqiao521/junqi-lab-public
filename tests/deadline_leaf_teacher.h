#pragma once

// Full-information research only. The saved model must never consume a referee truth state in dark modes.
static std::array<double,36> leaf_teacher_features(const State& state,int view) {
  std::array<double,36> x{};
  x[0]=leaf_value(state,team_of(view));
  x[1]=state.ply;
  x[2]=state.no_progress;
  x[3+(state.to_move-view+4)%4]=1;
  for (int i=0;i<kSeats;++i) {
    const int seat=(view+i)%4;
    const int k=7+7*i;
    x[k]=state.out[seat];
    for (int id=seat*kPiecesPerSeat;id<(seat+1)*kPiecesPerSeat;++id) {
      if (!state.alive[id] || !can_leave_node(state.rank_of[id],state.node_of[id])) continue;
      ++x[k+1];
      if (state.rank_of[id]>=Division) ++x[k+2];
      if (state.rank_of[id]==Bomb) ++x[k+3];
      if (state.rank_of[id]==Engineer) ++x[k+4];
    }
    if (state.out[seat]) continue;
    std::vector<Move> moves;
    gen_moves(StateView{state},seat,moves);
    for (Move move:moves) {
      if (state.rank_at[move.to]==Flag) ++x[k+5];
      if (Board::get().node(move.to).seat!=seat) ++x[k+6];
    }
  }
  x[35]=int(state.result==Result::Ongoing);
  return x;
}

static Result complete_leaf(State state,const Rules& rules,int view,int table,uint64_t seed) {
  auto rng=review_streams(seed);
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    step(state,full_study_fallback(state,rules,view,table,rng[actor]),rules);
  }
  return state.result;
}

static void emit_leaf_experience(const State& root,Move fallback,const Rules& rules,const ShortWeights& weights,
                                 uint64_t seed,const std::string& key,bool recheck) {
  const int view=root.to_move;
  const auto d=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto shortlist=budget_candidates(d,fallback,rules,4);
  const OpeningMarks marks(d.observation,false);
  const PublicReplyHistory history(root,false);
  std::cout<<"R "<<key<<' '<<root.ply+1<<' '<<state_hash(root)<<' '<<view<<' '
           <<shortlist.size()<<' '<<d.actions.size()<<'\n';
  for (Move action:shortlist) {
    const auto id=early_action_key(action);
    auto acted=root;
    step(acted,action,rules);
    for (int i=0;i<4;++i) {
      ResponseBranch branch(root,marks,history,mix_seed(seed,17930+i));
      branch.play(action,rules,true);
      ResponseAudit work{6,0,0,false,true};
      continue_response(branch,rules,weights,i%2,6,true,&work);
      const auto at=branch.sample;
      ResponseValue base;
      record_leaf(base,at,view,root.ply,false,rules,RoutePriority::Old);
      const int outcome=at.result==Result::Ongoing?-2:review_outcome(at.result,view);
      const auto begin=std::chrono::steady_clock::now();
      const auto x=leaf_teacher_features(at,view);
      const auto micros=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-begin).count();
      std::cout<<"L "<<key<<' '<<id<<' '<<i<<' '<<at.ply-root.ply<<' '<<state_hash(at)<<' '<<outcome<<' '
               <<review_outcome(acted.result,view)<<' '<<std::setprecision(17)<<base.value<<' '<<micros;
      for (double value:x) std::cout<<' '<<value;
      std::cout<<'\n';
      for (int table:{0,1,2,3}) for (int j=0;j<3;++j) {
        const auto end=complete_leaf(at,rules,view,table,mix_seed(seed,(recheck?74100:58100)+i*4+j));
        std::cout<<"Y "<<key<<' '<<id<<' '<<i<<' '<<table<<' '<<j<<' '
                 <<review_outcome(end,view)<<'\n';
      }
    }
    for (int table:{0,1,2,3}) for (int j=0;j<8;++j) {
      const auto end=complete_leaf(acted,rules,view,table,mix_seed(seed,(recheck?84569:68569)+j));
      std::cout<<"G "<<key<<' '<<id<<' '<<table<<' '<<j<<' '<<review_outcome(end,view)<<'\n';
    }
  }
}

static void run_leaf_teacher_data(const Rules& rules,bool recheck) {
  visit_reached_roots(rules,[&](const State& root,Move move,const FrozenReplyPlayers& parent,
                                const StudentNetwork&,uint64_t seed,const std::string& key,int) {
    emit_leaf_experience(root,move,rules,parent.peer,seed,key,recheck);
  });
}
