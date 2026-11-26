#pragma once

struct DarkOpeningPath {
  EarlyPrefix game;
  std::vector<Observation> observations;
  std::vector<Move> moves;
  std::vector<Outcome> collisions;
  std::vector<bool> flag_retirements;
  int first;

  DarkOpeningPath(State start,int learner):game(start,0),observations{observe(start,learner)},first(start.to_move) {
    game.reply_history=PublicReplyHistory(start,true,true);
  }

  void play(Move move,int learner,const Rules& rules) {
    moves.push_back(move);
    StepInfo event;
    collisions.push_back(game.play(move,learner,rules,&event));
    flag_retirements.push_back(std::find(std::begin(event.exit_reason),std::end(event.exit_reason),
                                        ExitReason::FlagRemoved)!=std::end(event.exit_reason));
    observations.push_back(observe(game.state,learner));
  }
};

static std::optional<EarlyPrefix> replay_dark_opening(const DarkOpeningPath& path,int learner,const Rules& rules,
    const std::array<int8_t,kNodes>& opening_ranks,PrefixReplayWork* report=nullptr) {
  const auto failed=[&](int ply,const char* reason)->std::optional<EarlyPrefix> {
    if (report) { report->failed_ply=ply; report->reason=reason; }
    return std::nullopt;
  };
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (int s=0;s<kSeats;++s) {
    for (int i=0;i<kPiecesPerSeat;++i) layouts[s][i]=opening_ranks[layout_slot_node(s,i)];
    if (!valid_layout(layouts[s])) return failed(0,"deployment");
  }
  State start;
  init_state(start,layouts,path.first,rules);
  EarlyPrefix world(start,0);
  world.reply_history=PublicReplyHistory(start,true,true);
  if (!same_observation(observe(start,learner),path.observations.front())) return failed(0,"opening-view");
  for (size_t i=0;i<path.moves.size();++i) {
    const auto move=path.moves[i];
    const int ply=static_cast<int>(i)+1;
    if (report) ++report->moves;
    if (!is_legal(world.state,move)) return failed(ply,"route");
    StepInfo event;
    if (world.play(move,learner,rules,&event)!=path.collisions[i]) return failed(ply,"collision");
    const bool flag_retired=std::find(std::begin(event.exit_reason),std::end(event.exit_reason),
                                      ExitReason::FlagRemoved)!=std::end(event.exit_reason);
    if (flag_retired!=path.flag_retirements[i]) return failed(ply,"flag-event");
    const auto actual=observe(world.state,learner);
    const auto& expected=path.observations[i+1];
    if (!same_observation(actual,expected)) return failed(ply,actual.turn!=expected.turn?"seat-control":
                                  actual.revealed_flags!=expected.revealed_flags?"flag-event":"observer-event");
  }
  if (!equal_dark_actor(world,path.game,learner)) return failed(world.state.ply,"actor-history");
  if (seat_decision(world.state).actions!=seat_decision(path.game.state).actions) return failed(world.state.ply,"legal-set");
  return world;
}

struct DarkOpeningWorlds {
  std::array<std::optional<EarlyPrefix>,2> sides;
  int nodes=0,contradicted=0,unresolved=0,replay_failed=0;
};

static DarkOpeningWorlds opening_reply_worlds(const DarkOpeningPath& path,const SeatDecision& d,const Rules& rules,
                                             int square,uint64_t seed) {
  DarkOpeningWorlds result;
  const auto& view=d.observation;
  const auto& game=path.game;
  for (int side=0;side<2;++side) for (int i=0;i<3 && !result.sides[side];++i) {
    std::array<unsigned,kNodes> pins{};
    pins[square]=side?(1u<<Bomb):all_evidence_ranks&~(1u<<Bomb);
    std::array<int8_t,kNodes> initial{};
    Rng rng(mix_seed(seed,200+i));
    CaptureSearch attempt{384,&rng,&initial};
    const auto status=game.reply_history.solve_candidate(view,game.marks[view.viewer],path.observations.front(),pins,attempt);
    result.nodes+=384-attempt.remaining;
    if (status==CaptureCompatibility::Contradicted) { ++result.contradicted; break; }
    if (status==CaptureCompatibility::Unresolved) { ++result.unresolved; continue; }
    result.sides[side]=replay_dark_opening(path,view.viewer,rules,initial);
    if (!result.sides[side]) ++result.replay_failed;
  }
  return result;
}

struct DarkOpeningDecision {
  std::array<Move,5> moves;
  int short_steps=0,final_steps=0,forced=0,unresolved=0;
};

static DarkOpeningDecision opening_reply_decision(const DarkOpeningWorlds& worlds,Move base,Move alt,
    double p,const Rules& rules,int learner,int table,uint64_t seed) {
  DarkOpeningDecision result{{base,base,base,base,base}};
  double short_value[2][2]{},lower[2][2]{},upper[2][2]{};
  for (int m=0;m<2;++m) for (int s=0;s<2;++s) for (int pressure=0;pressure<2;++pressure) {
    const auto& world=*worlds.sides[s];
    const auto line=dark_reply_checkpoint(world,m?alt:base,rules,table,seed,pressure!=0,48);
    const auto& checkpoint=line.game;
    const double weight=s?p:1-p;
    short_value[pressure][m]+=weight*dark_leaf_estimate(checkpoint.state,learner);
    result.short_steps+=checkpoint.state.ply-world.state.ply;
    result.forced+=line.reply!=line.natural;
    for (int i=0;i<2;++i) {
      auto continuation=checkpoint;
      auto streams=review_streams(mix_seed(seed,41011+i));
      dark_continue(continuation,rules,learner,table,world.state.ply+400,streams);
      const auto outcome=review_outcome(continuation.state.result,learner);
      result.final_steps+=continuation.state.ply-checkpoint.state.ply;
      lower[pressure][m]+=(outcome<0?0.:.25*outcome)*weight;
      upper[pressure][m]+=(outcome<0?.5:.25*outcome)*weight;
    }
  }
  const Move natural=short_value[0][1]>short_value[0][0]+1e-8?alt:base;
  for (int arm=0;arm<2;++arm) {
    result.moves[1+2*arm]=short_value[arm][1]>short_value[arm][0]+1e-8?alt:base;
    const auto lo=lower[arm][1]-upper[arm][0],hi=upper[arm][1]-lower[arm][0];
    const bool unresolved=lo<=1e-8 && hi>=-1e-8 && hi-lo>1e-8;
    result.moves[2+2*arm]=lo>1e-8?alt:unresolved?natural:base;
    result.unresolved+=unresolved;
  }
  return result;
}

static void emit_dark_opening(uint64_t seed,const DarkOpeningPath& path,const Rules& rules,int learner,int table,
    int learner_turns,const DarkOpeningWorlds& worlds,const DarkOpeningDecision& choice,
    const std::array<Rng,kSeats>& rngs,bool triggered) {
  std::cout<<"G "<<seed<<' '<<learner<<' '<<table<<' '<<int(triggered)<<' '<<path.game.state.ply<<' '<<learner_turns
           <<' '<<worlds.nodes<<' '<<worlds.contradicted<<' '<<worlds.unresolved<<' '<<worlds.replay_failed
           <<' '<<int(bool(worlds.sides[0]))<<' '<<int(bool(worlds.sides[1]))<<' '<<choice.short_steps
           <<' '<<choice.final_steps<<' '<<choice.forced<<' '<<choice.unresolved;
  std::vector<std::pair<Move,EarlyPrefix>> outcomes;
  for (const auto move:choice.moves) {
    auto prior=std::find_if(outcomes.begin(),outcomes.end(),[&](const auto& item){return item.first==move;});
    if (prior==outcomes.end()) {
      auto output=path.game;
      auto streams=rngs;
      if (triggered) output.play(move,learner,rules);
      dark_continue(output,rules,learner,table,400,streams);
      outcomes.emplace_back(move,std::move(output));
      prior=outcomes.end()-1;
    }
    const auto& end=prior->second;
    std::cout<<' '<<(triggered?early_action_key(move):-1)<<' '<<review_outcome(end.state.result,learner)
             <<' '<<end.state.ply<<' '<<end.trace;
  }
  std::cout<<'\n';
}

static void audit_dark_opening(uint64_t seed,const Rules& rules,int learner,int table,int first_ply) {
  DarkOpeningPath path(opening(seed,rules),learner);
  auto streams=review_streams(mix_seed(seed,53017));
  int learner_turns=0;
  while (path.game.state.result==Result::Ongoing && path.game.state.ply<400) {
    const auto decision=seat_decision(path.game.state);
    const auto& view=decision.observation;
    const auto actor=view.viewer;
    const auto base=dark_actor_move(path.game,decision,rules,learner,table,streams[actor]);
    learner_turns+=actor==learner;
    if (actor==learner && view.ply>=first_ply) {
      const auto evidence=path.game.reply_history.for_actor(view,path.game.marks[learner]);
      const auto focus=directed_contact(decision,evidence,rules,0);
      if (focus.kind==0 && decision.actions.size()>1) {
        auto alt=focus.move;
        if (alt==base) alt=budget_candidates(decision,base,rules,2).back();
        const auto salt=mix_seed(seed,public_rollout_seed(view,evidence,view.quiet_clock));
        const auto worlds=opening_reply_worlds(path,decision,rules,focus.move.to,salt);
        DarkOpeningDecision choice{{base,base,base,base,base}};
        if (worlds.sides[0] && worlds.sides[1]) {
          const auto possible=compatible_rank_mask(view,evidence,focus.move.to);
          int slots=0;
          for (int r=Flag;r<RankCount;++r) if (possible&(1u<<r)) slots+=kRankCount[r];
          single_case_check(slots>2 && !(alt==base),"unsupported natural dark contact");
          choice=opening_reply_decision(worlds,base,alt,2./slots,rules,learner,table,salt);
        }
        emit_dark_opening(seed,path,rules,learner,table,learner_turns,worlds,choice,streams,true);
        return;
      }
    }
    path.play(base,learner,rules);
  }
  emit_dark_opening(seed,path,rules,learner,table,learner_turns,{}, {},streams,false);
}

static void run_dark_opening_reply(const char* rules_path,bool compare_windows) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(rules_path),rules,error),"invalid full opening research rules");
  uint64_t seed;
  int learner,table;
  while (std::cin>>seed>>learner>>table) {
    single_case_check(learner>=0 && learner<kSeats && table>=0 && table<4,"invalid natural opening assignment");
    for (int first_ply:{0,72}) {
      if (!compare_windows && first_ply!=0) break;
      if (compare_windows) std::cout<<"W "<<first_ply<<' ';
      audit_dark_opening(seed,rules,learner,table,first_ply);
    }
  }
  single_case_check(std::cin.eof(),"incomplete natural opening request");
}
