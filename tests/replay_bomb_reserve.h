#pragma once

static std::array<std::optional<Move>,3> bomb_candidates(const SeatDecision& d,const OpeningMarks& history,
                                                        int target,const Rules& rules,const Policies& models) {
  const auto& view=d.observation;
  const auto p=review_learned(d,models.parent);
  const auto threat=visible_flag_threat(view,history,rules);
  std::array<std::optional<Move>,3> choices;
  std::array<double,3> best{-1.,-1.,-1.};
  for (size_t i=0;i<d.actions.size();++i) {
    const auto a=d.actions[i];
    const int rank=view.rank_at(a.from);
    int role=1;
    if (rank==Bomb) {
      if (a.to!=target) continue;
      role=0;
    } else if (threat.forced>0) {
      if (view.seat_at(a.to)==kNone) {
        const auto after=project_quiet(view,a);
        if (!visible_flag_threat(after,history,rules).origins) role=2;
      } else if (threat.origins==1 && a.to==threat.target) {
        const auto outcomes=collision_outcome_mask(rank,compatible_rank_mask(view,history,a.to),rules);
        if (outcomes && !(outcomes&(1u<<static_cast<unsigned>(Outcome::DefenderWins)))) role=2;
      }
    }
    if (p[i]>best[role]) { best[role]=p[i]; choices[role]=a; }
  }
  return choices;
}

static void bomb_reserve_play(std::ostream& out,const State& root,int role,Move first,Rules rules,
                              const Policies& models,int table,uint64_t seed,int cap) {
  State state=root;
  const int learner=state.to_move, rival=table ? first_enemy_seat(root,learner,rules) : kNone;
  auto streams=review_streams(seed);
  Timing timing;
  rules.truncation_cap=std::min(rules.truncation_cap,state.ply+cap);
  int reason=0;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const int actor=state.to_move;
    auto move=actor==rival ? review_move(d,learner,2,true,models.peer,streams[actor])
                           : select(d,learner,1,0,models,streams[actor]);
    const bool initial=state.ply==root.ply;
    if (initial) move=first;
    const auto event=step(state,move,rules);
    if (initial && !role) expect_value(event.outcome==Outcome::BothRemoved,"bomb did not remove both pieces");
    timing.note(event,team_of(learner),state.ply-root.ply);
    reason=int(event.end_reason);
  }
  out<<' '<<state.ply-root.ply<<' '<<review_outcome(state.result,learner)<<' '<<reason
     <<' '<<timing.own_exit<<' '<<timing.enemy_exit<<'\n';
}

static void bomb_reserve_root(std::ostream& out,const std::string& key,const State& state,
    const BombWindow& w,const SeatDecision& d,int target,const OpeningMarks& history,
    const Rules& rules,const Policies& models,int cap,int stream_offset) {
  const auto choices=bomb_candidates(d,history,target,rules,models);
  expect_value(choices[0].has_value(),"missing legal bomb branch");
  out<<"K "<<key<<' '<<w.ply<<' '<<w.ally<<' '<<state.ply<<' '<<d.observation.viewer;
  for (const auto& a:choices) out<<' '<<(a ? int(a->from)*kNodes+a->to : -1);
  out<<'\n';
  for (int table:{0,1}) for (int rep:{0,1}) {
    const auto seed=mix_seed(std::stoull(key.substr(0,16),nullptr,16),w.ply*16+w.ally*4+rep+stream_offset);
    for (int role=0;role<3;++role) if (choices[role]) {
      out<<"F "<<key<<' '<<w.ply<<' '<<w.ally<<' '<<table<<' '<<rep<<' '<<role;
      bomb_reserve_play(out,state,role,*choices[role],rules,models,table,seed,cap);
    }
  }
}
