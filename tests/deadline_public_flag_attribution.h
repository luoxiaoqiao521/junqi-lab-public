#pragma once

enum class RefereeFlagReply { Clear, Threat, Win, Draw, Loss, Unknown };

static RefereeFlagReply referee_flag_reply(State state,Move action,const Rules& rules) {
  const int actor=state.to_move;
  step(state,action,rules);
  if (state.result!=Result::Ongoing) {
    const auto score=team_score(state.result,actor);
    if (!score) return RefereeFlagReply::Unknown;
    if (*score==1) return RefereeFlagReply::Win;
    return *score==0.5?RefereeFlagReply::Draw:RefereeFlagReply::Loss;
  }
  if (team_of(state.to_move)==team_of(actor)) return RefereeFlagReply::Clear;
  std::vector<Move> replies;
  legal_moves(state,replies);
  for (auto move:replies)
    if (state.rank_at[move.to]==Flag &&
        resolve(state.rank_at[move.from],Flag,rules)!=Outcome::DefenderWins)
      return RefereeFlagReply::Threat;
  return RefereeFlagReply::Clear;
}

struct FlagAttribution {
  uint64_t seed;
  int table;
  int last_contact_ply[kSeats]{}, last_target[kSeats]{};
  int eligible[2]{}, selected[2]{}, removed[2]{};
  int flag_offers=0, flag_taken=0, known_offers=0, known_taken=0;
  bool awaiting_enemy=false;

  bool report_risk(const State& root,const SeatDecision& decision,const OpeningMarks& marks,Move fallback,
                   Move base,const Rules& rules) const {
    const auto exact=referee_flag_reply(root,base,rules);
    int in_list=0, outside=0, still_risk=0, unresolved=0, proved=0, draw=0, unknown=0;
    int first_safe=-1, first_public=-1;
    if (exact==RefereeFlagReply::Threat) {
      const auto list=budget_candidates(decision,fallback,rules,4);
      for (auto move:decision.actions) {
        const auto value=referee_flag_reply(root,move,rules);
        if (value!=RefereeFlagReply::Clear && value!=RefereeFlagReply::Win) {
          draw+=value==RefereeFlagReply::Draw;
          unknown+=value==RefereeFlagReply::Unknown;
          continue;
        }
        const auto claim=public_flag_reply(decision.observation,marks,move,rules);
        const bool listed=std::find(list.begin(),list.end(),move)!=list.end();
        in_list+=listed;
        outside+=!listed;
        if (first_safe<0) first_safe=early_action_key(move);
        if (claim==FlagReply::Clear && first_public<0) first_public=early_action_key(move);
        still_risk+=claim==FlagReply::Risk;
        unresolved+=claim==FlagReply::Unresolved;
        proved+=claim==FlagReply::Clear;
      }
    }
    std::cout<<"R "<<seed<<' '<<table<<' '<<root.ply+1<<' '<<int(root.to_move)<<' '
             <<early_action_key(base)<<' '<<int(exact)<<' '<<decision.actions.size()<<' '
             <<in_list<<' '<<outside<<' '<<still_risk<<' '<<unresolved<<' '<<proved<<' '
             <<draw<<' '<<unknown<<' '<<first_safe<<' '<<first_public<<'\n';
    return exact==RefereeFlagReply::Threat;
  }

  void opponent_reply(const SeatDecision& decision,Move chosen,bool captured,const Rules& rules) {
    const auto& view=decision.observation;
    const bool known=std::any_of(decision.actions.begin(),decision.actions.end(),[&](Move move) {
      return view.rank_at(move.to)==Flag && resolve(view.rank_at(move.from),Flag,rules)!=Outcome::DefenderWins;
    });
    ++flag_offers;
    flag_taken+=captured;
    known_offers+=known;
    known_taken+=captured && view.rank_at(chosen.to)==Flag;
  }

  void followthrough(const SeatDecision& decision,Move chosen,int side,Outcome outcome) {
    const auto& view=decision.observation;
    const int actor=view.viewer, partner=(actor+2)%kSeats;
    const bool window=last_contact_ply[partner]>0 && view.ply+1-last_contact_ply[partner]<=3;
    if (window && std::any_of(decision.actions.begin(),decision.actions.end(),[&](Move move) {
      return move.to==last_target[partner] && view.seat_at(move.to)>=0;
    })) {
      ++eligible[side];
      if (chosen.to==last_target[partner]) {
        ++selected[side];
        removed[side]+=outcome==Outcome::AttackerWins || outcome==Outcome::BothRemoved;
      }
    }
    if (view.seat_at(chosen.to)>=0) {
      last_contact_ply[actor]=view.ply+1;
      last_target[actor]=chosen.to;
    }
  }

  void after_play(const SeatDecision& decision,Move move,bool own,Outcome outcome,
                  bool captured,bool next_enemy,const Rules& rules) {
    followthrough(decision,move,!own,outcome);
    if (awaiting_enemy) {
      single_case_check(!own,"exact hostile reply must belong to an enemy");
      opponent_reply(decision,move,captured,rules);
    }
    awaiting_enemy=next_enemy;
  }

  void print() const {
    std::cout<<"P "<<seed<<' '<<table;
    for (int side:{0,1}) std::cout<<' '<<eligible[side]<<' '<<selected[side]<<' '<<removed[side];
    std::cout<<' '<<flag_offers<<' '<<flag_taken<<' '<<known_offers<<' '<<known_taken<<'\n';
  }
};
