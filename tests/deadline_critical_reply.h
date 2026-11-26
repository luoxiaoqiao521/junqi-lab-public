#pragma once

struct CriticalReply {
  int outcome=-1, elapsed=0, changes=0, first_change=0, flag_choice=0, flag_guard=0;
  int cutoff_flag=0, flag_taken=0, cutoff_contact=0, contact_taken=0;
  double cutoff_value=0;
};

static void record_critical_cutoff(CriticalReply& out,const jq::SeatDecision& decision,
    const jq::OpeningMarks& marks,jq::Move action,int side,const jq::Rules& rules) {
  using namespace jq;
  const auto& view=decision.observation;
  for (const auto move:decision.actions) {
    if (view.seat_at(move.to)<0) continue;
    const auto mask=compatible_rank_mask(view,marks,move.to);
    const auto outcomes=collision_outcome_mask(view.rank_at(move.from),mask,rules);
    if (mask==(1u<<Flag) && outcomes && !(outcomes&outcome_bit(Outcome::DefenderWins))) {
      out.cutoff_flag=side;
      if (move==action) out.flag_taken=side;
    }
    if (outcomes==outcome_bit(Outcome::AttackerWins)) {
      out.cutoff_contact=side;
      if (move==action) out.contact_taken=side;
    }
  }
}

static jq::Move critical_replan(const jq::SeatDecision& decision,const jq::OpeningMarks& evidence,
                               jq::Move fallback,jq::Move last,const jq::Rules& rules,CriticalReply& out) {
  using namespace jq;
  if (const auto win=public_flag_finish(decision,fallback,rules)) {
    out.flag_choice+=!(*win==fallback);
    return *win;
  }
  const auto threat=visible_flag_threat(decision.observation,evidence,rules);
  if (const auto guard=visible_intercept(decision,evidence,rules,fallback,threat)) {
    ++out.flag_guard;
    return *guard;
  }
  return plan(decision,evidence,fallback,last,rules);
}

static CriticalReply critical_reply_rollout(jq::State state,jq::Move first,const jq::Rules& rules,int table,
    const jq::ShortWeights& weights,uint64_t seed,const jq::OpeningMarks& root_evidence,
    const jq::PublicReplyHistory& prefix,bool challenge) {
  using namespace jq;
  const int root=state.to_move, at=state.ply;
  PublicReplyHistory history=prefix;
  std::array<OpeningMarks,kSeats> marks{prefix.marks(),prefix.marks(),prefix.marks(),prefix.marks()};
  marks[root]=root_evidence;
  auto streams=review_streams(seed);
  CriticalReply out;
  const auto play=[&](Move move) {
    const auto before=state;
    const auto event=step(state,move,rules);
    if (!challenge && state.ply>at+9) return;
    history.advance(before,state,event,rules);
    for (int seat=0;seat<kSeats;++seat) if (!state.out[seat])
      advance_contact_ranks(marks[seat],observe(before,seat),observe(state,seat),move,event.outcome,rules);
  };
  play(first);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const int actor=d.observation.viewer, elapsed=state.ply-at;
    const auto fallback=review_move(d,root,table,true,weights,streams[actor]);
    auto action=fallback;
    if (challenge || elapsed==9) {
      const auto evidence=history.for_actor(d.observation,marks[actor]);
      if (challenge) action=critical_replan(d,evidence,fallback,history.last_moves()[actor],rules,out);
      if (!(action==fallback)) {
        if (out.changes==0) out.first_change=elapsed+1;
        ++out.changes;
      }
      if (elapsed==9) {
        record_critical_cutoff(out,d,evidence,action,team_of(actor)==team_of(root)?1:-1,rules);
        out.cutoff_value=leaf_value(state,team_of(root))+readiness_term(state,rules,team_of(root),false,RoutePriority::Joint);
      }
    }
    play(action);
  }
  out.outcome=review_outcome(state.result,root);
  out.elapsed=state.ply-at;
  return out;
}
