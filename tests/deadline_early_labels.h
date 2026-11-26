#pragma once

struct EarlyPrefix {
  jq::State state;
  jq::PublicReplyHistory reply_history;
  std::array<jq::OpeningMarks,jq::kSeats> marks;
  std::array<jq::Rng,jq::kSeats> streams;
  uint64_t trace;
  int friendly_exit=0, enemy_exit=0, friendly_reason=0, enemy_reason=0;
  jq::EndReason ending=jq::EndReason::None;
  bool student_window[2]{};

  EarlyPrefix(jq::State start,uint64_t seed):state(start),reply_history(start,true),
    marks{jq::OpeningMarks(jq::observe(start,0),true),jq::OpeningMarks(jq::observe(start,1),true),
          jq::OpeningMarks(jq::observe(start,2),true),jq::OpeningMarks(jq::observe(start,3),true)},
    streams(jq::review_streams(seed)),trace(seed) {}

  jq::Outcome play(jq::Move action,int learner,const jq::Rules& rules,jq::StepInfo* detail=nullptr) {
    using namespace jq;
    const auto before=state;
    const auto event=step(state,action,rules);
    if (detail) *detail=event;
    const auto flag_capture=settled_flag_capture(observe(before,kNone),event,rules);
    reply_history.advance(before,state,event,rules,flag_capture);
    trace=mix_seed(trace,uint64_t(action.from) | (uint64_t(action.to)<<8) |
                   (uint64_t(event.seat)<<16) | (uint64_t(event.outcome)<<18) |
                   (uint64_t(event.eliminated_mask)<<22));
    for (int seat=0;seat<kSeats;++seat) {
      if (!state.out[seat])
        advance_contact_ranks(marks[seat],observe(before,seat),observe(state,seat),action,event.outcome,rules,flag_capture);
      if (!(event.eliminated_mask&(1u<<seat))) continue;
      const bool friendly=team_of(seat)==team_of(learner);
      auto& ply=friendly?friendly_exit:enemy_exit;
      auto& reason=friendly?friendly_reason:enemy_reason;
      if (!ply) { ply=state.ply; reason=static_cast<int>(event.exit_reason[seat]); }
    }
    ending=event.end_reason;
    return event.outcome;
  }
};

static jq::Move early_normal(EarlyPrefix& game,const jq::SeatDecision& d,int learner,int table,
    const jq::Rules& rules,const jq::ShortWeights& parent,const jq::ShortWeights& history,int partner=1,
    std::vector<std::pair<jq::Move,int>>* scores=nullptr) {
  using namespace jq;
  const int seat=d.observation.viewer;
  if (seat==(learner+2)%kSeats && partner==3)
    return review_move(d,learner,0,true,parent,game.streams[seat]);
  if (seat==(learner+2)%kSeats && partner==2)
    return review_move(d,learner,1,true,parent,game.streams[seat]);
  const auto root=short_move(d.observation,d.actions,seat==learner?parent:history,game.streams[seat]);
  if (seat==learner || (table && team_of(seat)!=team_of(learner)))
    return plan(d,game.marks[seat],root,game.reply_history.last_moves()[seat],rules,scores);
  return root;
}

static int early_action_key(jq::Move move) { return move.from*jq::kNodes+move.to; }

static void early_action_label(EarlyPrefix branch,jq::Move action,int learner,int table,
    int context,uint64_t seed,const jq::Rules& rules,const jq::ShortWeights& parent,const jq::ShortWeights& history,
    int partner,uint64_t reply_salt=0) {
  using namespace jq;
  if (reply_salt) branch.streams=review_streams(mix_seed(branch.trace,reply_salt));
  const auto initial_trace=branch.trace;
  const int at=branch.state.ply;
  branch.play(action,learner,rules);
  const int next=branch.state.result==Result::Ongoing?branch.state.to_move:kNone;
  int partner_turn=0, partner_contact=0, enemy_turn=0, enemy_contact=0;
  while (branch.state.result==Result::Ongoing) {
    const auto d=seat_decision(branch.state);
    const auto next_move=early_normal(branch,d,learner,table,rules,parent,history,partner);
    const int seat=d.observation.viewer;
    const bool contact=d.observation.seat_at(next_move.to)>=0;
    if (!partner_turn && (seat+2)%kSeats==learner) {
      partner_turn=branch.state.ply+1;
      partner_contact=contact;
    }
    if (!enemy_turn && team_of(seat)!=team_of(learner)) {
      enemy_turn=branch.state.ply+1;
      enemy_contact=contact;
    }
    branch.play(next_move,learner,rules);
  }
  std::cout<<"A "<<seed<<' '<<learner<<' '<<context<<' '<<initial_trace<<' '<<at<<' '<<early_action_key(action)
           <<' '<<branch.state.ply<<' '<<review_outcome(branch.state.result,learner)
           <<' '<<static_cast<int>(branch.ending)<<' '<<branch.trace<<' '<<next
           <<' '<<partner_turn<<' '<<partner_contact<<' '<<enemy_turn<<' '<<enemy_contact
           <<' '<<branch.friendly_exit<<' '<<branch.friendly_reason<<' '<<branch.enemy_exit<<' '<<branch.enemy_reason<<'\n';
}

static void early_export(EarlyPrefix& game,const jq::SeatDecision& d,jq::Move baseline,
    int learner,int table,int window,uint64_t seed,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& history,int partner,uint64_t reply_salt=0) {
  const auto& actor=game.marks[learner];
  const auto& view=d.observation;
  const auto marks=game.reply_history.for_actor(view,actor);
  const auto clock=view.quiet_clock;
  const int context=2*table+window;
  const auto shortlist=response_candidates(d,marks,baseline,rules,public_rollout_seed(view,actor,clock));
  ResponseWork old,joint;
  const auto first=response_choice(d,actor,game.reply_history,baseline,rules,history,clock,old,!game.enemy_exit);
  const auto second=response_choice(d,actor,game.reply_history,baseline,rules,history,clock,joint,
                                    !game.enemy_exit,RoutePriority::Joint);
  single_case_check(old.samples==joint.samples && shortlist.size()<=6 &&
                    std::find(shortlist.begin(),shortlist.end(),first)!=shortlist.end() &&
                    std::find(shortlist.begin(),shortlist.end(),second)!=shortlist.end(),
                    "teacher comparison changed its candidates or budget");
  std::cout<<"T "<<seed<<' '<<learner<<' '<<context<<' '<<early_action_key(baseline)<<' '
           <<early_action_key(first)<<' '<<early_action_key(second)<<' '<<old.samples<<' '<<shortlist.size()<<'\n';
  print_learning_root(std::cout,d,marks,clock,seed,learner,context,game.trace);
  for (const auto action:shortlist)
    early_action_label(game,action,learner,table,context,seed,rules,parent,history,partner,reply_salt);
}

static void early_learning_trial(uint64_t seed,int learner,int table,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& history,int partner=1) {
  using namespace jq;
  EarlyPrefix game(opening(seed,rules),seed);
  bool exported[2]{};
  while (game.state.result==Result::Ongoing && game.state.ply<100) {
    const auto d=seat_decision(game.state);
    const auto baseline=early_normal(game,d,learner,table,rules,parent,history,partner);
    const int window=game.state.ply>=56?1:0;
    if (d.observation.viewer==learner && game.state.ply>=24 && !exported[window]) {
      early_export(game,d,baseline,learner,table,window,seed,rules,parent,history,partner);
      exported[window]=true;
    }
    game.play(baseline,learner,rules);
  }
  std::cout<<"G "<<seed<<' '<<learner<<' '<<table<<' '<<exported[0]<<' '<<exported[1]<<'\n';
}
