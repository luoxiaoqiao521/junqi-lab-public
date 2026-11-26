#pragma once

struct ConversionPrefix {
  EarlyPrefix game;
  int early_at=0, early_move=-1;
};

static uint64_t conversion_marks(uint64_t hash,const OpeningMarks& marks);

struct ConversionJournal {
  std::vector<int> actions;
  std::array<uint64_t,kSeats> views{};

  void play(EarlyPrefix& game,Move move,const ExchangePolicy& p) {
    StepInfo event;
    game.play(move,p.learner,p.rules,&event);
    actions.push_back(early_action_key(move));
    uint64_t public_step=uint64_t(early_action_key(move));
    for (const auto value:{int(event.seat),int(event.outcome),int(event.eliminated_mask),int(event.revealed_mask),
                           int(event.skipped_mask),int(event.end_reason)}) public_step=mix_seed(public_step,uint64_t(value));
    for (const auto reason:event.exit_reason) public_step=mix_seed(public_step,uint64_t(reason));
    for (int seat=0;seat<kSeats;++seat) {
      const auto view=observe(game.state,seat);
      const auto witnessed=conversion_marks(public_rollout_seed(view,game.marks[seat],view.quiet_clock),game.marks[seat]);
      views[seat]=mix_seed(mix_seed(views[seat],public_step),witnessed);
    }
  }
};

static ConversionPrefix conversion_prefix(uint64_t seed,const ExchangePolicy& p,bool prepare,ConversionJournal* journal=nullptr) {
  ConversionPrefix prefix{EarlyPrefix(opening(seed,p.rules),seed)};
  auto& game=prefix.game;
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    if (game.enemy_exit && d.observation.viewer==p.learner) break;
    const bool probe=!prefix.early_at && !game.enemy_exit && game.state.ply<140 && d.observation.viewer==p.learner;
    std::vector<std::pair<Move,int>> scores;
    auto action=early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner,probe?&scores:nullptr);
    if (probe) if (const auto offer=choose_approach(game,d,action,p.rules,scores)) {
      if (prepare) action=offer->move;
      prefix.early_at=game.state.ply+1;
      prefix.early_move=early_action_key(action);
    }
    if (journal) journal->play(game,action,p);
    else game.play(action,p.learner,p.rules);
  }
  return prefix;
}

static uint64_t conversion_marks(uint64_t hash,const OpeningMarks& marks) {
  hash=fnv1a(marks.ranks.data(),sizeof(marks.ranks),hash);
  hash=fnv1a(marks.moved.data(),sizeof(marks.moved),hash);
  for (const auto& losses:marks.removed) {
    hash=mix_seed(hash,losses.size);
    hash=fnv1a(losses.masks.data(),losses.size*sizeof(losses.masks[0]),hash);
  }
  return hash;
}

static uint64_t conversion_checkpoint(const EarlyPrefix& game) {
  uint64_t hash=mix_seed(state_hash(game.state),game.trace);
  hash=mix_seed(hash,uint64_t(game.enemy_reason*16+game.friendly_reason));
  hash=conversion_marks(hash,game.reply_history.marks());
  for (int seat=0;seat<kSeats;++seat) {
    const auto view=observe(game.state,seat);
    hash=conversion_marks(public_rollout_seed(view,game.marks[seat],view.quiet_clock),game.marks[seat])^mix_seed(hash,game.streams[seat].s);
    hash=mix_seed(hash,uint64_t(early_action_key(game.reply_history.last_moves()[seat])));
  }
  return hash;
}

static ExchangeChoice conversion_choice(const EarlyPrefix& game,const SeatDecision& d,Move ordinary,const ExchangePolicy& p) {
  const auto& actor=game.marks[p.learner];
  if (const auto finish=public_flag_finish(d,ordinary,p.rules)) return {*finish,1,0};
  const auto evidence=game.reply_history.for_actor(d.observation,actor);
  const auto threat=visible_flag_threat(d.observation,evidence,p.rules);
  if (const auto guard=visible_intercept(d,evidence,p.rules,ordinary,threat)) return {*guard,2,0};
  ResponseWork work;
  const auto move=response_choice(d,actor,game.reply_history,ordinary,p.rules,p.history,d.observation.quiet_clock,
                                  work,false,RoutePriority::Joint,false,false,true);
  return {move,work.samples?3:4,work.samples};
}

static void conversion_tail(const EarlyPrefix& start,const ExchangePolicy& p,bool improve) {
  auto game=start;
  int base=-1, picked=-1, kind=0, samples=0;
  if (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    const auto ordinary=early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner);
    const auto choice=improve?conversion_choice(game,d,ordinary,p):ExchangeChoice{ordinary};
    base=early_action_key(ordinary); picked=early_action_key(choice.action);
    kind=choice.kind; samples=choice.samples;
    game.play(choice.action,p.learner,p.rules);
    if (kind==1) single_case_check(review_outcome(game.state.result,p.learner)==2,"certified conversion did not win");
  }
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    game.play(early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner),p.learner,p.rules);
  }
  std::cout<<' '<<base<<' '<<picked<<' '<<kind<<' '<<samples;
  print_approach_result(game,p.learner);
}

static void print_conversion(uint64_t seed,const ExchangePolicy& p,bool early,const ConversionPrefix& prefix) {
  const auto& game=prefix.game;
  const auto view=observe(game.state,p.learner);
  const bool eligible=game.state.result==Result::Ongoing;
  const int friends=occupied_seats(view).teams[team_of(p.learner)];
  int engineers=0,bombs=0;
  for (int n=0;n<kNodes;++n) if (view.seat_at(n)==p.learner && can_leave_node(view.rank_at(n),n)) {
    engineers+=view.rank_at(n)==Engineer;
    bombs+=view.rank_at(n)==Bomb;
  }
  std::cout<<seed<<' '<<p.learner<<' '<<p.partner<<' '<<p.table<<' '<<early<<' '<<prefix.early_at
           <<' '<<prefix.early_move<<' '<<(eligible?game.state.ply+1:0)<<' '<<conversion_checkpoint(game)
           <<' '<<friends<<' '<<engineers<<' '<<bombs<<' '<<view.quiet_clock<<' '<<unsigned(view.revealed_flags)
           <<' '<<game.enemy_exit<<' '<<game.enemy_reason<<' '<<game.friendly_exit<<' '<<game.friendly_reason;
  for (bool improve:{false,true}) conversion_tail(game,p,improve);
  std::cout<<'\n';
}

static void cross_conversion(uint64_t seed,const ExchangePolicy& p,bool early) {
  print_conversion(seed,p,early,conversion_prefix(seed,p,early));
}
