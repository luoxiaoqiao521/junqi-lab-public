#pragma once

struct DelayFork {
  EarlyPrefix root;
  Move base,alternative;
  int listed=0;
};

static bool delay_win_in_shortlist(const SeatDecision& d,const FinishPolicy& plan,
                                    Move finish,const Rules& rules) {
  for (const auto move:plan.shortlist)
    if (move==finish || (d.observation.rank_at(move.to)==Flag &&
                         public_flag_finish(d,move,rules)==move)) return true;
  return false;
}

static int delay_advance(const Move move,const std::vector<int>& flags) {
  return flag_distance(move.from,flags)-flag_distance(move.to,flags);
}

static std::optional<Move> delay_reroute(const SeatDecision& d,const OpeningMarks& marks,
                                        const FinishPolicy& plan,const PublicThreat& threat) {
  const auto& view=d.observation;
  if (threat.origins || view.quiet_clock<8 || plan.base==plan.pressure ||
      view.seat_at(plan.base.to)>=0 || view.seat_at(plan.pressure.to)>=0 ||
      occupied_seats(view).teams[1-view.team(view.viewer)]!=1) return {};
  const auto flags=flag_targets(view,marks);
  if (flags.empty() || delay_advance(plan.pressure,flags)<1 || delay_advance(plan.base,flags)>0) return {};
  return plan.pressure;
}

static void delay_save(std::optional<DelayFork>& fork,const EarlyPrefix& game,
                         const std::array<Rng,kSeats>& streams,Move base,Move alternative,int listed) {
  fork=DelayFork{game,base,alternative,listed};
  fork->root.streams=streams;
}

static void delay_moves(EarlyPrefix& game,const SeatDecision& d,const FinishPolicy& plan,
                         const std::array<Rng,kSeats>& streams,const ExchangePolicy& fixed,
                         std::array<std::optional<DelayFork>,3>& forks,Move previous,
                         const std::string& key,int start) {
  const auto& view=d.observation;
  const auto& marks=game.marks[fixed.learner];
  const auto win=public_flag_finish(d,plan.base,fixed.rules);
  const auto threat=visible_flag_threat(view,marks,fixed.rules);
  const auto guard=visible_intercept(d,marks,fixed.rules,plan.base,threat);
  const auto route=delay_reroute(d,marks,plan,threat);
  if (!forks[0] && win && !(*win==plan.base))
    delay_save(forks[0],game,streams,plan.base,*win,delay_win_in_shortlist(d,plan,*win,fixed.rules));
  if (!forks[1] && guard)
    delay_save(forks[1],game,streams,plan.base,*guard,0);
  if (!forks[2] && !win && route)
    delay_save(forks[2],game,streams,plan.base,*route,1);
  const bool calm=view.seat_at(plan.base.to)<0;
  const bool reverse=calm && previous.from==plan.base.to && previous.to==plan.base.from;
  std::cout<<"T "<<key<<' '<<game.state.ply-start<<' '<<view.quiet_clock<<' '
           <<early_action_key(plan.base)<<' '<<early_action_key(plan.pressure)<<' '
           <<bool(win)<<' '<<(win && !(*win==plan.base))<<' '<<threat.origins<<' '
           <<bool(guard)<<' '<<bool(route)<<' '<<calm<<' '<<reverse<<' '
           <<occupied_seats(view).teams[1-view.team(view.viewer)]<<'\n';
}

static void delay_branches(const DelayFork& test,const ExchangePolicy& fixed,const std::string& key,
                            int step,int start) {
  std::cout<<"P "<<key<<' '<<step<<' '<<test.root.state.ply-start<<' '
           <<early_action_key(test.base)<<' '<<early_action_key(test.alternative)<<' '<<test.listed<<'\n';
  for (int stream=0;stream<3;++stream) {
    auto phase=test.root;
    if (stream) phase.streams=review_streams(mix_seed(phase.trace,67501+stream));
    for (int arm:{0,1}) {
      std::cout<<"V "<<key<<' '<<step<<' '<<stream<<' '<<arm;
      takeover_game(phase,fixed,arm?test.alternative:test.base,3);
    }
  }
}

static void delay_replay(const FinishReplay& book,TakeoverCut cut,int rep,const ExchangePolicy& frozen,
                          const std::string& key) {
  auto game=book.positions[cut.at];
  const ExchangePolicy fixed{frozen.rules,frozen.parent,frozen.history,game.state.to_move,
                              1+(rep/2)%2,rep%2,frozen.student};
  game.streams=review_streams(mix_seed(game.trace,48211+rep));
  std::array<std::optional<DelayFork>,3> forks;
  ExchangeTime clock;
  int first=0,second=0;
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    const auto streams=game.streams;
    const auto plan=finish_policy(game,d,fixed);
    if (d.observation.viewer==fixed.learner)
      delay_moves(game,d,plan,streams,fixed,forks,game.reply_history.last_moves()[fixed.learner],key,cut.at);
    const auto before=occupied_seats(d.observation);
    exchange_step(game,plan.base,fixed,clock);
    for (int foe=0;foe<kSeats;++foe)
      if (team_of(foe)!=team_of(fixed.learner) && (before.occupied&(1u<<foe)) && game.state.out[foe]) {
        if (!first) first=game.state.ply-cut.at;
        second=game.state.ply-cut.at;
      }
  }
  std::cout<<"E "<<key<<' '<<game.state.ply-cut.at<<' '<<review_outcome(game.state.result,fixed.learner)
           <<' '<<game.trace<<' '<<first<<' '<<second<<' '<<clock.max_quiet<<' '<<clock.reverse<<'\n';
  for (int role=0;role<int(forks.size());++role)
    if (forks[role]) delay_branches(*forks[role],fixed,key,role,cut.at);
}

static void delay_archive(const std::string& key,const ArchiveRows& raw,int kind,int rep,
                           const ExchangePolicy& frozen) {
  auto rules=frozen.rules;
  std::optional<FinishReplay> archive;
  for (bool ccw:{false,true}) {
    rules.ccw=ccw;
    try {
      archive=finish_replay(raw,rules,std::stoull(key.substr(0,16),nullptr,16));
      break;
    } catch (const std::runtime_error&) {
      // Check the other archive direction if the original moves reject this one.
    }
  }
  expect_value(archive.has_value(),"sampled archive no longer strict-replays");
  const auto cuts=finish_cuts(*archive,kind==3);
  auto entry=std::find_if(cuts.begin(),cuts.end(),[&](TakeoverCut cut){return cut.kind==kind;});
  expect_value(entry!=cuts.end(),"sampled cut no longer restores");
  delay_replay(*archive,*entry,rep,{rules,frozen.parent,frozen.history,kNone,1,0,frozen.student},key);
}
