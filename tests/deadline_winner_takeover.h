#pragma once

struct FinishReplay {
  std::vector<EarlyPrefix> positions;
  std::vector<Move> moves;
  std::vector<unsigned> flags;
  int winner=kNone;
};

static FinishReplay finish_replay(const ArchiveRows& raw,const Rules& rules,uint64_t seed) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  EarlyPrefix game(review.state,seed);
  FinishReplay result;
  for (int i=0;i<raw.entries;) {
    const auto row=raw.row(i++);
    expect_value(raw.byte(row)==95,"non-move before archive snapshot");
    const auto before=review.state;
    result.positions.push_back(game);
    review.play(raw,row,i,rules);
    review.frames.str("");
    unsigned flags=0;
    while (i<raw.entries && raw.byte(raw.row(i))==245) {
      const auto notice=raw.row(i++);
      if (raw.byte(notice+1)==3 && raw.byte(notice+3)==1)
        flags|=1u<<raw.seat.at(raw.byte(notice+2));
      review.control(raw,notice,i,rules);
    }
    result.flags.push_back(flags);
    result.moves.push_back(review.last.move);
    archive_observe(game,review,before,rules);
  }
  expect_value(review.ended && review.captured_flag==kNone &&
       (review.state.result==Result::Team0Wins || review.state.result==Result::Team1Wins),
       "original winner or flag clearance unresolved");
  result.winner=review.state.result==Result::Team0Wins?0:1;
  return result;
}

struct TakeoverCut { int kind,at; };

static int previous_seat_turn(const FinishReplay& game,int from,int seat,int n) {
  for (int i=from;i>=0;--i) if (game.positions[i].state.to_move==seat && --n==0) return i;
  return kNone;
}

static std::vector<TakeoverCut> finish_cuts(const FinishReplay& game,bool comeback) {
  const int total=int(game.moves.size());
  const int team=comeback?1-game.winner:game.winner;
  std::vector<int> captures;
  for (int at=0;at<total;++at)
    if (game.flags[at]&(team==0?10u:5u)) captures.push_back(at);
  if (captures.empty() || (!comeback && (captures.size()!=2 || captures.back()!=total-1))) return {};
  int after=captures.front()+1;
  if (comeback)
    while (after<total && team_of(game.positions[after].state.to_move)!=team) ++after;
  if (after==total || team_of(game.positions[after].state.to_move)!=team) return {};
  if (comeback) return {{3,after}};
  const int early=previous_seat_turn(game,captures.front(),game.positions[after].state.to_move,2);
  const int late=previous_seat_turn(game,captures.back()-1,game.positions[captures.back()].state.to_move,1);
  if (early<0 || late<after) return {};
  return {{0,early},{1,after},{2,late}};
}

struct FinishPolicy {
  Move base{},pressure{};
  std::vector<Move> shortlist;
  bool public_win=false;
};

static FinishPolicy finish_policy(EarlyPrefix& game,const SeatDecision& d,const ExchangePolicy& table,
                                  bool public_first=false) {
  if (public_first && d.observation.viewer==table.learner)
    if (const auto win=public_flag_finish(d,d.actions.front(),table.rules)) return {*win,*win,{},true};
  const auto pressure=early_normal(game,d,table.learner,table.table,table.rules,table.parent,table.history,
                                   table.partner);
  if (d.observation.viewer!=table.learner) return {pressure,pressure,{}};
  const auto& marks=game.marks[table.learner];
  auto actions=response_candidates(d,marks,pressure,table.rules,public_rollout_seed(d.observation,marks,
                                    d.observation.quiet_clock));
  const auto choice=student_pick(d,marks,pressure,actions,*table.student,table.rules);
  return {choice,pressure,std::move(actions)};
}

struct FinishTelemetry {
  int corrections=0,guards=0,misses=0,known_win=0,first=kNone,late_flag_seen=0;
};

constexpr int proof_only=4, guard_only=5;

static Move finish_action(const SeatDecision& d,const OpeningMarks& marks,const FinishPolicy& choices,
                          const Rules& rules,int arm,FinishTelemetry& info) {
  if (choices.public_win) {
    ++info.known_win;
    info.late_flag_seen=d.observation.ply+1;
    return choices.base;
  }
  if (arm==proof_only) return choices.base;
  const auto terminal=public_flag_finish(d,choices.base,rules);
  if (terminal) {
    ++info.known_win;
    info.misses+=!(*terminal==choices.base);
    if (info.late_flag_seen==0) info.late_flag_seen=d.observation.ply+1;
    if (arm==1) {info.corrections+=!(*terminal==choices.base); return *terminal;}
  }
  if (arm==1 || arm==guard_only) {
    const auto threat=visible_flag_threat(d.observation,marks,rules);
    if (const auto guard=visible_intercept(d,marks,rules,choices.base,threat)) {
      ++info.guards;
      return *guard;
    }
  }
  return arm==2?choices.pressure:choices.base;
}

// Copy the fork inside this function to avoid an MSVC /O2 failure in the caller.
#ifdef _MSC_VER
__declspec(noinline)
#endif
static void takeover_game(const EarlyPrefix& root,const ExchangePolicy& table,Move original,int arm) {
  auto game=root;
  const int start=game.state.ply;
  int earliest_enemy=0, latest_enemy=0;
  ExchangeTime clock;
  FinishTelemetry info;
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    const int seat=d.observation.viewer;
    const auto policy=finish_policy(game,d,table,arm==proof_only);
    auto choice=policy.base;
    if (seat==table.learner) {
      const auto& marks=game.marks[seat];
      choice=finish_action(d,marks,policy,table.rules,arm,info);
      if (game.state.ply==start && arm==3) choice=original;
    }
    if (game.state.ply==start) info.first=early_action_key(choice);
    const auto before=occupied_seats(d.observation);
    exchange_step(game,choice,table,clock);
    if (policy.public_win)
      single_case_check(review_outcome(game.state.result,table.learner)==2,"public proof did not win at once");
    for (int foe=0;foe<kSeats;++foe)
      if (team_of(foe)!=team_of(table.learner) && (before.occupied&(1u<<foe)) && game.state.out[foe]) {
        if (!earliest_enemy) earliest_enemy=game.state.ply-start;
        latest_enemy=game.state.ply-start;
      }
  }
  std::cout<<' '<<game.state.ply-start<<' '<<review_outcome(game.state.result,table.learner)
           <<' '<<int(game.ending)<<' '<<game.trace<<' '<<info.first<<' '<<earliest_enemy<<' '<<latest_enemy
           <<' '<<info.corrections<<' '<<info.guards<<' '<<info.misses<<' '<<info.known_win
           <<' '<<(info.late_flag_seen?info.late_flag_seen-start:0)<<' '<<clock.max_quiet
           <<' '<<(game.friendly_exit?game.friendly_exit-start:0)<<' '<<clock.reverse<<'\n';
}

static void finish_at(const std::string& key,const FinishReplay& game,TakeoverCut cut,int reps,
                      const ExchangePolicy& frozen,bool confirm) {
  const auto& root=game.positions[cut.at];
  const auto d=seat_decision(root.state);
  auto copy=root;
  auto table=frozen;
  table.learner=d.observation.viewer;
  const auto choice=finish_policy(copy,d,table);
  const auto human=game.moves[cut.at];
  const auto& view=d.observation;
  const auto& marks=root.marks[table.learner];
  const double recorded_score=frozen.student->score(student_features(d,marks,human));
  const auto short_rank=1+std::count_if(choice.shortlist.begin(),choice.shortlist.end(),[&](Move m) {
    return frozen.student->score(student_features(d,marks,m))>recorded_score;
  });
  const auto legal=std::find(d.actions.begin(),d.actions.end(),human)!=d.actions.end();
  expect_value(legal,"recorded human opener missing from restored legal actions");
  const auto offered=std::find(choice.shortlist.begin(),choice.shortlist.end(),human)!=choice.shortlist.end();
  const auto flag_nodes=std::count_if(d.actions.begin(),d.actions.end(),[&](Move m){return view.rank_at(m.to)==Flag;});
  const auto proof=public_flag_finish(d,choice.base,frozen.rules);
  const auto threat=visible_flag_threat(view,marks,frozen.rules);
  std::cout<<"R "<<key<<' '<<cut.kind<<' '<<cut.at<<' '<<game.moves.size()-cut.at<<' '<<view.viewer
           <<' '<<early_action_key(human)<<' '<<d.actions.size()<<' '<<offered<<' '<<(offered?short_rank:0)
           <<' '<<flag_nodes<<' '<<bool(proof)<<' '<<threat.origins<<' '<<threat.forced
           <<' '<<occupied_seats(view).teams[1-team_of(view.viewer)]<<' '<<root.state.no_progress
           <<' '<<root.trace<<'\n';
  for (int rep=0;rep<reps;++rep) {
    auto phase=root;
    phase.streams=review_streams(mix_seed(root.trace,48211+rep));
    table.partner=confirm && rep>=2?3:1+(rep/2)%2;
    table.table=rep%2;
    const auto arms=confirm?std::vector<int>{0,proof_only,guard_only}:std::vector<int>{0,1,2,3};
    for (int arm:arms) {
      std::cout<<"F "<<key<<' '<<cut.kind<<' '<<rep<<' '<<arm;
      takeover_game(phase,table,human,arm);
    }
  }
}

static void takeover_archive(const std::string& key,const ArchiveRows& raw,int mode,int reps,
                             const ExchangePolicy& frozen,bool confirm=false) {
  const auto seed=std::stoull(key.substr(0,16),nullptr,16);
  std::vector<TakeoverCut> cuts;
  std::optional<FinishReplay> game;
  auto rules=frozen.rules;
  try {
    for (bool ccw:{false,true}) {
      rules.ccw=ccw;
      try {
        auto replay=finish_replay(raw,rules,seed);
        cuts=finish_cuts(replay,mode==1);
        game=std::move(replay);
        break;
      } catch (const std::runtime_error&) {
        // Validate the alternate direction without altering the original tape.
      }
    }
  } catch (const std::runtime_error&) {
    game.reset();
  }
  if (!game || cuts.empty()) {std::cout<<"X "<<key<<' '<<bool(game)<<'\n'; return;}
  const ExchangePolicy table{rules,frozen.parent,frozen.history,kNone,1,0,frozen.student};
  for (const auto cut:cuts) finish_at(key,*game,cut,reps,table,confirm);
}
