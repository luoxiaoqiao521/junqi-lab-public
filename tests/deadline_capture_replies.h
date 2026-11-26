#pragma once

struct CaptureReplyCut {
  std::string key;
  int ply=0, target=-1;
  uint64_t world_seed=0, reply_seed=0;
  const StudentNetwork* student=nullptr;
  ShortWeights parent{};
};

static void audit_casualty_credit(const Observation& view,const OpeningMarks& marks,
    const std::vector<CaptureEvidence>& rows,const int8_t (&initial)[kNodes],const std::string* key) {
  int losses[kSeats][kSeats][RankCount]{};
  for (const auto& row:rows) {
    if (row.outcome==Outcome::BothRemoved || row.outcome==Outcome::DefenderWins)
      ++losses[row.defender_seat][row.attacker_seat][initial[row.attacker]];
    if (row.outcome==Outcome::BothRemoved || row.outcome==Outcome::AttackerWins)
      ++losses[row.attacker_seat][row.defender_seat][initial[row.defender]];
  }
  for (int other=0;other<kSeats;++other) for (int victim=0;victim<kSeats;++victim) if (other!=victim) {
    size_t total=0;
    for (int rank=Flag;rank<RankCount;++rank) total+=losses[other][victim][rank];
    for (int rank=Flag;rank<RankCount;++rank) {
      const auto actual=losses[other][victim][rank];
      const auto check=collision_losses(view,marks,rows,other,victim,rank);
      single_case_check(check.total==total && check.rank.minimum<=actual && actual<=check.rank.maximum,
                        "attributed range excluded a replay casualty");
      if (key && total) std::cout<<"L "<<*key<<' '<<other<<' '<<victim<<' '<<rank<<' '<<total<<' '
                                <<check.rank.minimum<<' '<<check.rank.maximum<<'\n';
    }
  }
}

static Move capture_student_anchor(const EarlyPrefix& game,const SeatDecision& decision,Move archived,
    const OpeningMarks& evidence,const std::vector<CaptureEvidence>& rows,const Rules& rules,
    const ShortWeights& peer,const CaptureReplyCut& cut) {
  const auto& view=decision.observation;
  const auto& actor=game.marks[view.viewer];
  auto evaluation=game;
  evaluation.streams=review_streams(public_rollout_seed(view,actor,view.quiet_clock));
  const auto choice=finish_policy(evaluation,decision,ExchangePolicy{rules,cut.parent,peer,view.viewer,1,1,cut.student});
  const auto revised=student_pick(decision,evidence,choice.pressure,choice.shortlist,*cut.student,rules);
  int feature_changes=0, collision_changes=0;
  for (const auto move:decision.actions) {
    const auto base=student_features(decision,actor,move,&rules);
    const auto refined=student_features(decision,evidence,move,&rules);
    feature_changes+=base!=refined;
    collision_changes+=!std::equal(base.begin()+student_inputs,base.end(),refined.begin()+student_inputs);
  }
  std::cout<<"S "<<cut.key<<' '<<view.viewer<<' '<<early_action_key(archived)<<' '<<early_action_key(choice.base)
           <<' '<<early_action_key(revised)<<' '<<early_action_key(choice.pressure)<<' '<<feature_changes
           <<' '<<collision_changes;
  for (const auto move:choice.shortlist) std::cout<<' '<<early_action_key(move);
  std::cout<<'\n';
  for (const auto& event:rows)
    std::cout<<"E "<<cut.key<<' '<<event.ply<<' '<<int(event.attacker)<<' '<<int(event.defender)<<' '
             <<int(event.attacker_seat)<<' '<<int(event.defender_seat)<<' '<<int(event.outcome)<<' '
             <<event.attacker_ranks<<' '<<event.defender_ranks<<'\n';
  return choice.base;
}

static void capture_reply_root(const EarlyPrefix& game,Move original,const Rules& rules,
                               const ShortWeights& peer,const CaptureReplyCut& cut,const int8_t (&initial)[kNodes]) {
  const auto decision=seat_decision(game.state);
  const auto& view=decision.observation;
  const auto& actor=game.marks[view.viewer];
  std::vector<CaptureEvidence> rows;
  const auto evidence=game.reply_history.for_actor(view,actor,&rows);
  if (cut.student) original=capture_student_anchor(game,decision,original,evidence,rows,rules,peer,cut);
  std::vector<State> worlds;
  int rejects=0,unresolved=0;
  const auto public_seed=public_rollout_seed(view,evidence,view.quiet_clock);
  const auto world_seed=mix_seed(cut.world_seed,public_seed);
  const auto reply_seed=mix_seed(cut.reply_seed,public_seed);
  for (int attempt=0;attempt<48 && worlds.size()<4;++attempt) {
    const auto sample=public_sample(view,evidence,mix_seed(world_seed,attempt));
    if (!sample) continue;
    const auto result=game.reply_history.check_candidate(view,actor,sample->rank_at,384);
    rejects+=result==CaptureCompatibility::Contradicted;
    unresolved+=result==CaptureCompatibility::Unresolved;
    if (result==CaptureCompatibility::Consistent && std::none_of(worlds.begin(),worlds.end(),[&](const State& other) {
          return std::equal(std::begin(other.rank_at),std::end(other.rank_at),std::begin(sample->rank_at));
        })) worlds.push_back(*sample);
  }
  std::vector<Move> moves;
  for (const auto move:decision.actions)
    if (cut.target<0 || move==original || early_action_key(move)==cut.target) moves.push_back(move);
  single_case_check(std::find(moves.begin(),moves.end(),original)!=moves.end() &&
                    (cut.target<0 || moves.size()==2),"capture reply requires legal locked actions");
  const int target_seat=view.seat_at(original.to);
  AttributedLosses bombs;
  if (target_seat>=0) bombs=collision_losses(view,evidence,rows,view.viewer,target_seat,Bomb);
  std::cout<<"R "<<cut.key<<' '<<cut.ply<<' '<<early_action_key(original)<<' '<<decision.actions.size()
           <<' '<<worlds.size()<<' '<<rejects<<' '<<unresolved<<' '<<bombs.total<<' '<<bombs.rank.minimum
           <<' '<<bombs.rank.maximum<<'\n';
  for (auto move:moves) for (size_t i=0;i<worlds.size();++i) for (int desk:{0,1}) {
    ResponseAudit work{140};
    ResponseValue value;
    response_trial(worlds[i],move,evidence,game.reply_history,rules,peer,desk,
                   mix_seed(reply_seed,i+uint64_t(desk)*1103),value,false,RoutePriority::Old,true,&work);
    std::cout<<"A "<<cut.key<<' '<<early_action_key(move)<<' '<<i<<' '<<desk<<' '
             <<(value.wins?2:value.draws?1:value.losses?0:-1)<<' '<<work.plies<<'\n';
  }
  audit_casualty_credit(view,evidence,rows,initial,cut.student?&cut.key:nullptr);
}

static void advance_capture_reply(EarlyPrefix& game,Move move,const Rules& rules) {
  single_case_check(is_legal(game.state,move),"illegal capture reply prefix action");
  game.play(move,game.state.to_move,rules);
}

static void qh_capture_reply(const CaptureReplyCut& cut,const ShortWeights& peer) {
  int8_t layouts[kSeats][25]{};
  std::vector<std::array<int,4>> rows;
  single_case_check(read_qh(layouts,rows) && static_cast<int>(rows.size())==cut.ply,"bad capture course input");
  Rules rules;
  rules.ccw=false;
  rules.no_progress_limit=0;
  State start;
  init_state(start,layouts,0,rules);
  EarlyPrefix game(start,cut.world_seed);
  game.reply_history=PublicReplyHistory(start,true,true);
  for (const auto& row:rows) {
    const int from=point(row[0],row[1]),to=point(row[2],row[3]);
    single_case_check(from>=0 && to>=0,"invalid course move");
    const Move move{uint8_t(from),uint8_t(to)};
    if (game.state.ply+1==cut.ply) capture_reply_root(game,move,rules,peer,cut,start.rank_at);
    else advance_capture_reply(game,move,rules);
  }
}

static void qq_capture_reply(const CaptureReplyCut& cut,const ShortWeights& peer,const ArchiveRows& raw,
                             const Rules& rules) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  const auto start=review.state;
  EarlyPrefix game(start,cut.world_seed);
  game.reply_history=PublicReplyHistory(start,true,true);
  for (int i=0;i<raw.entries;) {
    const auto row=raw.row(i++);
    const auto before=review.state;
    review.play(raw,row,i,rules);
    if (before.ply+1==cut.ply) {
      capture_reply_root(game,review.last.move,rules,peer,cut,start.rank_at);
      return;
    }
    archive_observe(game,review,before,rules);
    while (i<raw.entries && raw.byte(raw.row(i))==245) {
      const auto control=raw.row(i++);
      review.control(raw,control,i,rules);
      game.reply_history.after_control(review.state);
      game.state=review.state;
    }
    review.frames.str("");
  }
  single_case_check(false,"missing QQ capture reply cut");
}

static void run_capture_replies(bool use_student) {
  ShortWeights peer{};
  for (size_t i=0;i<5;++i)
    single_case_check(bool(std::cin>>peer[i]) && std::isfinite(peer[i]),"invalid reply policy");
  CaptureReplyCut cut;
  StudentNetwork student;
  if (use_student) {
    for (size_t i=0;i<5;++i)
      single_case_check(bool(std::cin>>cut.parent[i]) && std::isfinite(cut.parent[i]),"invalid student fallback policy");
    single_case_check(student.load(std::cin,true),"invalid 44-input student encoding");
    cut.student=&student;
  }
  char kind;
  int ccw;
  while (std::cin>>kind>>cut.key>>cut.ply>>cut.target>>cut.world_seed>>cut.reply_seed>>ccw) {
    if (kind=='H') qh_capture_reply(cut,peer);
    else {
      std::string error,hex;
      Rules rules;
      single_case_check(kind=='G' && (ccw==0 || ccw==1) && bool(std::cin>>hex) &&
          parse_rules(archive_review_rules(read_text("config/ruleset-selfplay.json"),bool(ccw)),rules,error),
          "invalid archive reply input");
      qq_capture_reply(cut,peer,ArchiveRows(decode_archive_tape(hex)),rules);
    }
  }
  single_case_check(std::cin.eof(),"incomplete capture reply payload");
}
