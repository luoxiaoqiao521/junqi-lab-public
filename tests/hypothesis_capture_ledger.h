#pragma once

static void capture_fixture() {
  const auto& board=Board::get();
  auto node=[&](int row,int col){return static_cast<uint8_t>(board.seat_node(2,row,col));};
  const Move hit{node(4,1),node(4,0)}, flying{node(4,0),node(0,4)};
  const Rules rules;
  std::optional<Observation> first_view;
  std::pair<unsigned,unsigned> first_masks{};
  for (bool engineer:{false,true}) {
    auto s=single_case_state();
    s.ply=s.no_progress=0;
    s.to_move=1;
    s.out[2]=false;
    place_single_case_piece(s,1,4,Platoon);
    place_single_case_piece(s,26,hit.from,engineer?Engineer:Company);
    place_single_case_piece(s,50,node(5,1),Flag);
    place_single_case_piece(s,51,hit.to,engineer?Mine:Platoon);
    place_single_case_piece(s,52,node(2,1),Platoon);
    PublicReplyHistory tracked(s,true,true), original(s,true);
    OpeningMarks private_marks(observe(s,0),true);
    auto apply=[&](Move move) {
      const auto before=s;
      const auto event=recorded_move(s,move,rules);
      tracked.advance(before,s,event,rules);
      original.advance(before,s,event,rules);
      advance_contact_ranks(private_marks,observe(before,0),observe(s,0),move,event.outcome,rules);
    };
    const auto turn=[&](int seat) {
      s.to_move=static_cast<int8_t>(seat);
      tracked.after_control(s);
      original.after_control(s);
    };
    apply(hit);
    turn(0);
    std::vector<CaptureEvidence> rows;
    const auto view=observe(s,0);
    auto evidence=tracked.for_actor(view,private_marks,&rows);
    const auto uncertain_loss=collision_losses(view,evidence,rows,1,2,Mine);
    expect_value(uncertain_loss.total==1,"initial collision lost the opposing seat attribution");
    check_stock_range(uncertain_loss.rank,0,1);
    check_stock_range(collision_losses(view,evidence,rows,0,2,Mine).rank,0,0);
    expect_value(rows.size()==1 && rows[0].attacker_seat==1 && rows[0].defender_seat==2 &&
                 rows[0].attacker==hit.from && rows[0].defender==hit.to &&
                 (rows[0].defender_ranks&(1u<<Mine)) && (rows[0].defender_ranks&(1u<<Platoon)),
                 "capture fixture leaked enemy or teammate ranks before evidence");
    const auto sampled=public_sample(view,evidence,23);
    expect_value(sampled && tracked.check_candidate(view,private_marks,sampled->rank_at,64)==
                 CaptureCompatibility::Consistent,"compatible hypothesis was not recognized");
    expect_value(tracked.check_candidate(view,private_marks,sampled->rank_at,0)==
                 CaptureCompatibility::Unresolved,"exhausted proof budget decided a hypothesis");
    auto contradicted=*sampled;
    contradicted.rank_at[hit.to]=Bomb;
    expect_value(tracked.check_candidate(view,private_marks,contradicted.rank_at,64)==
                 CaptureCompatibility::Contradicted,"collision survivor was assigned a bomb");
    if (first_view) {
      expect_value(same_observation(*first_view,view) &&
                   first_masks==std::pair{rows[0].attacker_ranks,rows[0].defender_ranks},
                   "same observation produced different prior collision evidence");
    } else {
      first_view=view;
      first_masks={rows[0].attacker_ranks,rows[0].defender_ranks};
    }
    const auto replayed=tracked;
    const auto pre_move_marks=private_marks;
    if (!engineer) continue;
    turn(1);
    apply(flying);
    turn(0);
    const auto after=observe(s,0);
    evidence=tracked.for_actor(after,private_marks,&rows);
    const auto legacy=original.for_actor(after,private_marks);
    expect_value(rows.size()==1 && rows[0].ply==1 && rows[0].attacker==hit.from &&
      rows[0].attacker_ranks==(1u<<Engineer) && rows[0].defender_ranks==(1u<<Mine) &&
      legacy.removed[2].masks[0]!=(1u<<Mine) && evidence.removed[2].confirmed[Mine]==1,
      "later route did not refine the same recorded loss");
    check_stock_range(rank_stock(after,evidence,2,Mine).collision_lost,1,1);
    check_stock_range(collision_losses(after,evidence,rows,1,2,Mine).rank,1,1);
    replayed.for_actor(view,pre_move_marks,&rows);
    expect_value((rows[0].defender_ranks&(1u<<Platoon))!=0,"a later hypothesis changed a preserved snapshot");
    StepInfo exit;
    recorded_retirement(s,1,exit,ExitReason::NoLegalMove);
    tracked.after_control(s);
    evidence=tracked.for_actor(observe(s,0),private_marks,&rows);
    expect_value(rows.size()==1 && evidence.removed[1].size==0 && evidence.removed[2].size==1,
                 "administrative clearance changed capture history");
    check_stock_range(collision_losses(observe(s,0),evidence,rows,1,2,Mine).rank,1,1);
  }
}

static void capture_attribution_fixture() {
  const auto view=observe(single_case_state(),0);
  OpeningMarks marks(view,false);
  const unsigned lead=(1u<<Marshal)|(1u<<Corps);
  marks.removed[1].add(lead);
  marks.removed[1].add(lead);
  marks.removed[2].add(1u<<Bomb);
  const std::vector<CaptureEvidence> rows{{1,4,30,0,1,Outcome::AttackerWins,1u<<Marshal,lead},
                                         {2,51,31,2,1,Outcome::BothRemoved,1u<<Bomb,lead}};
  check_stock_range(collision_losses(view,marks,rows,0,1,Marshal).rank,0,1);
  check_stock_range(collision_losses(view,marks,rows,2,1,Marshal).rank,0,1);
  check_stock_range(rank_stock(view,marks,1,Marshal).collision_lost,1,1);
  check_stock_range(collision_losses(view,marks,rows,1,2,Bomb).rank,1,1);
  expect_value(collision_losses(view,marks,rows,1,0,Marshal).total==0,
               "surviving attacker was charged as a casualty");
  marks.removed[1]=RemovedRanks{};
  marks.removed[1].add(1u<<Corps);
  marks.removed[1].add(lead);
  check_stock_range(collision_losses(view,marks,rows,2,1,Marshal).rank,1,1);
}

struct CaptureArchiveCounts {
  int roots=0, records=0, tightened=0, decisive_losses=0, stock_checks=0, controls=0, contacts_changed=0;
  int samples=0, compatible=0, contradictions=0, unresolved=0, truth_unresolved=0;
};

static void check_capture_worlds(const State& state,const Observation& view,const OpeningMarks& actor,
    const OpeningMarks& evidence,const PublicReplyHistory& history,CaptureArchiveCounts& count) {
  using Result=CaptureCompatibility;
  constexpr unsigned budget=96;
  // Actual live identities serve only as a positive audit control.
  const auto truth=history.check_candidate(view,actor,state.rank_at,budget);
  expect_value(truth!=Result::Contradicted,"joint constraints excluded the replay's real living pieces");
  count.truth_unresolved+=truth==Result::Unresolved;
  for (uint64_t sample=0;sample<4;++sample) {
    auto world=public_sample(view,evidence,mix_seed(static_cast<uint64_t>(view.ply),sample));
    if (!world) continue;
    ++count.samples;
    const auto result=history.check_candidate(view,actor,world->rank_at,budget);
    count.compatible+=result==Result::Consistent;
    count.contradictions+=result==Result::Contradicted;
    count.unresolved+=result==Result::Unresolved;
  }
}

static void check_capture_root(const State& state,const OpeningMarks& actor,const PublicReplyHistory& baseline,
    const PublicReplyHistory& tracker,const std::array<std::vector<int>,kSeats>& deaths,
    const Rules& rules,CaptureArchiveCounts& count) {
  const auto view=observe(state,state.to_move);
  std::vector<CaptureEvidence> records;
  const auto before=baseline.for_actor(view,actor);
  const auto after=tracker.for_actor(view,actor,&records);
  check_capture_worlds(state,view,actor,after,tracker,count);
  ++count.roots;
  count.records+=static_cast<int>(records.size());
  for (int node=0;node<kNodes;++node) if (state.seat_at[node]>=0) {
    const auto mask=compatible_rank_mask(view,after,node);
    expect_value((mask&(1u<<state.rank_at[node]))!=0,"cross-collision evidence discarded the original living identity");
    expect_value((mask&~compatible_rank_mask(view,before,node))==0,"cross-collision evidence expanded a living identity");
    count.tightened+=mask!=compatible_rank_mask(view,before,node);
  }
  const auto decision=seat_decision(state);
  for (const auto move:decision.actions) if (view.seat_at(move.to)>=0) {
    const int attacker=view.rank_at(move.from);
    const auto old=collision_outcome_mask(attacker,compatible_rank_mask(view,before,move.to),rules);
    const auto narrowed=collision_outcome_mask(attacker,compatible_rank_mask(view,after,move.to),rules);
    count.contacts_changed+=old!=narrowed;
  }
  for (int seat=0;seat<kSeats;++seat) {
    expect_value(after.removed[seat].size==deaths[seat].size(),"clearance counted as contact loss");
    for (size_t d=0;d<deaths[seat].size();++d) {
      const auto mask=after.removed[seat].masks[d], old=before.removed[seat].masks[d];
      expect_value((mask&(1u<<deaths[seat][d])) && !(mask&~old),"invalid past casualty refinement");
      count.tightened+=mask!=old;
      count.decisive_losses+=(mask!=old && (mask&(mask-1))==0);
    }
    const int rank=(view.ply+seat)%RankCount;
    const auto counts=rank_stock(view,after,seat,rank);
    int living=0;
    for (int n=0;n<kNodes;++n) living+=state.seat_at[n]==seat && state.rank_at[n]==rank;
    const int lost=static_cast<int>(std::count(deaths[seat].begin(),deaths[seat].end(),rank));
    expect_value(counts.on_board.minimum<=living && living<=counts.on_board.maximum &&
      counts.collision_lost.minimum<=lost && lost<=counts.collision_lost.maximum,"archived stock is outside the observer interval");
    ++count.stock_checks;
  }
}

static void capture_archive(const ArchiveRows& raw,const Rules& rules) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  PublicReplyHistory baseline(review.state,true), tracked(review.state,true,true);
  std::vector<OpeningMarks> private_marks;
  for (int seat=0;seat<kSeats;++seat) private_marks.emplace_back(observe(review.state,seat),true);
  std::array<std::vector<int>,kSeats> lost;
  CaptureArchiveCounts counts;
  for (int i=0;i<raw.entries;++i) {
    const auto row=raw.row(i);
    if (raw.byte(row)==245) {
      review.control(raw,row,i+1,rules);
      baseline.after_control(review.state);
      tracked.after_control(review.state);
      ++counts.controls;
      continue;
    }
    const auto before=review.state;
    if (before.ply%24==0) check_capture_root(before,private_marks[before.to_move],baseline,tracked,lost,rules,counts);
    review.play(raw,row,i+1,rules);
    review.frames.str("");
    const auto& move=review.last.move;
    const auto outcome=review.last.outcome;
    if (outcome!=Outcome::Move && outcome!=Outcome::AttackerWins)
      lost[before.to_move].push_back(before.rank_at[move.from]);
    if (outcome!=Outcome::Move && outcome!=Outcome::DefenderWins)
      lost[before.seat_at[move.to]].push_back(before.rank_at[move.to]);
    tracked.advance(before,review.state,review.last,rules);
    baseline.advance(before,review.state,review.last,rules);
    for (int seat=0;seat<kSeats;++seat) if (!review.state.out[seat])
      advance_contact_ranks(private_marks[seat],observe(before,seat),observe(review.state,seat),move,outcome,rules);
  }
  std::cout<<review.state.ply<<' '<<counts.roots<<' '<<counts.records<<' '<<counts.tightened<<' '
           <<counts.decisive_losses<<' '<<counts.stock_checks<<' '<<counts.controls<<' '<<counts.contacts_changed<<' '
           <<counts.samples<<' '<<counts.compatible<<' '<<counts.contradictions<<' '<<counts.unresolved<<' '
           <<counts.truth_unresolved<<'\n';
}

static std::pair<unsigned,unsigned> course_contact_support(const PublicReplyHistory& history,
    const Observation& view,const OpeningMarks& actor,const OpeningMarks& evidence,Move attack,const Rules& rules) {
  int8_t pins[kNodes];
  std::fill(std::begin(pins),std::end(pins),Unknown);
  const auto pool=compatible_rank_mask(view,evidence,attack.to);
  unsigned feasible=0, unresolved=0;
  for (int rank=Flag;rank<RankCount;++rank) if (pool&(1u<<rank)) {
    pins[attack.to]=static_cast<int8_t>(rank);
    const auto result=history.check_candidate(view,actor,pins,384);
    const unsigned outcome=1u<<static_cast<unsigned>(resolve(view.rank_at(attack.from),rank,rules));
    if (result==CaptureCompatibility::Consistent) feasible|=outcome;
    if (result==CaptureCompatibility::Unresolved) unresolved|=outcome;
  }
  return {feasible,unresolved&~feasible};
}

static void capture_course() {
  int8_t layouts[4][25]{};
  std::vector<std::array<int,4>> rows;
  expect_value(read_qh(layouts,rows) && rows.size()==172,"missing course prefix");
  Rules rules;
  rules.ccw=false;
  rules.no_progress_limit=0;
  State state;
  init_state(state,layouts,0,rules);
  PublicReplyHistory baseline(state,true), history(state,true,true);
  std::vector<OpeningMarks> actor;
  for (int seat=0;seat<kSeats;++seat) actor.emplace_back(observe(state,seat),true);
  const auto action=[](const auto& row) {
    const int from=point(row[0],row[1]),to=point(row[2],row[3]);
    expect_value(from>=0 && to>=0,"unsupported course action");
    return Move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
  };
  for (size_t i=0;i+1<rows.size();++i) {
    const auto move=action(rows[i]);
    expect_value(is_legal(state,move),"course prefix diverged");
    const auto before=state;
    const auto event=step(state,move,rules);
    baseline.advance(before,state,event,rules);
    history.advance(before,state,event,rules);
    for (int seat=0;seat<kSeats;++seat) if (!state.out[seat])
      advance_contact_ranks(actor[seat],observe(before,seat),observe(state,seat),move,event.outcome,rules);
  }
  const auto decision=seat_decision(state);
  const auto& view=decision.observation;
  const auto& private_marks=actor[view.viewer];
  const auto attack=action(rows.back());
  expect_value(is_legal(state,attack) && state.rank_at[attack.from]==Battalion &&
               state.rank_at[attack.to]==Division,"course cut is not the cited collision");
  const auto prior=baseline.for_actor(view,private_marks);
  const auto joined=history.for_actor(view,private_marks);
  const auto [supported,pending]=course_contact_support(history,view,private_marks,joined,attack,rules);
  const auto options=[&](const OpeningMarks& marks) {
    return collision_outcome_mask(view.rank_at(attack.from),compatible_rank_mask(view,marks,attack.to),rules);
  };
  CaptureArchiveCounts count;
  check_capture_worlds(state,view,private_marks,joined,history,count);
  int quiet=0,contacts=0,changed=0;
  for (const auto move:decision.actions) {
    if (view.seat_at(move.to)<0) { ++quiet; continue; }
    ++contacts;
    changed+=collision_outcome_mask(view.rank_at(move.from),compatible_rank_mask(view,prior,move.to),rules)!=
              collision_outcome_mask(view.rank_at(move.from),compatible_rank_mask(view,joined,move.to),rules);
  }
  std::cout<<state.ply+1<<' '<<quiet<<' '<<contacts<<' '<<options(prior)<<' '<<options(joined)<<' '
           <<changed<<' '<<count.samples<<' '<<count.compatible<<' '<<count.contradictions<<' '
           <<count.unresolved<<' '<<count.truth_unresolved<<' '<<supported<<' '<<pending<<'\n';
}
