#pragma once

static EvaluatorBounds full_route_proof(const State& state,Move move,const Rules& rules,int depth) {
  single_case_check(is_legal(state,move),"full-mode proof anchor is illegal");
  auto branch=state;
  step(branch,move,rules);
  int nodes=12000;
  return bounded_team_value(branch,rules,state.to_move,depth-1,nodes);
}

static int full_mode_bound_cases(const Rules& rules) {
  auto root=selfplay_fixture();
  place_single_case_piece(root,1,46,Marshal);
  place_single_case_piece(root,2,52,Engineer);
  place_single_case_piece(root,26,51,Platoon);
  place_single_case_piece(root,27,30,Engineer);
  const Move marshal{46,51}, engineer{52,51};
  int alternatives=0;
  for (int blocked:{0,1}) {
    if (blocked) root.rank_at[51]=root.rank_of[26]=Mine;
    const auto dark=compare_visibility(root,IdentityVisibility::OwnAndPublic);
    const auto full=compare_visibility(root,IdentityVisibility::AllIdentities);
    const auto dual=compare_visibility(root,IdentityVisibility::AllyAndPublic);
    single_case_check(dark.actions==full.actions && dual.actions==full.actions,"mode changed legal moves at the same root");
    const auto first=full_route_proof(root,marshal,rules,3), second=full_route_proof(root,engineer,rules,3);
    alternatives+=blocked ? second.lower==2 && first.lower<2 : first.lower==2;
    Move baseline{};
    Rng rng(42);
    const DecisionScorer scorer(full.observation,rules,DecisionKind::FlagRoute);
    single_case_check(scorer.choose(rng,baseline),"full-mode practice did not choose a move");
    const auto practice=full_route_proof(root,baseline,rules,3);
    std::cout<<"F routes "<<blocked<<' '<<full.actions.size()<<' '<<first.lower<<' '<<first.upper
             <<' '<<second.lower<<' '<<second.upper<<' '<<early_action_key(baseline)<<' '
             <<practice.lower<<' '<<practice.upper<<'\n';
  }
  auto bait=selfplay_fixture();
  bait.no_progress=rules.no_progress_limit-1;
  place_single_case_piece(bait,1,4,Brigade);
  place_single_case_piece(bait,26,30,Engineer);
  place_single_case_piece(bait,27,21,Engineer);
  const auto take=full_route_proof(bait,{4,30},rules,2), hold=full_route_proof(bait,{4,3},rules,2);
  single_case_check(take.lower==0 && take.upper==0 && hold.lower==1 && hold.upper==1,
                    "bait/hold control no longer settles at the expected score");
  std::cout<<"F bait 0 2 "<<take.lower<<' '<<take.upper<<' '<<hold.lower<<' '<<hold.upper<<'\n';
  return alternatives;
}

struct RuleComparison {
  int agrees=0, divergence=0;
  const char* reason="matched";
};

static const char* replay_difference(const State& local,const State& archived,const StepInfo& event,const StepInfo& source) {
  if (event.outcome!=source.outcome) return "collision";
  if (event.revealed_mask!=source.revealed_mask) return "flag_reveal";
  if (!std::equal(std::begin(local.out),std::end(local.out),std::begin(archived.out)) ||
      !std::equal(std::begin(local.seat_at),std::end(local.seat_at),std::begin(archived.seat_at))) return "retirement";
  if (local.result!=archived.result) return "terminal";
  if (local.to_move!=archived.to_move && local.result==Result::Ongoing) return "next_seat";
  if (local.no_progress!=archived.no_progress) return "quiet_clock";
  return nullptr;
}

static RuleComparison compare_archived_rules(const ArchiveRows& raw,const Rules& archived,const Rules& local,int cap) {
  ArchiveReview reference;
  init_state(reference.state,raw.layouts,archive_first_seat(raw),archived);
  auto shadow=reference.state;
  for (int i=0;reference.state.ply<cap && i<raw.entries;) {
    const auto row=raw.row(i++);
    reference.play(raw,row,i,archived);
    while (i<raw.entries && raw.byte(raw.row(i))==245) {
      const auto notice=raw.row(i++);
      reference.control(raw,notice,i,archived);
    }
    const auto ply=reference.state.ply;
    if (!is_legal(shadow,reference.last.move)) return {ply-1,ply,"illegal"};
    const auto result=step(shadow,reference.last.move,local);
    if (const auto why=replay_difference(shadow,reference.state,result,reference.last)) return {ply-1,ply,why};
    if (reference.ended) break;
    reference.frames.str("");
  }
  return {reference.state.ply,0,"matched"};
}

static void run_full_mode_baseline(const char* rules_path,const Rules& rules) {
  const auto alternatives=full_mode_bound_cases(rules);
  std::cout<<"X route_controls "<<alternatives<<'\n';
  std::string key,hex,error;
  int ccw,cap;
  while (std::cin>>key>>ccw>>cap>>hex) {
    Rules archive;
    single_case_check((ccw==0 || ccw==1) && cap>0 && cap<=240 &&
        parse_rules(archive_review_rules(read_text(rules_path),bool(ccw)),archive,error),"invalid full-mode archive input");
    const ArchiveRows raw(decode_archive_tape(hex));
    auto aligned=rules;
    for (int arm:{0,1,2}) {
      if (arm==1) aligned.ccw=archive.ccw;
      if (arm==2) aligned.marshal_reveals_flag=archive.marshal_reveals_flag;
      const auto diff=compare_archived_rules(raw,archive,aligned,cap);
      std::cout<<"J "<<key<<' '<<arm<<' '<<diff.agrees<<' '<<diff.divergence<<' '<<diff.reason<<'\n';
    }
  }
  single_case_check(std::cin.eof(),"incomplete full-mode baseline input");
}
