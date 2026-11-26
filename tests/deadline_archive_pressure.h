#pragma once

struct ArchivePressure {
  std::string key;
  bool found[2][2]{};

  void inspect(const State& state,Move human,const Rules& rules) {
    const int seat=state.to_move, ally=(seat+2)%kSeats;
    const bool unassisted=state.out[ally];
    if (state.ply<36 || (found[0][unassisted] && found[1][unassisted])) return;
    const auto& board=Board::get();
    int flag=kNone;
    for (int c:{1,3}) {
      const int point=board.seat_node(seat,5,c);
      if (state.seat_at[point]==seat && state.rank_at[point]==Flag) flag=point;
    }
    if (flag==kNone) return;
    int threats=0;
    for (int enemy:{(seat+1)%kSeats,(seat+3)%kSeats}) {
      if (state.out[enemy]) continue;
      std::vector<Move> attacks;
      gen_moves(StateView{state},enemy,attacks);
      threats+=std::any_of(attacks.begin(),attacks.end(),[&](Move m) {
        if (m.to!=flag) return false;
        const auto result=resolve(state.rank_at[m.from],Flag,rules);
        return result==Outcome::AttackerWins || result==Outcome::BothRemoved;
      });
    }
    if (!threats || found[threats-1][unassisted]) return;
    found[threats-1][unassisted]=true;
    const auto decision=compare_visibility(state,IdentityVisibility::AllIdentities);
    Move chosen{};
    Rng rng(42);
    single_case_check(DecisionScorer(decision.observation,rules,DecisionKind::FlagRoute).choose(rng,chosen) &&
                      is_legal(state,chosen),"archive pressure baseline selected an illegal action");
    auto after=state;
    step(after,chosen,rules);
    const auto score=team_score(after.result,seat);
    const auto proof=pressure_round(state,decision,rules,std::nullopt,state.id_at[flag],chosen,human,240);
    EvaluatorBounds baseline_team{}, safer_team{};
    int alternative=-1;
    if (proof.chosen.upper==0 && proof.first_safe) {
      auto safer=state;
      step(safer,*proof.first_safe,rules);
      int baseline_nodes=1000, safer_nodes=1000;
      baseline_team=bounded_team_value(after,rules,seat,5,baseline_nodes);
      safer_team=bounded_team_value(safer,rules,seat,5,safer_nodes);
      alternative=early_action_key(*proof.first_safe);
    }
    std::cout<<"P "<<key<<' '<<state.ply+1<<' '<<threats<<' '<<unassisted<<' '<<decision.actions.size()<<' '
             <<proof.safe<<' '<<proof.lost<<' '<<proof.unknown<<' '<<proof.chosen.lower<<' '<<proof.chosen.upper<<' '
             <<proof.guard.lower<<' '<<proof.guard.upper<<' '<<(score?int(*score*2):-1)<<' '<<proof.nodes<<' '
             <<early_action_key(chosen)<<' '<<early_action_key(human)<<' '<<alternative<<' '
             <<baseline_team.lower<<' '<<baseline_team.upper<<' '<<safer_team.lower<<' '<<safer_team.upper<<'\n';
  }
};

using PressureAdvance=std::function<void(const State&,const StepInfo&,const State&)>;

template <class Scan>
static RuleComparison scan_jqh_pressure(Scan& scan,const Rules& local,int cap,const PressureAdvance& advance={}) {
  int8_t layouts[4][25]{};
  std::vector<std::array<int,4>> rows;
  single_case_check(read_qh(layouts,rows) && cap<=static_cast<int>(rows.size()),"invalid JQH pressure input");
  const int from=point(rows[0][0],rows[0][1]);
  single_case_check(from>=0 && Board::get().node(from).seat>=0,"JQH missing first seat");
  Rules course;
  course.ccw=false;
  course.no_progress_limit=0;
  State shadow, reference;
  init_state(shadow,layouts,Board::get().node(from).seat,local);
  init_state(reference,layouts,shadow.to_move,course);
  for (int i=0;i<cap;++i) {
    const auto& row=rows[i];
    const int source=point(row[0],row[1]), target=point(row[2],row[3]);
    if (source<0 || target<0) return {i,i+1,"control_or_coordinate"};
    const Move move{uint8_t(source),uint8_t(target)};
    if (!is_legal(reference,move) || !is_legal(shadow,move)) return {i,i+1,"illegal"};
    const auto before=shadow;
    const auto recorded=step(reference,move,course), projected=step(shadow,move,local);
    if (const auto why=replay_difference(shadow,reference,projected,recorded)) return {i,i+1,why};
    scan.inspect(before,move,local);
    if (advance) advance(before,projected,shadow);
  }
  return {cap,0,"matched"};
}

template <class Scan>
static RuleComparison scan_jgs_pressure(Scan& scan,const ArchiveRows& raw,const Rules& archive,
                                        const Rules& local,int cap,const PressureAdvance& advance={}) {
  ArchiveReview reference;
  init_state(reference.state,raw.layouts,archive_first_seat(raw),archive);
  auto shadow=reference.state;
  for (int i=0;shadow.ply<cap && i<raw.entries;) {
    const auto before=shadow;
    const auto row=raw.row(i++);
    reference.play(raw,row,i,archive);
    while (i<raw.entries && raw.byte(raw.row(i))==245) {
      const auto control=raw.row(i++);
      reference.control(raw,control,i,archive);
    }
    const auto move=reference.last.move;
    if (!is_legal(shadow,move)) return {before.ply,before.ply+1,"illegal"};
    const auto result=step(shadow,move,local);
    if (const auto why=replay_difference(shadow,reference.state,result,reference.last))
      return {before.ply,before.ply+1,why};
    scan.inspect(before,move,local);
    if (advance) advance(before,result,shadow);
    if (reference.ended) break;
    reference.frames.str("");
  }
  return {shadow.ply,0,"matched"};
}

static void run_archive_pressure(const char* rules_path,const Rules& rules) {
  std::string tag,key;
  int ccw,cap;
  const auto config=read_text(rules_path);
  while (std::cin>>tag>>key>>ccw>>cap) {
    single_case_check((ccw==0 || ccw==1) && cap>=100 && cap<=140,"invalid pressure replay plan");
    ArchivePressure scan{key};
    auto local=rules;
    local.ccw=ccw!=0;
    RuleComparison result;
    if (tag=="H") result=scan_jqh_pressure(scan,local,cap);
    else {
      Rules archive;
      std::string tape,error;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid JGS pressure input");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      result=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cap);
    }
    std::cout<<"A "<<key<<' '<<result.agrees<<' '<<result.divergence<<' '<<result.reason<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete archive pressure input");
}
