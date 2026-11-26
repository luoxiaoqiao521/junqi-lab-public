#pragma once

struct RouteRisk {
  // -1 means this move gives our own team a turn, so the immediate reply test does not apply.
  int level=-1;
  std::optional<Move> reply;
};

static RouteRisk next_enemy_risk(const State& root,Move action,const Rules& rules) {
  const int actor=root.to_move, moved=root.id_at[action.from];
  auto next=root;
  step(next,action,rules);
  if (const auto terminal=team_score(next.result,actor)) return {*terminal==0 ? 3 : 0,std::nullopt};
  if (next.result!=Result::Ongoing || team_of(next.to_move)==team_of(actor)) return {};
  RouteRisk worst{int(!next.alive[moved]),std::nullopt};
  std::vector<Move> replies;
  legal_moves(next,replies);
  for (const auto reply:replies) {
    auto after=next;
    const auto contact=step(after,reply,rules);
    int level=0;
    if (team_score(after.result,actor)==0.0) level=3;
    else if (contact.exit_reason[actor]==ExitReason::FlagRemoved ||
             contact.exit_reason[(actor+2)%4]==ExitReason::FlagRemoved) level=2;
    else if (!after.alive[moved]) level=1;
    if (level>worst.level) worst={level,reply};
    if (worst.level==3) break;
  }
  return worst;
}

static void terminal_counterattack_control(const Rules& rules) {
  auto state=selfplay_fixture();
  place_single_case_piece(state,1,51,Engineer);
  place_single_case_piece(state,26,21,Platoon);
  std::vector<Move> enemy;
  gen_moves(StateView{state},1,enemy);
  const Move wins{51,56};
  Rng rng(42);
  single_case_check(std::find(enemy.begin(),enemy.end(),Move{21,26})!=enemy.end() &&
                    route_baseline(state,rules,rng)==wins && next_enemy_risk(state,wins,rules).level==0,
                    "direct team victory must take precedence over enemy flag pressure");
  const auto event=step(state,wins,rules);
  single_case_check(team_score(state.result,0)==1.0 && event.exit_reason[1]==ExitReason::FlagRemoved,
                    "winning counterattack must end before the enemy reply");
  std::cout<<"C direct-team-win\n";
}

struct NaturalRoutePair {
  Move base, alternative, human;
  RouteRisk base_risk, alternative_risk, human_risk;
};

static std::optional<NaturalRoutePair> compare_natural_routes(const State& root,Move human,const Rules& rules,
                                                              Move base,const RouteRisk& risk) {
  const auto decision=compare_visibility(root,IdentityVisibility::AllIdentities);
  const DecisionScorer scorer(decision.observation,rules,DecisionKind::FlagRoute);
  std::optional<NaturalRoutePair> result;
  int best=-100000;
  for (const auto action:decision.actions) {
    if (action==base) continue;
    const auto alternative=next_enemy_risk(root,action,rules);
    if (alternative.level<0 || alternative.level>=risk.level) continue;
    const auto score=scorer.score(action).total();
    if (result && (alternative.level>result->alternative_risk.level ||
                   (alternative.level==result->alternative_risk.level && score<=best))) continue;
    result=NaturalRoutePair{base,action,human,risk,alternative,{}};
    best=score;
  }
  if (result) result->human_risk=next_enemy_risk(root,human,rules);
  return result;
}

struct NaturalRouteScan {
  std::string key;
  uint64_t seed=0;
  int paired=0, exposed=0, inspected=0, previous_actor=kNone;
  std::array<int,kSeats> last_sector{kNone,kNone,kNone,kNone};
  std::optional<State> previous_root;

  void inspect(const State& root,Move human,const Rules& rules) {
    const auto& board=Board::get();
    const int actor=root.to_move, sector_before=last_sector[actor], prior=previous_actor;
    const auto previous=previous_root;
    previous_root=root;
    previous_actor=actor;
    last_sector[actor]=board.node(human.to).seat;
    if (paired || root.ply<31 || !previous || prior==kNone || team_of(prior)==team_of(actor)) return;
    ++inspected;
    Rng rng(42);
    const auto base=route_baseline(root,rules,rng);
    const auto risk=next_enemy_risk(root,base,rules);
    if (!risk.reply) return;
    ++exposed;
    auto comparison=*previous;
    comparison.to_move=static_cast<int8_t>(actor);
    if (comparison.out[actor] || comparison.id_at[base.from]!=root.id_at[base.from] || !is_legal(comparison,base)) return;
    const auto earlier=next_enemy_risk(comparison,base,rules);
    if (earlier.level<0 || earlier.level>=risk.level) return;
    const auto pair=compare_natural_routes(root,human,rules,base,risk);
    if (!pair) return;
    ++paired;
    const int base_sector=board.node(base.to).seat, alt_sector=board.node(pair->alternative.to).seat;
    const auto action_code=[](const auto& move){return move?early_action_key(*move):-1;};
    std::cout<<"P "<<key<<' '<<root.ply+1<<' '<<actor<<' '<<int(root.out[(actor+2)%4])<<' '
             <<sector_before<<' '<<base_sector<<' '<<alt_sector<<' '<<early_action_key(base)<<' '
             <<early_action_key(pair->alternative)<<' '<<early_action_key(human)<<' '
             <<risk.level<<' '<<pair->alternative_risk.level<<' '<<pair->human_risk.level<<' '
             <<action_code(risk.reply)<<' '<<action_code(pair->alternative_risk.reply)<<' '<<earlier.level<<' '
             <<int(root.rank_at[base.from])<<' '<<int(root.rank_at[base.to])<<'\n';
    const std::array<Move,3> actions{base,pair->alternative,human};
    const std::array<RouteRisk,3> risks{risk,pair->alternative_risk,pair->human_risk};
    for (int arm=0;arm<3;++arm) for (int table:{0,1}) for (int stream=0;stream<4;++stream) {
      route_continuation(root,actions[arm],rules,actor,table,mix_seed(seed,stream),key,root.ply+1,arm,stream,
                         table==1?risks[arm].reply:std::nullopt,false);
    }
  }
};

static void run_natural_routes(const char* rules_path,const Rules& rules) {
  terminal_counterattack_control(rules);
  std::string tag,key,tape,error;
  const auto config=read_text(rules_path);
  int ccw,cap;
  uint64_t seed;
  while (std::cin>>tag>>key>>ccw>>cap>>seed) {
    single_case_check((ccw==0 || ccw==1) && cap>=72 && cap<=140,"invalid route discovery replay plan");
    NaturalRouteScan scan;
    scan.key=key;
    scan.seed=seed;
    auto local=rules;
    local.ccw=ccw!=0;
    RuleComparison result;
    if (tag=="H") result=scan_jqh_pressure(scan,local,cap);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid natural route replay input");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      result=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cap);
    }
    std::cout<<"A "<<key<<' '<<result.agrees<<' '<<result.divergence<<' '<<result.reason<<' '
             <<scan.inspected<<' '<<scan.exposed<<' '<<scan.paired<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete natural replay payload");
}
