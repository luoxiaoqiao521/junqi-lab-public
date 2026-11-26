#pragma once

struct PressureObjective {
  int viewer, flag;
  bool team_result;
};

static EvaluatorBounds pressure_value(const State& state,const Rules& rules,const PressureObjective& goal,
                                      std::optional<Move> nonresponse,int depth,int& budget) {
  if (goal.team_result) {
    if (const auto score=team_score(state.result,goal.viewer)) return {int(*score*2),int(*score*2)};
  } else {
    if (!state.alive[goal.flag]) return {0,0};
    if (state.to_move==goal.viewer || team_score(state.result,goal.viewer)) return {2,2};
  }
  if (state.result!=Result::Ongoing || !depth || !budget) return {};
  --budget;
  std::vector<Move> actions;
  legal_moves(state,actions);
  if (actions.empty()) return {};
  const bool friendly=team_of(state.to_move,state.match)==team_of(goal.viewer,state.match);
  if (friendly && state.to_move!=goal.viewer && nonresponse) {
    if (std::find(actions.begin(),actions.end(),*nonresponse)==actions.end()) return {};
    auto branch=state;
    step(branch,*nonresponse,rules);
    return pressure_value(branch,rules,goal,std::nullopt,depth-1,budget);
  }
  EvaluatorBounds result=friendly?EvaluatorBounds{0,0}:EvaluatorBounds{2,2};
  for (auto move:actions) {
    auto branch=state;
    step(branch,move,rules);
    const auto child=pressure_value(branch,rules,goal,nonresponse,depth-1,budget);
    if (result.include(child,friendly)) break;
  }
  return result;
}

static State pressure_position(bool late,bool unavailable) {
  auto state=selfplay_fixture();
  state.out[2]=state.out[3]=false;
  place_single_case_piece(state,1,22,Marshal);
  place_single_case_piece(state,26,21,Engineer);
  place_single_case_piece(state,27,54,Platoon);
  place_single_case_piece(state,50,86,Flag);
  place_single_case_piece(state,51,20,unavailable?Mine:Brigade);
  place_single_case_piece(state,52,67,Platoon);
  place_single_case_piece(state,75,116,Flag);
  place_single_case_piece(state,76,late?24:25,Engineer);
  return state;
}

static void verify_pressure_witness(const Rules& rules) {
  auto after_east=pressure_position(false,false);
  const auto play=[&](State& s,int seat,Move move) {
    single_case_check(s.to_move==seat && is_legal(s,move),"dual-pressure witness has wrong turn or illegal move");
    step(s,move,rules);
  };
  play(after_east,0,{22,21});
  play(after_east,1,{54,53});

  auto helped=after_east;
  play(helped,2,{20,25});
  single_case_check(helped.alive[0] && helped.to_move==0,
                    "ally failed to remove the second flag threat before it could move");

  auto unsupported=after_east;
  play(unsupported,2,{67,68});
  play(unsupported,3,{25,26});
  single_case_check(!unsupported.alive[0] && unsupported.out[0],"quiet ally reply must permit the flag capture");
}

struct PressureProof {
  int safe=0, lost=0, unknown=0, nodes=0;
  EvaluatorBounds chosen{}, guard{};
  std::optional<Move> first_safe;
};

static PressureProof pressure_round(const State& root,const SeatDecision& decision,const Rules& rules,
                                    std::optional<Move> nonresponse,int flag,Move chosen,Move guard,int limit=12000) {
  PressureProof result;
  for (const auto move:decision.actions) {
    auto branch=root;
    step(branch,move,rules);
    int budget=limit;
    const auto proof=pressure_value(branch,rules,{root.to_move,flag,false},nonresponse,4,budget);
    result.nodes+=limit-budget;
    result.safe+=proof.lower==2;
    if (proof.lower==2 && !result.first_safe) result.first_safe=move;
    result.lost+=proof.upper==0;
    result.unknown+=proof.lower!=proof.upper;
    if (move==chosen) result.chosen=proof;
    if (move==guard) result.guard=proof;
  }
  return result;
}

static void run_pressure_case(const Rules& rules,int variant,int rotation) {
  const auto root=rotate_panel(pressure_position(variant==3,variant==2),rotation);
  const auto node=[&](int n){return static_cast<uint8_t>(rotate_panel_node(n,rotation));};
  const auto decision=compare_visibility(root,IdentityVisibility::AllIdentities);
  const Move prescribed{node(67),node(68)}, guard{node(22),node(21)};
  const auto hold=variant==1?std::optional<Move>{prescribed}:std::nullopt;
  single_case_check(std::find(decision.actions.begin(),decision.actions.end(),guard)!=decision.actions.end(),
                    "dual pressure control lost the first defense");
  int immediate=0;
  for (int enemy:{(rotation+1)%4,(rotation+3)%4}) {
    std::vector<Move> attacks;
    gen_moves(StateView{root},enemy,attacks);
    immediate+=std::any_of(attacks.begin(),attacks.end(),[&](Move m){return m.to==node(26);});
  }
  Move actual{};
  Rng rng(42);
  const DecisionScorer scorer(decision.observation,rules,DecisionKind::FlagRoute);
  single_case_check(scorer.choose(rng,actual) && is_legal(root,actual),"practice choice is illegal");
  const auto result=pressure_round(root,decision,rules,hold,rotation*25,actual,guard);
  auto next=root;
  step(next,guard,rules);
  int team_nodes=12000;
  const auto team=pressure_value(next,rules,{rotation,rotation*25,true},hold,5,team_nodes);
  std::cout<<"D "<<rotation<<' '<<variant<<' '<<immediate<<' '<<decision.actions.size()<<' '
           <<result.safe<<' '<<result.lost<<' '<<result.unknown<<' '<<early_action_key(actual)<<' '
           <<result.chosen.lower<<' '<<result.chosen.upper<<' '<<result.guard.lower<<' '<<result.guard.upper<<' '
           <<team.lower<<' '<<team.upper<<' '
           <<result.nodes<<' '<<12000-team_nodes<<'\n';
}

static void run_dual_pressure(const Rules& rules) {
  verify_pressure_witness(rules);
  for (int rotation=0;rotation<kSeats;++rotation) for (int variant:{0,1,2,3})
    run_pressure_case(rules,variant,rotation);
}
