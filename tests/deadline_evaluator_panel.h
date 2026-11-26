#pragma once

static int rotate_panel_node(int node,int rotation) {
  return (node+rotation*30)%120;
}

static jq::State rotate_panel(jq::State source,int rotation) {
  using namespace jq;
  auto state=source;
  std::fill(std::begin(state.id_at),std::end(state.id_at),int8_t(kNone));
  std::fill(std::begin(state.seat_at),std::end(state.seat_at),int8_t(kNone));
  std::fill(std::begin(state.rank_at),std::end(state.rank_at),int8_t(Unknown));
  std::fill(std::begin(state.alive),std::end(state.alive),false);
  state.to_move=static_cast<int8_t>((source.to_move+rotation)%kSeats);
  for (int seat=0;seat<kSeats;++seat) state.out[(seat+rotation)%kSeats]=source.out[seat];
  for (int n=0;n<kNodes;++n) if (source.seat_at[n]>=0) {
    single_case_check(n<120,"unexpected centre piece in synthetic panel");
    place_single_case_piece(state,(source.id_at[n]+rotation*25)%100,rotate_panel_node(n,rotation),source.rank_at[n]);
  }
  return state;
}

static void score_panel_actions(const jq::State& state,jq::Move first,const jq::Rules& rules,
    const jq::ShortWeights& peer,uint64_t seed,bool trace_gate=false) {
  using namespace jq;
  single_case_check(is_legal(state,first),"panel supplied illegal action");
  auto branch=state;
  step(branch,first,rules);
  int budget=12000;
  const auto proof=bounded_team_value(branch,rules,state.to_move,11,budget);
  const PublicReplyHistory prefix(state,false);
  const OpeningMarks observer(observe(state,state.to_move),false);
  std::cout<<' '<<early_action_key(first)<<' '<<proof.lower<<' '<<proof.upper<<' '<<12000-budget;
  for (int variant:{0,1,2}) {
    ResponseAudit audit{variant==0?9:12,0,0,variant==2,true};
    ResponseValue values;
    const int samples=variant==0?4:3;
    for (int i=0;i<samples;++i)
      response_trial(state,first,observer,prefix,rules,peer,i%2,mix_seed(seed,i),values,false,
                     RoutePriority::Joint,true,&audit);
    single_case_check(audit.plies<=36,"rollout transition allowance exceeded");
    std::cout<<' '<<values.value/samples<<' '<<audit.plies<<' '<<audit.extended;
    if (trace_gate && variant==2) {
      std::cout<<' '<<audit.gate_visits<<' '<<audit.early_settled<<' '<<audit.early_unresolved<<' '<<audit.extension_steps;
      for (const int count:audit.gate_reasons) std::cout<<' '<<count;
    }
  }
}

static void compare_evaluator_panel(const jq::Rules& rules,const jq::ShortWeights& peer,bool trace_gate=false) {
  using namespace jq;
  prove_evaluator_controls(rules);
  std::cout<<std::setprecision(17);
  for (int family:{0,1}) {
    auto root=selfplay_fixture();
    const Move bait{4,30}, retreat{4,3}, advance{46,51}, wait{46,45};
    if (family==0) {
      place_single_case_piece(root,1,4,Brigade);
      place_single_case_piece(root,26,30,Engineer);
      place_single_case_piece(root,27,21,Engineer);
    } else {
      place_single_case_piece(root,1,46,Marshal);
      place_single_case_piece(root,26,51,Platoon);
      place_single_case_piece(root,27,30,Engineer);
    }
    for (int remaining:{1,2,5,9,10,11}) for (int rotation:{0,1,2,3}) {
      auto state=rotate_panel(root,rotation);
      state.no_progress=rules.no_progress_limit-remaining;
      std::cout<<"panel "<<family<<' '<<remaining<<' '<<rotation;
      for (const auto move:family==0?std::array<Move,2>{bait,retreat}:std::array<Move,2>{advance,wait}) {
        const Move action{static_cast<uint8_t>(rotate_panel_node(move.from,rotation)),
                          static_cast<uint8_t>(rotate_panel_node(move.to,rotation))};
        score_panel_actions(state,action,rules,peer,uint64_t(241400+family*100+remaining),trace_gate);
      }
      std::cout<<'\n';
    }
  }
}
