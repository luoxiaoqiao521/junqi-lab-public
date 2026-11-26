#pragma once

static Move route_baseline(const State& state,const Rules& rules,Rng& rng) {
  Move result;
  const auto observation=observe(state,state.to_move,IdentityVisibility::AllIdentities);
  single_case_check(DecisionScorer(observation,rules,DecisionKind::FlagRoute).choose(rng,result) &&
                    is_legal(state,result),"route policy failed at legal decision");
  return result;
}

static void route_switch_controls(const Rules& rules) {
  constexpr Move upper{50,51}, outside{50,55};
  for (int condition:{0,1,2,3}) {
    auto root=selfplay_fixture();
    place_single_case_piece(root,1,50,Engineer);
    place_single_case_piece(root,26,51,Mine);
    place_single_case_piece(root,27,std::array<int,4>{30,52,55,52}[condition],Marshal);
    Rng rng(42);
    const auto baseline=route_baseline(root,rules,rng);
    const int depth=condition==3?2:3;
    const auto main=full_route_proof(root,upper,rules,depth), side=full_route_proof(root,outside,rules,depth);
    const int expected_main=condition==1 || condition==3?0:2;
    single_case_check(baseline==upper && main.upper==expected_main && main.lower==expected_main &&
                      side.lower==(condition>=2?0:2) &&
                      side.upper==(condition==2?0:2),"route switch no longer has the registered competing outcomes");

    const auto decision=compare_visibility(root,IdentityVisibility::AllIdentities);
    const auto dark=compare_visibility(root,IdentityVisibility::OwnAndPublic);
    const auto dual=compare_visibility(root,IdentityVisibility::AllyAndPublic);
    single_case_check(decision.actions==dark.actions && decision.actions==dual.actions,
                      "visibility changed legal route coverage");
    std::optional<Move> selected=baseline;
    if (main.upper<2) {
      selected.reset();
      for (const auto action:decision.actions) {
        if (full_route_proof(root,action,rules,depth).lower==2) { selected=action; break; }
      }
    }
    single_case_check(condition==3 ? !selected : selected==(condition==1?outside:upper),
                      "certified route differs or opt-out did not occur");
    std::cout<<"C "<<condition<<' '<<decision.actions.size()<<' '<<early_action_key(baseline)<<' '
             <<(selected?early_action_key(*selected):-1)<<' '<<main.lower<<' '<<main.upper<<' '
             <<side.lower<<' '<<side.upper<<'\n';
  }
}

static int find_route_slot(int seat,int node) {
  for (int slot=0;slot<kPiecesPerSeat;++slot) if (layout_slot_node(seat,slot)==node) return slot;
  throw std::runtime_error("mine panel contains a non-deployment node");
}

static void validate_mine_panel(const std::array<int,3>& local) {
  for (int seat=0;seat<kSeats;++seat) for (bool mirror:{false,true}) {
    int8_t layout[kPiecesPerSeat];
    Rng rng(20261123+seat);
    generate_layout(layout,rng);
    const auto& board=Board::get();
    const auto slot=[&](int point) { return find_route_slot(seat,board.seat_node(seat,point/5,mirror?4-point%5:point%5)); };
    const auto flag=slot(26);
    std::swap(layout[flag],layout[formation_flag_slot(layout)]);
    std::array<bool,kPiecesPerSeat> targets{};
    for (const int point:local) {
      const int dest=slot(point);
      single_case_check(!targets[dest] && dest!=flag,"duplicate mine or flag collision in panel");
      targets[dest]=true;
    }
    for (int dest=0;dest<kPiecesPerSeat;++dest) if (targets[dest] && layout[dest]!=Mine) {
      int source=0;
      while (source<kPiecesPerSeat && (targets[source] || layout[source]!=Mine)) ++source;
      single_case_check(source<kPiecesPerSeat,"not enough swap mines for panel");
      std::swap(layout[dest],layout[source]);
    }
    single_case_check(valid_layout(layout),"image panel could not be placed legally");
  }
}

static void compare_panel_routes(const Rules& rules,int id,const std::array<int,3>& local,int guard) {
  auto root=selfplay_fixture();
  place_single_case_piece(root,1,46,Engineer);
  place_single_case_piece(root,29,guard?45:30,guard?Marshal:Platoon);
  for (int i=0;i<3;++i) place_single_case_piece(root,26+i,local[i]+30,Mine);
  const auto mask=[&] (int point) { return std::find(local.begin(),local.end(),point)!=local.end(); };
  const int entries=int(mask(21))|int(mask(25))*2|int(mask(27))*4;
  const auto top_now=full_route_proof(root,{46,51},rules,3);
  const auto top=full_route_proof(root,{46,51},rules,5);
  const auto outside=full_route_proof(root,{46,50},rules,5);
  std::cout<<"M "<<id<<' '<<guard<<' '<<entries<<' '<<top_now.lower<<' '<<top_now.upper<<' '
           <<top.lower<<' '<<top.upper<<' '<<outside.lower<<' '<<outside.upper<<'\n';
}

static void run_route_panels(const Rules& rules) {
  route_switch_controls(rules);
  int id,r1,r2,r3;
  while (std::cin>>id>>r1>>r2>>r3) {
    const std::array<int,3> local{r1,r2,r3};
    validate_mine_panel(local);
    for (int guard:{0,1}) compare_panel_routes(rules,id,local,guard);
  }
  single_case_check(std::cin.eof(),"incomplete route panel payload");
}

static void run_route_resources(const Rules& rules) {
  int id,r1,r2,r3;
  while (std::cin>>id>>r1>>r2>>r3) {
    for (int attacker:{Engineer,Marshal}) {
      auto root=selfplay_fixture();
      place_single_case_piece(root,1,46,attacker);
      place_single_case_piece(root,29,34,Platoon);
      const std::array<int,3> mines{r1,r2,r3};
      for (int i=0;i<3;++i) place_single_case_piece(root,26+i,mines[i]+30,Mine);
      Rng rng(42);
      const auto baseline=route_baseline(root,rules,rng);
      std::cout<<"R "<<id<<' '<<attacker<<' '<<early_action_key(baseline);
      for (const auto move:{Move{46,51},Move{46,50},Move{46,52}}) {
        const auto bound=full_route_proof(root,move,rules,5);
        std::cout<<' '<<bound.lower<<' '<<bound.upper;
      }
      std::cout<<'\n';
    }
  }
  single_case_check(std::cin.eof(),"incomplete mine resource payload");
}

static void route_continuation(State state,Move action,const Rules& rules,int view,int table,uint64_t seed,
                               const std::string& key,int cut,int arm,int stream,
                               std::optional<Move> first_reply=std::nullopt,bool vary_partner=true) {
  auto streams=review_streams(seed);
  int own_flag=0;
  const auto play=[&](Move move) {
    single_case_check(is_legal(state,move),"locked route became illegal");
    const auto event=step(state,move,rules);
    if (event.exit_reason[view]==ExitReason::FlagRemoved) own_flag=state.ply;
  };
  play(action);
  if (first_reply) {
    single_case_check(state.result==Result::Ongoing && team_of(state.to_move)!=team_of(view),
                      "forced reply does not belong to an acting opponent");
    play(*first_reply);
  }
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    auto& rng=streams[actor];
    if (vary_partner && table==1 && actor==(view+2)%4) {
      Move next;
      const auto observation=observe(state,actor,IdentityVisibility::AllIdentities);
      single_case_check(heuristic_move(observation,rng,next),"frozen partner did not act");
      play(next);
    } else play(route_baseline(state,rules,rng));
  }
  const auto team=team_score(state.result,view);
  std::cout<<"G "<<key<<' '<<cut<<' '<<arm<<' '<<table<<' '<<stream<<' '<<early_action_key(action)<<' '
           <<(team?int(*team*2):-1)<<' '<<state.ply<<' '<<own_flag<<' '<<state_hash(state)<<'\n';
}

struct RouteReplay {
  std::string key;
  int cut=0, current=-1, alternative=-1;
  uint64_t seed=0;
  bool found=false;

  void inspect(const State& root,Move human,const Rules& rules) {
    if (root.ply+1!=cut) return;
    const auto unpack=[](int action) { return Move{uint8_t(action/kNodes),uint8_t(action%kNodes)}; };
    Rng rng(42);
    single_case_check(route_baseline(root,rules,rng)==unpack(current) && current!=alternative,
                      "earlier first-choice baseline no longer matches the registered replay intervention");
    found=true;
    const std::array<Move,3> actions{unpack(current),unpack(alternative),human};
    for (int arm=0;arm<3;++arm) for (int table:{0,1}) for (int stream=0;stream<8;++stream) {
      route_continuation(root,actions[arm],rules,root.to_move,table,mix_seed(seed,stream),key,cut,arm,stream);
    }
  }
};

static void run_route_replays(const char* rules_path,const Rules& rules) {
  std::string tag,tape,error;
  const auto config=read_text(rules_path);
  RouteReplay cut;
  int ccw,cap;
  while (std::cin>>tag>>cut.key>>ccw>>cap>>cut.cut>>cut.current>>cut.alternative>>cut.seed) {
    auto local=rules;
    local.ccw=ccw!=0;
    cut.found=false;
    single_case_check((ccw==0 || ccw==1) && cut.cut>0 && cut.cut<=cap && cut.current>=0 && cut.alternative>=0 &&
                      cut.current<kNodes*kNodes && cut.alternative<kNodes*kNodes,"invalid route replay identity");
    if (tag=="H") scan_jqh_pressure(cut,local,cut.cut);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid route replay input");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      scan_jgs_pressure(cut,ArchiveRows(decode_archive_tape(tape)),archive,local,cut.cut);
    }
    single_case_check(cut.found,"route cut did not satisfy the archived rule alignment");
  }
  single_case_check(std::cin.eof(),"incomplete route replay payload");
}
