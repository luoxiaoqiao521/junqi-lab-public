#pragma once

static int64_t shadow_ns(std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
}

static int shadow_phase(const State& s) {
  const auto retired=std::count(std::begin(s.out),std::end(s.out),true);
  if (retired>=2) return 3;
  if (retired==1) return 2;
  return s.ply<20?0:1;
}

static void shadow_json_controls() {
  const auto literal=parse_json(R"("军棋 🧩 it's")");
  const auto escaped=parse_json(R"("\u519b\u68cb \ud83e\udde9 it\u0027s")");
  single_case_check(escaped.str()==literal.str() &&
                    parse_json(dump_json(parse_json(R"("\u0000\b\f")"))).str()==std::string("\0\b\f",3),
                    "JSON escaped strings differ from their literal equivalents");
  for (const auto* invalid:{R"("\ud83e")",R"("\udde9")",R"("\ud83e\u0027")",
                            R"("\u00-1")",R"({"a":1,"\u0061":2})"}) {
    bool rejected=false;
    try { parse_json(invalid); }
    catch (const std::runtime_error&) { rejected=true; }
    single_case_check(rejected,"malformed escape or disguised duplicate accepted");
  }
}

static void shadow_mine_and_bomb(const Rules& rules) {
  constexpr Move clash{50,51};
  for (const Rank attacker:{Marshal,Bomb}) {
    auto root=selfplay_fixture();
    place_single_case_piece(root,1,clash.from,attacker);
    place_single_case_piece(root,2,4,Platoon);
    place_single_case_piece(root,26,clash.to,Mine);
    place_single_case_piece(root,27,52,Platoon);
    const auto view=observe(root,root.to_move,IdentityVisibility::AllIdentities);
    single_case_check(is_legal(root,clash) && view.rank_at(clash.to)==Mine,"visible mine fixture invalid");
    const auto event=step(root,clash,rules);
    if (attacker==Marshal) {
      single_case_check(event.outcome==Outcome::DefenderWins && !root.alive[1] && root.alive[26] &&
                        (event.revealed_mask&1),"visible mine did not beat marshal and reveal the flag");
    } else {
      single_case_check(event.outcome==Outcome::BothRemoved && !root.alive[1] && !root.alive[26],
                        "visible mine and bomb did not both die");
    }
  }
}

static void run_product_shadow(const char* log) {
  const std::string text=read_text(log);
  single_case_check(!text.empty() && text.back()=='\n',"product log not complete");
  std::istringstream input(text);
  std::string line;
  std::getline(input,line);
  const auto header=parse_json(line);
  Rules rules;
  IdentityVisibility visibility;
  auto live=replay_initial(header,rules,&visibility);
  const auto policy=simulation_policy(header.at("policy").str());
  const bool settlement=header.object.count("settlement_detail")!=0;
  Rng rng(mix_seed(std::stoull(header.at("seed").str()),1));
  single_case_check(bool(std::getline(input,line)) &&
    dump_json(parse_json(line))==dump_json(parse_json(frame_json(live,nullptr,visibility))),"first frame differs");
  std::cout<<"H "<<rules_hash(rules,live.match)<<' '<<int(visibility)<<' '<<state_hash(live)<<'\n';
  int probes=0;
  while (std::getline(input,line)) {
    const auto frame=parse_json(line);
    const auto observation=observe(live,live.to_move,visibility);
    const int phase=shadow_phase(live);
    auto begin=std::chrono::steady_clock::now();
    std::vector<Move> legal;
    gen_moves(observation,observation.viewer,legal);
    const auto generation=shadow_ns(begin);
    begin=std::chrono::steady_clock::now();
    const DecisionScorer scorer(observation,rules,policy.at(live.to_move));
    Move chosen;
    single_case_check(scorer.choose(rng,chosen) &&
                      std::find(legal.begin(),legal.end(),chosen)!=legal.end(),"shared scorer failed");
    const auto selection=shadow_ns(begin);
    if (visibility==IdentityVisibility::AllIdentities && probes==0 && legal.size()>1) {
      const auto alternative=*std::find_if(legal.begin(),legal.end(),[&](Move m){return !(m==chosen);});
      Rng copy=rng;
      Move injected;
      single_case_check(preferred_move(legal,copy,injected,[&](Move m){return m==alternative?1:0;}) &&
                        injected==alternative && !(chosen==injected),"injected selection could not change a legal choice");
      std::cout<<"P "<<early_action_key(chosen)<<' '<<early_action_key(injected)<<'\n';
      ++probes;
    }
    auto stepped=live;
    begin=std::chrono::steady_clock::now();
    const auto event=step(stepped,chosen,rules);
    const auto advance=shadow_ns(begin);
    State undo;
    const auto archived=replay_frame(frame,rules,live,undo,visibility,policy,false,settlement);
    single_case_check(chosen==archived.move && event.end_reason==archived.end_reason &&
                      state_hash(stepped)==state_hash(live),"shadow selection or outcome differs from product");
    const auto after_hash=state_hash(live);
    restore(live,undo);
    single_case_check(state_hash(live)==state_hash(undo) &&
                      step(live,chosen,rules).outcome==archived.outcome &&
                      state_hash(live)==after_hash,"undo and replay drift");
    std::cout<<"M "<<live.ply<<' '<<phase<<' '<<legal.size()<<' '<<generation<<' '<<selection<<' '<<advance
             <<' '<<state_hash(live)<<' '<<int(archived.outcome)<<'\n';
  }
  single_case_check(live.result!=Result::Ongoing,"incomplete product game");
  if (visibility==IdentityVisibility::AllIdentities) {
    shadow_json_controls();
    shadow_mine_and_bomb(rules);
    route_switch_controls(rules);
    selfplay_elimination_cases();
  }
  std::cout<<"E "<<live.ply<<' '<<int(live.result)<<' '<<state_hash(live)<<' '<<probes<<'\n';
}
