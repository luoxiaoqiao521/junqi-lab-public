#pragma once

static bool heritage_survives(const SeatDecision& d,const OpeningMarks& e,Move move,const Rules& rules) {
  const auto& view=d.observation;
  const auto target=compatible_rank_mask(view,e,move.to);
  return target && collision_outcome_mask(view.rank_at(move.from),target,rules)==outcome_bit(Outcome::AttackerWins);
}

static std::vector<Move> heritage_contact_list(const SeatDecision& d,const OpeningMarks& e,
    const std::vector<Move>& narrow,const std::vector<Move>& expanded,
    const StudentNetwork& model,const Rules& rules,bool action_level) {
  if (!action_level && std::any_of(narrow.begin(),narrow.end(),[&](Move m){return heritage_survives(d,e,m,rules);}))
    return expanded;
  std::vector<Move> contacts;
  for (const auto move:d.actions)
    if (std::find(narrow.begin(),narrow.end(),move)==narrow.end() && heritage_survives(d,e,move,rules))
      contacts.push_back(move);
  if (contacts.empty()) return expanded;
  const auto pick=student_pick(d,e,contacts.front(),contacts,model,rules,false);
  auto result=narrow;
  if (result.size()<expanded.size()) result.push_back(pick);
  return result;
}

static void heritage_games(const ReplanRoot& root,const std::array<Move,4>& choices,const Rules& rules,
                           const StudentNetwork& model,OrdinaryDarkPeers peers,int viewer,int table,uint64_t salt,int repetitions=3) {
  for (int arm=0;arm<4;++arm) {
    const Move move=choices[arm];
    const auto existing=std::find(choices.begin(),choices.begin()+arm,move);
    if (existing!=choices.begin()+arm || std::all_of(choices.begin(),choices.end(),[&](Move m){return m==move;})) continue;
    for (const int desk:{table,3}) for (int rep=0;rep<repetitions;++rep) {
      const auto stream=review_streams(mix_seed(salt,497821+rep));
      const auto result=finish_replan(root.ordinary,move,viewer,desk,rules,model,peers,stream);
      std::cout<<"G "<<early_action_key(move)<<' '<<desk<<' '<<rep<<' '<<result.result<<' '<<result.steps<<'\n';
    }
  }
}

static void audit_heritage_nominations(uint64_t seed,int viewer,int table,int cut,const Rules& rules,
                                       const StudentNetwork& model,bool action_level,bool force_contact) {
  const OrdinaryDarkPeers peers;
  const auto root=restore_replan_root(seed,viewer,table,cut,-1,rules,model,peers,true);
  const auto& game=root.ordinary;
  const bool valid=game.state.to_move==viewer && game.state.result==Result::Ongoing && root.anchor.has_value();
  std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<game.state.ply<<' '<<valid<<'\n';
  if (!valid) return;
  const auto decision=seat_decision(game.state);
  const auto evidence=game.reply_history.for_actor(decision.observation,game.marks[viewer]);
  auto stream=root.streams[viewer];
  const auto fallback=ordinary_dark_move(game,decision,rules,OrdinaryDarkPolicy::Tactical,stream);
  const auto narrow=budget_candidates(decision,fallback,rules,4);
  const auto expanded=budget_candidates(decision,fallback,rules,5);
  const auto base=student_pick(decision,evidence,fallback,narrow,model,rules,false);
  const auto wide=student_pick(decision,evidence,fallback,expanded,model,rules,false);
  const auto salt=public_rollout_seed(decision.observation,evidence,decision.observation.quiet_clock);
  const ReplanPortfolio portfolio{base,wide,expanded,int(narrow.size())};
  auto reused=expanded;
  int work=0,isolation=0;
  const auto worlds=assemble_replan_worlds(root.path,decision,rules,salt,2);
  if (worlds.games.size()==2) {
    reused=seat_nominations(decision,portfolio,worlds.games,rules,peers,salt,work);
    if (seed%16==0) {
      const auto swap=seat_decision(worlds.games.back().state);
      const auto swap_e=worlds.games.back().reply_history.for_actor(swap.observation,worlds.games.back().marks[viewer]);
      int other_work=0;
      single_case_check(same_observation(swap.observation,decision.observation) && swap.actions==decision.actions &&
                        equal_dark_marks(swap_e,evidence) &&
                        seat_nominations(swap,portfolio,worlds.games,rules,peers,salt,other_work)==reused &&
                        other_work==work,"reused nomination leaked true hidden identity");
      isolation=1;
    }
  }
  const auto targeted=heritage_contact_list(decision,evidence,narrow,expanded,model,rules,action_level);
  const std::array<std::vector<Move>,4> lists{narrow,expanded,reused,targeted};
  std::array<Move,4> choices{};
  std::cout<<"W "<<work<<' '<<worlds.games.size()<<' '<<worlds.rejected<<' '<<isolation<<'\n';
  for (int arm=0;arm<4;++arm) {
    const auto& actions=lists[arm];
    choices[arm]=student_pick(decision,evidence,fallback,actions,model,rules,false);
    if (force_contact && arm==3) {
      const bool omitted=actions.size()>narrow.size() && heritage_survives(decision,evidence,actions.back(),rules);
      choices[arm]=omitted?actions.back():choices[0];
    }
    std::cout<<"A "<<arm<<' '<<early_action_key(choices[arm])<<' '<<actions.size();
    for (const auto move:actions) std::cout<<' '<<early_action_key(move);
    std::cout<<'\n';
  }
  for (const auto move:decision.actions)
    if (heritage_survives(decision,evidence,move,rules)) std::cout<<"C "<<early_action_key(move)<<'\n';
  if (force_contact)
    heritage_games(root,{choices[0],choices[0],choices[0],choices[3]},rules,model,peers,viewer,table,
                   mix_seed(salt,679421),8);
  else
    heritage_games(root,choices,rules,model,peers,viewer,table,salt);
}

static void run_dark_heritage_nominations(const Rules& rules,bool action_level=false,bool force_contact=false) {
  StudentNetwork model;
  single_case_check(model.load(std::cin,true),"legacy nomination needs the active four-dark model");
  uint64_t seed;
  int viewer,table,cut;
  while (std::cin>>seed>>viewer>>table>>cut) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==1) && cut>=0 && cut<=160,
                      "invalid historical nomination assignment");
    audit_heritage_nominations(seed,viewer,table,cut,rules,model,action_level,force_contact);
  }
  single_case_check(std::cin.eof(),"incomplete historical nomination assignments");
}
