#pragma once

struct EvaluatorBounds {
  int lower=0, upper=2;

  // Only the extreme preferred by the acting side settles the remaining siblings.
  bool include(EvaluatorBounds child,bool maximizing) {
    if (maximizing) {
      lower=std::max(lower,child.lower);
      upper=std::max(upper,child.upper);
      return lower==2;
    }
    lower=std::min(lower,child.lower);
    upper=std::min(upper,child.upper);
    return upper==0;
  }
};

// The proof assumes both teams know all ranks and optimize the specified team result.
static EvaluatorBounds bounded_team_value(const jq::State& state,const jq::Rules& rules,
                                          int root,int depth,int& budget) {
  using namespace jq;
  if (const auto final=team_score(state.result,root)) {
    const int value=static_cast<int>(*final*2);
    return {value,value};
  }
  if (state.result!=Result::Ongoing || depth==0 || budget==0) return {};
  --budget;
  std::vector<Move> actions;
  legal_moves(state,actions);
  if (actions.empty()) return {};
  const bool maximizing=team_of(state.to_move,state.match)==team_of(root,state.match);
  EvaluatorBounds out=maximizing?EvaluatorBounds{0,0}:EvaluatorBounds{2,2};
  for (auto move:actions) {
    auto next=state;
    step(next,move,rules);
    const auto branch=bounded_team_value(next,rules,root,depth-1,budget);
    if (out.include(branch,maximizing)) break;
  }
  return out;
}

static void print_proof(const char* label,const jq::State& state,jq::Move move,const jq::Rules& rules,
                         int depth,int low,int high,int limit=12000) {
  using namespace jq;
  single_case_check(is_legal(state,move),"illegal proof anchor");
  auto next=state;
  step(next,move,rules);
  int budget=limit;
  const auto result=bounded_team_value(next,rules,state.to_move,depth-1,budget);
  single_case_check(result.lower==low && result.upper==high,"proof disagrees with the registered control");
  std::cout<<label<<' '<<depth<<' '<<result.lower<<' '<<result.upper<<' '<<limit-budget<<'\n';
}

static void prove_evaluator_controls(const jq::Rules& rules) {
  using namespace jq;
  single_case_check(rules.no_progress_limit>0 && rules.flag_eliminates && !rules.draw_before_win &&
                    !rules.reset_on_capture_only,"unsupported proof fixture rules");
  auto choice=selfplay_fixture();
  place_single_case_piece(choice,1,55,Brigade);
  place_single_case_piece(choice,26,50,Marshal);
  std::vector<Move> moves;
  legal_moves(choice,moves);
  const Move surrender{55,50}, win{55,56};
  single_case_check(!moves.empty() && moves.front()==surrender && is_legal(choice,win),
                    "proof cutoff control needs a losing first child and a later winning child");
  auto loses=choice, wins=choice;
  step(loses,surrender,rules);
  step(wins,win,rules);
  single_case_check(team_score(loses.result,0)==0.0 && team_score(wins.result,0)==1.0,
                    "proof cutoff control must settle both actions immediately");
  for (int viewer:{0,1}) {
    int nodes=1;
    const auto proof=bounded_team_value(choice,rules,viewer,1,nodes);
    const int expected=viewer==0?2:0;
    single_case_check(proof.lower==expected && proof.upper==expected,"adverse first reply ended the proof early");
    std::cout<<"E1_"<<(viewer==0?"max":"min")<<"_first_child 1 "<<proof.lower<<' '<<proof.upper<<" 1\n";
  }
  auto bait=selfplay_fixture();
  bait.no_progress=rules.no_progress_limit-1;
  place_single_case_piece(bait,1,4,Brigade);
  place_single_case_piece(bait,26,30,Engineer);
  place_single_case_piece(bait,27,21,Engineer);
  const Move take{4,30}, quiet{4,3}, equal{4,2};
  auto enemy=bait;
  step(enemy,take,rules);
  single_case_check(enemy.to_move==1 && is_legal(enemy,{21,26}),"retaliation witness is unreachable");
  step(enemy,{21,26},rules);
  single_case_check(team_score(enemy.result,0)==0.0,"retaliation did not settle the game");
  print_proof("E1_bait_before_reply",bait,take,rules,1,0,2);
  print_proof("E1_bait_after_reply",bait,take,rules,2,0,0);
  print_proof("E1_bait_quiet",bait,quiet,rules,2,1,1);
  print_proof("E1_equal_quiet",bait,equal,rules,2,1,1);

  auto advance=selfplay_fixture();
  advance.no_progress=rules.no_progress_limit-1;
  place_single_case_piece(advance,1,46,Marshal);
  place_single_case_piece(advance,26,51,Platoon);
  place_single_case_piece(advance,27,30,Engineer);
  const Move remove{46,51};
  print_proof("E1_advance_before_own_return",advance,remove,rules,2,0,2);
  print_proof("E1_advance_after_own_return",advance,remove,rules,3,2,2);
  print_proof("E1_node_limit",advance,remove,rules,3,0,2,0);
  auto cap=rules;
  cap.truncation_cap=1;
  print_proof("E1_experiment_cap",advance,remove,cap,3,0,2);

  auto weak=bait;
  weak.id_at[21]=weak.seat_at[21]=kNone;
  weak.rank_at[21]=Unknown;
  weak.alive[27]=false;
  auto strong=weak;
  strong.rank_at[30]=strong.rank_of[26]=Marshal;
  const auto a=seat_decision(weak), b=seat_decision(strong);
  single_case_check(same_observation(a.observation,b.observation) && a.actions==b.actions,
                    "hidden worlds changed the actor's information or ordered legal moves");
  print_proof("E2_weak_take",weak,take,rules,1,2,2);
  print_proof("E2_strong_take",strong,take,rules,1,0,0);
  print_proof("E2_weak_quiet",weak,quiet,rules,1,1,1);
  print_proof("E2_strong_quiet",strong,quiet,rules,1,1,1);
}
