#pragma once

// Fixed-deadline task: did seat 1 lose its flag by capture in the allotted turns?
static EvaluatorBounds relay_capture(const State& state,const Rules& rules,bool decline,int depth,int& nodes) {
  if (!depth || state.result!=Result::Ongoing) return {0,0};
  if (!nodes) return {};
  --nodes;
  std::vector<Move> actions;
  legal_moves(state,actions);
  if (actions.empty()) return {};
  const Move retreat{57,58};
  if (decline && state.to_move==2 && std::find(actions.begin(),actions.end(),retreat)!=actions.end())
    actions={retreat};
  const bool friendly=team_of(state.to_move)==0;
  EvaluatorBounds result=friendly?EvaluatorBounds{0,0}:EvaluatorBounds{2,2};
  for (const auto action:actions) {
    auto next=state;
    const auto event=step(next,action,rules);
    const auto branch=event.exit_reason[1]==ExitReason::FlagRemoved ? EvaluatorBounds{2,2} :
                      relay_capture(next,rules,decline,depth-1,nodes);
    if (result.include(branch,friendly)) break;
  }
  return result;
}

static State relay_fixture(bool reinforcement,bool reserve=false) {
  auto state=selfplay_fixture();
  state.out[2]=state.out[3]=false;
  place_single_case_piece(state,1,50,Engineer);
  place_single_case_piece(state,26,51,Mine);
  place_single_case_piece(state,27,52,Bomb);
  place_single_case_piece(state,28,34,Platoon);
  place_single_case_piece(state,50,86,Flag);
  place_single_case_piece(state,51,57,Engineer);
  if (reserve) place_single_case_piece(state,52,67,Platoon);
  place_single_case_piece(state,75,116,Flag);
  place_single_case_piece(state,76,reinforcement?53:114,Marshal);
  return state;
}

static StepInfo relay_play(State& state,const Rules& rules,int seat,Move action) {
  single_case_check(state.to_move==seat && is_legal(state,action),"relay witness is not a legal turn");
  return step(state,action,rules);
}

static void run_team_relay(const Rules& rules) {
  for (int condition:{0,1,2}) {
    auto state=relay_fixture(condition==2);
    const Move clear{50,51};
    single_case_check(relay_play(state,rules,0,clear).outcome==Outcome::AttackerWins,"engineer did not clear the mine");
    int budget=40000;
    const auto proof=relay_capture(state,rules,condition==1,4,budget);
    const int expected=condition==0?2:0;
    single_case_check(proof.lower==expected && proof.upper==expected,"relay proof contradicts the registered control");
    std::cout<<"T "<<condition<<' '<<proof.lower<<' '<<proof.upper<<' '<<40000-budget<<'\n';
  }

  auto root=relay_fixture(false);
  relay_play(root,rules,0,{50,51});
  auto bomb_front=root;
  relay_play(bomb_front,rules,1,{52,51});
  auto decline=bomb_front;
  const Move take{57,56};
  const auto full=compare_visibility(bomb_front,IdentityVisibility::AllIdentities);
  const auto dual=compare_visibility(bomb_front,IdentityVisibility::AllyAndPublic);
  const auto dark=compare_visibility(bomb_front,IdentityVisibility::OwnAndPublic);
  Rng rng(42);
  single_case_check(full.actions==dual.actions && dual.actions==dark.actions &&
                    full.observation.rank_at(56)==Flag && dual.observation.rank_at(56)==Unknown &&
                    dark.observation.rank_at(56)==Unknown && !(route_baseline(bomb_front,rules,rng)==take),
                    "last-mover relay must remain legal and expose the exhaustion tradeoff to the policy");
  const auto finish=relay_play(bomb_front,rules,2,take);
  single_case_check(finish.exit_reason[1]==ExitReason::FlagRemoved && team_score(bomb_front.result,2)==0.0,
                    "first flag must not hide immediate team defeat when the last mover is locked");
  relay_play(decline,rules,2,{57,58});
  single_case_check(decline.alive[25],"declining relay unexpectedly took the flag");

  auto resourced=relay_fixture(false,true);
  relay_play(resourced,rules,0,{50,51});
  relay_play(resourced,rules,1,{52,51});
  Rng reserve_rng(42);
  single_case_check(route_baseline(resourced,rules,reserve_rng)==take &&
                    relay_play(resourced,rules,2,take).exit_reason[1]==ExitReason::FlagRemoved &&
                    resourced.result==Result::Ongoing,"extra surviving mover should allow the current policy to convert the flag");

  auto reinforced=relay_fixture(true);
  relay_play(reinforced,rules,0,{50,51});
  relay_play(reinforced,rules,1,{52,57});
  single_case_check(reinforced.out[2],"removed second attacker should be skipped");
  relay_play(reinforced,rules,3,{53,51});
  single_case_check(reinforced.out[0] && reinforced.alive[25],"enemy reinforcement failed to prevent the relay");
  std::cout<<"V execute decline intercept exhaustion shared-legals\n";
}
