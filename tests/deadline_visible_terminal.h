#pragma once

static void verify_visible_terminal(const State& state, const Rules& rules, int (&checks)[5]) {
  const auto full=compare_visibility(state,IdentityVisibility::AllIdentities);
  const auto reconstructed=visible_terminal_board(full.observation);
  single_case_check(reconstructed.has_value(),"full observation did not restore a position");
  std::vector<Move> restored_actions;
  legal_moves(*reconstructed,restored_actions);
  single_case_check(restored_actions==full.actions,"full observation changed legal moves");
  for (const Move move:full.actions) {
    auto actual=state;
    step(actual,move,rules);
    ++checks[static_cast<int>(actual.result)];
    single_case_check(visible_result(reconstructed,move,rules)==actual.result,
                      "full observation changed team settlement");
  }
}

static void run_visible_terminal(const Rules& rules,uint64_t seed) {
  int controls[5]{};
  auto root=relay_fixture(false);
  relay_play(root,rules,0,{50,51});
  relay_play(root,rules,1,{52,51});
  const Move take{57,56}, quiet{57,52};
  for (int condition=0;condition<5;++condition) {
    auto state=root;
    if (condition==1) place_single_case_piece(state,52,67,Platoon);
    if (condition==2) {
      place_single_case_piece(state,52,85,Platoon);
      place_single_case_piece(state,53,80,Mine);
      std::vector<Move> available;
      gen_moves(StateView{state},2,available);
      single_case_check(std::none_of(available.begin(),available.end(),[](Move m){return m.from==85;}),
                        "reserve must be blocked but able to move out of its station");
    }
    if (condition==3) place_single_case_piece(state,52,88,Platoon);
    if (condition==4) {
      StepInfo exit{};
      recorded_retirement(state,3,exit,ExitReason::FlagRemoved);
    }
    verify_visible_terminal(state,rules,controls);
    const auto full=compare_visibility(state,IdentityVisibility::AllIdentities);
    DecisionScorer old(full.observation,rules,DecisionKind::CollisionHeuristic);
    DecisionScorer corrected(full.observation,rules,DecisionKind::TerminalHeuristic);
    auto next=state;
    relay_play(next,rules,2,take);
    const auto expected=condition==4?Result::Team0Wins:
        (condition==0 || condition==3?Result::Team1Wins:Result::Ongoing);
    single_case_check(next.result==expected,"unexpected result for lock, blockage, or last enemy");
    Move prior{}, chosen{};
    Rng old_rng(42), new_rng(42);
    single_case_check(old.choose(old_rng,prior) && corrected.choose(new_rng,chosen),"relay choice missing");
    single_case_check(prior==take && (chosen==take)==(expected!=Result::Team1Wins),
                      "terminal distinction did not control the real choice");
    for (const auto mode:{IdentityVisibility::OwnAndPublic,IdentityVisibility::AllyAndPublic}) {
      const auto masked=compare_visibility(state,mode);
      DecisionScorer restricted(masked.observation,rules,DecisionKind::TerminalHeuristic);
      DecisionScorer before(masked.observation,rules,DecisionKind::CollisionHeuristic);
      single_case_check(!restricted.visible_board,"reconstruction accepted a hidden piece");
      for (const Move move:masked.actions)
        single_case_check(restricted.score(move).total()==before.score(move).total(),"hidden piece affected v3 score");
    }
    std::cout<<"C "<<condition<<' '<<static_cast<int>(expected)<<' '<<prior.from+0<<' '<<prior.to+0
             <<' '<<chosen.from+0<<' '<<chosen.to+0<<'\n';
  }
  auto draw=rules, cap=rules, alternate=rules;
  draw.no_progress_limit=root.no_progress+1;
  cap.truncation_cap=root.ply+1;
  alternate.flag_eliminates=false;
  const auto observed=compare_visibility(root,IdentityVisibility::AllIdentities);
  verify_visible_terminal(root,draw,controls);
  verify_visible_terminal(root,cap,controls);
  const auto restored=visible_terminal_board(observed.observation);
  single_case_check(visible_result(restored,quiet,draw)==Result::Draw &&
                    visible_result(restored,quiet,cap)==Result::Truncated,
                    "rule draw was conflated with an experiment cutoff");
  single_case_check(!DecisionScorer(observed.observation,alternate,DecisionKind::TerminalHeuristic).visible_board,
                    "unverified settlement profile accepted");
  std::cout<<"B";
  for (const int count:controls) std::cout<<' '<<count;
  std::cout<<'\n';

  int generated[5]{}, samples=0;
  for (const auto match:{MatchKind::FourSeats,MatchKind::SouthNorth}) {
    for (int i=0;i<12;++i) {
      Rng layout_rng(seed+i), play_rng(seed+901+i);
      int8_t layouts[kSeats][kPiecesPerSeat];
      for (auto& layout:layouts) generate_layout(layout,layout_rng);
      State state;
      init_state(state,layouts,0,rules,match);
      std::vector<Move> moves;
      while (state.result==Result::Ongoing && state.ply<180) {
        if (state.ply%7==0) {
          verify_visible_terminal(state,rules,generated);
          ++samples;
        }
        legal_moves(state,moves);
        single_case_check(!moves.empty(),"generated decision has no legal move");
        step(state,moves[play_rng.below(static_cast<uint32_t>(moves.size()))],rules);
      }
    }
  }
  std::cout<<"G "<<samples;
  for (const int count:generated) std::cout<<' '<<count;
  std::cout<<'\n';
}
