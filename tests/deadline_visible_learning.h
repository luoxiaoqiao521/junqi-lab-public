#pragma once

static SeatDecision compare_visibility(const State& state,IdentityVisibility visibility) {
  SeatDecision decision{observe(state,state.to_move,visibility),{}};
  gen_moves(decision.observation,state.to_move,decision.actions);
  return decision;
}

static double exact_opening_payoff(State state,Move move,const Rules& rules) {
  const int viewer=state.to_move;
  single_case_check(is_legal(state,move),"visibility gate fixture has illegal action");
  step(state,move,rules);
  const auto payoff=team_score(state.result,viewer);
  single_case_check(payoff.has_value(),"visibility gate fixture did not settle");
  return *payoff;
}

static void audit_visible_learning(const Rules& rules,const StudentNetwork& student) {
  auto base=selfplay_fixture();
  base.no_progress=rules.no_progress_limit-1;
  place_single_case_piece(base,1,4,Division);
  place_single_case_piece(base,26,30,Brigade);
  for (int rotation=0;rotation<kSeats;++rotation) {
    const auto lower=rotate_panel(base,rotation);
    auto upper=lower;
    const auto node=[&](int n) { return static_cast<uint8_t>(rotate_panel_node(n,rotation)); };
    const Move exchange{node(4),node(30)}, secure_draw{node(4),node(3)};
    upper.rank_at[exchange.to]=upper.rank_of[upper.id_at[exchange.to]]=Corps;
    const auto dark_a=seat_decision(lower), dark_b=seat_decision(upper);
    const auto full_a=compare_visibility(lower,IdentityVisibility::AllIdentities);
    const auto full_b=compare_visibility(upper,IdentityVisibility::AllIdentities);
    single_case_check(same_observation(dark_a.observation,dark_b.observation) &&
                      dark_a.actions==dark_b.actions && dark_a.actions==full_a.actions &&
                      full_a.actions==full_b.actions,"visibility changed the ordered legal candidates");
    const OpeningMarks dark(dark_a.observation,false), a(full_a.observation,false), b(full_b.observation,false);
    for (const auto action:full_a.actions) {
      const auto x=student_features(full_a,a,action), y=student_features(full_b,b,action);
      single_case_check(short_features(full_a.observation,action)==short_features(full_b.observation,action) &&
                        x==y && student.score(x)==student.score(y),"registered 41-input alias changed");
      single_case_check(student_features(dark_a,dark,action,&rules)==student_features(dark_b,dark,action,&rules),
                        "44-input test exposed a dark identity");
    }
    const auto low=student_features(full_a,a,exchange,&rules);
    const auto high=student_features(full_b,b,exchange,&rules);
    single_case_check(low[41]==1. && low[42]==0. && high[41]==0. && high[42]==1.,
                      "existing collision inputs did not encode the visible difference");
    const auto open_low=exact_opening_payoff(lower,exchange,rules),open_high=exact_opening_payoff(upper,exchange,rules);
    const auto draw_low=exact_opening_payoff(lower,secure_draw,rules),draw_high=exact_opening_payoff(upper,secure_draw,rules);
    single_case_check(open_low==1. && open_high==0. && draw_low==.5 && draw_high==.5,"wrong preference control");
    const PublicReplyHistory history(lower,false);
    history.for_actor(dark_a.observation,dark);
    bool rejected=false;
    try { history.for_actor(full_a.observation,a); }
    catch (const std::runtime_error&) { rejected=true; }
    single_case_check(rejected,"four-dark reply history accepted full identity input");
    std::cout<<rotation<<' '<<full_a.actions.size()<<' '<<open_low<<' '<<open_high
             <<' '<<draw_low<<' '<<draw_high<<'\n';
  }
}
