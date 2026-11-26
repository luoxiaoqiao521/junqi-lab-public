#pragma once

static Outcome collision_reference(int attacker,int defender,int variant) {
  constexpr const char* table[]={
    "DDDDDDDDDDDD", "WWDDLLLLLLLL", "WLDWDLLLLLLL", "WLDWWDLLLLLL", "WLDWWWDLLLLL",
    "WLDWWWWDLLLL", "WLDWWWWWDLLL", "WLDWWWWWWDLL", "WLDWWWWWWWDL", "WLDWWWWWWWWD"
  };
  char expected=table[attacker-Bomb][defender];
  if (attacker==Bomb && defender==Flag && (variant&1)) expected='W';
  if (attacker==Bomb && defender==Mine && (variant&2)) expected='L';
  if (attacker>Engineer && defender==Mine && (variant&4)) expected='D';
  if (expected=='W') return Outcome::AttackerWins;
  return expected=='L'?Outcome::DefenderWins:Outcome::BothRemoved;
}

static State collision_semantics_position(int attacker,int defender,Move action) {
  auto state=selfplay_fixture();
  state.out[3]=false;
  place_single_case_piece(state,75,116,Flag);
  place_single_case_piece(state,76,114,Platoon);
  place_single_case_piece(state,1,action.from,attacker);
  place_single_case_piece(state,2,20,Engineer);
  if (defender!=Flag) place_single_case_piece(state,26,action.to,defender);
  place_single_case_piece(state,27,34,Platoon);
  return state;
}

static int collision_semantics_case(const Rules& rules,int attacker,int defender,Outcome expected,int& selections) {
  const Move move=defender==Flag?Move{51,56}:Move{45,50};
  auto state=collision_semantics_position(attacker,defender,move);
  single_case_check(is_legal(state,move),"matrix move illegal");
  const auto visible=compare_visibility(state,IdentityVisibility::AllIdentities);
  const int objective=expected==Outcome::AttackerWins?(defender==Flag?10000:60):expected==Outcome::DefenderWins?-60:0;
  const DecisionScorer old(visible.observation,rules,DecisionKind::Heuristic);
  const DecisionScorer fixed(visible.observation,rules,DecisionKind::CollisionHeuristic);
  const auto legacy=old.score(move), repaired=fixed.score(move);
  single_case_check(repaired.objective==objective && legacy.position==repaired.position && legacy.exposure==repaired.exposure,
                    "known collision objective differs from independent reference");
  auto hidden=state;
  const auto old_hidden=seat_decision(state);
  hidden.rank_at[move.to]=hidden.rank_of[hidden.id_at[move.to]]=defender==Bomb?Mine:Bomb;
  const auto swapped=seat_decision(hidden);
  const DecisionScorer corrected_dark(old_hidden.observation,rules,DecisionKind::CollisionHeuristic);
  const DecisionScorer swapped_dark(swapped.observation,rules,DecisionKind::CollisionHeuristic);
  single_case_check(same_observation(old_hidden.observation,swapped.observation) && old_hidden.actions==swapped.actions &&
                    corrected_dark.score(move).objective==0 &&
                    corrected_dark.score(move).exposure==unknown_contact_preference(attacker) &&
                    corrected_dark.score(move).total()==swapped_dark.score(move).total(),"concealed rank changed unknown-target scoring");
  const OpeningMarks hidden_marks(old_hidden.observation,false);
  single_case_check(student_features(old_hidden,hidden_marks,move,&rules)==student_features(swapped,hidden_marks,move,&rules),
                    "concealed rank changed collision training features");
  Rng hidden_rng(81), swapped_rng(81);
  Move original_hidden{},new_hidden{};
  single_case_check(corrected_dark.choose(hidden_rng,original_hidden) && swapped_dark.choose(swapped_rng,new_hidden) &&
                    original_hidden==new_hidden,"concealed rank changed corrected policy choice");
  const OpeningMarks marks(visible.observation,false);
  const auto features=student_features(visible,marks,move,&rules);
  single_case_check(features[41]==double(expected==Outcome::AttackerWins) &&
                    features[42]==double(expected==Outcome::DefenderWins) &&
                    features[43]==double(expected==Outcome::BothRemoved),"student collision encoding differs from reference");
  Rng a(81),b(81);
  Move old_move{},new_move{};
  single_case_check(old.choose(a,old_move) && fixed.choose(b,new_move),"score did not produce a move");
  selections+=!(old_move==new_move);
  const auto event=step(state,move,rules);
  const bool surviving=expected==Outcome::AttackerWins;
  single_case_check(event.outcome==expected && state.alive[1]==surviving &&
                    state.alive[defender==Flag?25:26]==(expected==Outcome::DefenderWins),
                    "referee transition differs from independent reference");
  return legacy.objective!=repaired.objective;
}

static void verify_corrected_terminal_priority(Rules rules) {
  auto state=selfplay_fixture();
  place_single_case_piece(state,1,51,Bomb);
  place_single_case_piece(state,2,45,Engineer);
  place_single_case_piece(state,26,50,Mine);
  place_single_case_piece(state,27,34,Platoon);
  rules.bomb_captures_flag=false;
  const auto view=observe(state,0,IdentityVisibility::AllIdentities);
  const DecisionScorer fixed(view,rules,DecisionKind::CollisionHeuristic);
  Move chosen{};
  Rng rng(10);
  single_case_check(fixed.choose(rng,chosen) && chosen==Move{51,56},"mutual removal missed immediate team victory");
  const auto result=step(state,chosen,rules);
  single_case_check(result.outcome==Outcome::BothRemoved && team_score(state.result,0)==1.0,
                    "terminal bomb flag proof did not match referee");
}

static void run_collision_semantics(const Rules& profile) {
  for (int variant=0;variant<8;++variant) {
    auto rules=profile;
    rules.bomb_captures_flag=bool(variant&1);
    rules.bomb_mine_both_removed=!(variant&2);
    rules.mine_survives=!(variant&4);
    int deltas=0,selections=0;
    for (int a=Bomb;a<RankCount;++a) for (int d=Flag;d<RankCount;++d)
      deltas+=collision_semantics_case(rules,a,d,collision_reference(a,d,variant),selections);
    std::cout<<"M "<<variant<<" 120 "<<deltas<<' '<<selections<<'\n';
  }
  verify_corrected_terminal_priority(profile);
  std::cout<<"T bomb-flag-terminal-before-local-score\n";
}
