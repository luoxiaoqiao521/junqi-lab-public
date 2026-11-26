#pragma once
#include "single_rules_cases.h"
#include "../sim/exchange_counterexamples.h"

namespace jq {
inline State sparse_south_north() {
  State s{};
  std::fill(std::begin(s.seat_at),std::end(s.seat_at),int8_t(kNone));
  std::fill(std::begin(s.rank_at),std::end(s.rank_at),int8_t(Unknown));
  std::fill(std::begin(s.id_at),std::end(s.id_at),int8_t(kNone));
  s.match=MatchKind::SouthNorth;
  s.out[1]=s.out[3]=true;
  place_single_case_piece(s,0,26,Flag);
  place_single_case_piece(s,50,86,Flag);
  return s;
}

inline void single_match_wings() {
  auto s=sparse_south_north();
  for (const auto& move : {Move{4,50},Move{0,114}}) {
    auto trial=s;
    place_single_case_piece(trial,1,move.from,Brigade);
    place_single_case_piece(trial,51,64,Engineer);
    single_case_check(is_legal(trial,move),"ordinary piece cannot reach an empty wing");
    step(trial,move,Rules{});
    single_case_check(trial.seat_at[move.to]==0 && trial.to_move==2 &&
                      trial.out[1] && trial.out[3],"wing used a nonparticipant's turn or ownership");
  }
  place_single_case_piece(s,1,4,Engineer);
  place_single_case_piece(s,51,30,Brigade);
  single_case_check(is_legal(s,{4,30}),"north piece is not attackable in a wing");
  const auto view=observe(s,0,IdentityVisibility::AllyAndPublic);
  const auto full=observe(s,0,IdentityVisibility::AllIdentities);
  std::vector<Move> actions;
  gen_moves(view,0,actions);
  single_case_check(view.rank_at(30)==Unknown && full.rank_at(30)==Brigade &&
                    std::find(actions.begin(),actions.end(),Move{4,30})!=actions.end(),
                    "single-dark view incorrectly exposes or excludes the opponent");
  step(s,{4,30},Rules{});
  single_case_check(s.seat_at[30]==2 && s.no_progress==0 && s.to_move==2,
                    "normal combat in the wing failed");
}

inline void single_match_results() {
  for (const auto seat : {0,2}) {
    auto state=sparse_south_north();
    const auto move=seat==0 ? Move{81,86} : Move{21,26};
    place_single_case_piece(state,seat*kPiecesPerSeat+1,move.from,Engineer);
    state.to_move=int8_t(seat);
    single_case_check(is_legal(state,move),"south/north flag capture considered friendly");
    const auto event=step(state,move,Rules{});
    const int opponent=2-seat;
    single_case_check(event.eliminated_mask==(1u<<opponent) &&
                      team_score(state.result,seat,state.match)==1.0 &&
                      team_score(state.result,opponent,state.match)==0.0 &&
                      !team_score(state.result,1,state.match),
                      "capture awarded the wrong result in south-north match");
  }
  auto s=sparse_south_north();
  place_single_case_piece(s,1,4,Brigade);
  const auto event=step(s,{4,30},Rules{});
  single_case_check(s.result==Result::Team0Wins && event.eliminated_mask==4 && s.out[2],
                    "immobile north did not lose");
  const auto single_hash=state_hash(s);
  s.match=MatchKind::FourSeats;
  single_case_check(single_hash!=state_hash(s) &&
                    rules_hash(Rules{})!=rules_hash(Rules{},MatchKind::SouthNorth),
                    "single and four-seat hashes collided in the fixture");
}

inline void single_match_flag_and_decoy() {
  Rules rules;
  auto state=sparse_south_north();
  place_single_case_piece(state,1,83,Brigade);
  place_single_case_piece(state,51,88,Platoon);
  const auto opening=observe(state,0);
  single_case_check(opening.rank_at(86)==Unknown && opening.rank_at(88)==Unknown &&
                    sole_enemy_hq(opening,2,rules)==kNone,
                    "two occupied opposing headquarters cannot be distinguished in the dark");
  auto immobilized=state;
  single_case_check(is_legal(immobilized,{83,88}),"decoy capture was illegal");
  const auto no_move=step(immobilized,{83,88},rules);
  single_case_check(no_move.outcome==Outcome::AttackerWins && no_move.eliminated_mask==4,
                    "flag defender with no movable piece must still be eliminated");
  place_single_case_piece(state,52,64,Engineer);
  const auto capture=step(state,{83,88},rules);
  const auto after=observe(state,0);
  single_case_check(capture.outcome==Outcome::AttackerWins && capture.eliminated_mask==0 &&
                    state.to_move==2 && state.result==Result::Ongoing && state.rank_at[86]==Flag &&
                    after.rank_at(86)==Unknown && sole_enemy_hq(after,2,rules)==86,
                    "decoy capture should preserve the north seat and let the remaining HQ be inferred");
}

inline void single_match_contract() {
  single_match_wings();
  single_match_results();
  single_match_flag_and_decoy();
  std::cout << "PASS single-dark profile: both wings, private opponents, both flag directions, no-move loss\n";
  std::cout << "PASS true/decoy HQ with and without a legal reply; same-rank masks before contact\n";
}
}
