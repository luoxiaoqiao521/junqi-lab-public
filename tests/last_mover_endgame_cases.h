#pragma once
#include "selfplay_elimination_cases.h"

namespace jq {

inline void last_mover_reserve(State& state,int ally) {
  const auto& board=Board::get();
  state.out[ally]=false;
  place_single_case_piece(state,ally*kPiecesPerSeat,board.seat_node(ally,5,1),Flag);
  place_single_case_piece(state,ally*kPiecesPerSeat+1,board.seat_node(ally,0,2),Platoon);
}

inline void last_mover_flag_race(const Rules& rules) {
  auto root=selfplay_fixture();
  root.ply=rules.truncation_cap-1;
  place_single_case_piece(root,1,51,Marshal);
  place_single_case_piece(root,26,21,Engineer);
  const Move capture[]={{51,56},{21,26}};
  for (int first : {0,1}) {
    auto state=root;
    state.to_move=static_cast<int8_t>(first);
    const auto move=capture[first];
    const int defender=1-first;
    single_case_check(is_legal(state,move),"flag race fixture needs both legal choices");
    const auto result=step(state,move,rules);
    single_case_check(result.outcome==Outcome::AttackerWins &&
                      result.eliminated_mask==(1u << defender) &&
                      result.exit_reason[defender]==ExitReason::FlagRemoved &&
                      result.end_reason==EndReason::TeamEliminated &&
                      team_score(state.result,first)==1.0 && team_score(state.result,defender)==0.0 &&
                      state.ply==rules.truncation_cap && state.seat_at[move.to]==first &&
                      !can_leave_node(state.rank_at[move.to],move.to),
                      "the first flag action should win despite the cap and locked surviving attacker");
  }
  std::cout<<"PASS last-mover flag race: marshal or engineer wins by acting first; flag before cap\n";
}

inline void last_mover_mutual_matrix(const Rules& rules) {
  for (int attacker_rank : {Brigade,Bomb}) for (int allies=0;allies<4;++allies) {
    auto state=selfplay_fixture();
    if (allies & 1) last_mover_reserve(state,2);
    if (allies & 2) last_mover_reserve(state,3);
    place_single_case_piece(state,1,4,attacker_rank);
    place_single_case_piece(state,26,30,Brigade);
    single_case_check(is_legal(state,{4,30}),"last-mover mutual fixture is not legal");
    const auto result=step(state,{4,30},rules);
    const Result expected[]={Result::Truncated,Result::Team0Wins,Result::Team1Wins,Result::Ongoing};
    const EndReason reason[]={EndReason::SimultaneousExhaustion,EndReason::TeamEliminated,
                              EndReason::TeamEliminated,EndReason::None};
    single_case_check(result.outcome==Outcome::BothRemoved && result.eliminated_mask==3 &&
                      result.exit_reason[0]==ExitReason::MovablePiecesExhausted &&
                      result.exit_reason[1]==ExitReason::MovablePiecesExhausted &&
                      state.result==expected[allies] && result.end_reason==reason[allies] &&
                      !state.alive[0] && !state.alive[25],
                      "same-move retirements must respect both active partners");
    if (allies==0) single_case_check(!team_score(state.result,0),"simultaneous exhaustion must be unscored");
    if (allies==3) single_case_check(state.to_move==2,"only the surviving partners should act next");
  }
  std::cout<<"PASS last-mover mutual collisions: 8 cases with equal ranks and bomb, all surviving-partner combinations\n";
}

inline void last_mover_flag_priority(const Rules& rules) {
  for (int rank : {Marshal,Bomb}) for (int allies=0;allies<4;++allies) {
    auto state=selfplay_fixture();
    if (allies & 1) last_mover_reserve(state,2);
    if (allies & 2) last_mover_reserve(state,3);
    place_single_case_piece(state,1,51,rank);
    place_single_case_piece(state,26,30,Brigade);
    single_case_check(is_legal(state,{51,56}),"last attacker flag fixture is not legal");
    const auto result=step(state,{51,56},rules);
    const bool enemy_ally=(allies & 2)!=0;
    const Result expected[]={Result::Team0Wins,Result::Team0Wins,Result::Team1Wins,Result::Ongoing};
    single_case_check(result.outcome==(rank==Bomb?Outcome::BothRemoved:Outcome::AttackerWins) &&
                      result.exit_reason[1]==ExitReason::FlagRemoved && !state.alive[26] &&
                      result.eliminated_mask==(enemy_ally?3:2) && state.result==expected[allies] &&
                      result.exit_reason[0]==(enemy_ally?ExitReason::MovablePiecesExhausted:ExitReason::None) &&
                      state.alive[1]==(!enemy_ally && rank!=Bomb) &&
                      result.end_reason==(allies==3?EndReason::None:EndReason::TeamEliminated),
                      "flag priority must distinguish ending the enemy team from only retiring one seat");
  }
  std::cout<<"PASS last-mover flag priority: 8 cases distinguish last enemy flag from a single seat exit\n";
}

inline void last_mover_rank_and_mines(const Rules& rules) {
  for (int south_rank : {Marshal,Engineer}) {
    auto state=selfplay_fixture();
    place_single_case_piece(state,1,4,south_rank);
    place_single_case_piece(state,26,30,south_rank==Marshal?Engineer:Marshal);
    single_case_check(is_legal(state,{4,30}),"rank collision fixture is not legal");
    const auto contact=step(state,{4,30},rules);
    const int winning_seat=south_rank==Marshal?0:1;
    single_case_check(contact.outcome==(winning_seat==0?Outcome::AttackerWins:Outcome::DefenderWins) &&
                      contact.exit_reason[1-winning_seat]==ExitReason::MovablePiecesExhausted &&
                      team_score(state.result,winning_seat)==1.0,
                      "the larger rank should win legal direct contact in either role");
  }
  const auto& board=Board::get();
  const int south_bottom=board.seat_node(0,5,0), south_mine=board.seat_node(0,4,0);
  const int beside_south_mine=board.seat_node(0,4,1);
  const Move clear{static_cast<uint8_t>(beside_south_mine),static_cast<uint8_t>(south_mine)};
  const Move emerge{static_cast<uint8_t>(south_bottom),static_cast<uint8_t>(south_mine)};
  for (int rank : {Marshal,Engineer}) {
    auto state=selfplay_fixture();
    block_last_mover(state,0);
    place_single_case_piece(state,26,beside_south_mine,rank);
    std::vector<Move> trapped;
    gen_moves(StateView{state},0,trapped);
    single_case_check(trapped.empty() && state.alive[1] && can_leave_node(Platoon,south_bottom),
                      "the south last mover must be alive and intrinsically mobile while blocked");
    state.to_move=1;
    single_case_check(is_legal(state,clear),"attacking an accessible enemy mine must be legal");
    const auto result=step(state,clear,rules);
    if (rank==Marshal) {
      single_case_check(result.outcome==Outcome::DefenderWins &&
                        result.exit_reason[1]==ExitReason::MovablePiecesExhausted && state.alive[2] &&
                        team_score(state.result,0)==1.0,"a legal mine collision can cost the last mover");
    } else {
      single_case_check(result.outcome==Outcome::AttackerWins && !result.eliminated_mask &&
                        state.alive[1] && !state.alive[2] && state.alive[26] && state.to_move==0 &&
                        is_legal(state,emerge),"the engineer must free a legal action for the trapped enemy");
    }
  }
  std::cout<<"PASS last-mover ranks and mines: two rank contacts, friendly trap, legal enemy collision, engineer releases trap\n";
}

inline void last_mover_endgame_cases(const Rules& rules) {
  if (!rules.eliminate_exhausted_movers || !rules.flag_eliminates || rules.bomb_captures_flag ||
      !rules.mine_survives || rules.truncation_cap<2)
    throw std::runtime_error("last-mover fixtures are for the current selfplay contract");
  last_mover_flag_race(rules);
  last_mover_mutual_matrix(rules);
  last_mover_flag_priority(rules);
  last_mover_rank_and_mines(rules);
  exhaustion_boundaries(rules);
}
}
