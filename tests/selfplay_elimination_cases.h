#pragma once
#include "single_rules_cases.h"
#include "../sim/opening_train.h"

namespace jq {
inline State selfplay_fixture() {
  auto state=single_case_state();
  state.ply=state.no_progress=0;
  return state;
}

inline void block_last_mover(State& state,int seat) {
  const auto& b=Board::get();
  place_single_case_piece(state,seat*25+1,b.seat_node(seat,5,0),Platoon);
  place_single_case_piece(state,seat*25+2,b.seat_node(seat,4,0),Mine);
}

inline void temporary_block_can_clear(const Rules& rules) {
  const auto& b=Board::get();
  auto s=selfplay_fixture();
  block_last_mover(s,1);
  const auto at=[&](int r,int c){return static_cast<uint8_t>(b.seat_node(1,r,c));};
  place_single_case_piece(s,1,at(4,2),Engineer);
  const Move prepare{at(4,2),at(4,1)},open{at(4,1),at(4,0)};
  single_case_check(is_legal(s,prepare),"temporary block fixture invalid");
  auto archive=s;
  const auto old=step(archive,prepare,Rules{});
  single_case_check(archive.out[1] && old.eliminated_mask==2,"old rule comparison did not eliminate blocked seat");
  const auto skipped=step(s,prepare,rules);
  single_case_check(!s.out[1] && s.to_move==0 && s.ply==1 && s.no_progress==1 &&
                    skipped.eliminated_mask==0 && skipped.skipped_mask==2 &&
                    skipped.exit_reason[1]==ExitReason::None && s.alive[25] && s.alive[26],
                    "blocked mover was removed or a skipped turn incremented the clock");
  single_case_check(is_legal(s,open),"opening the block must be a legal engineer move");
  const auto restored=step(s,open,rules);
  single_case_check(!s.out[1] && s.to_move==1 && !restored.eliminated_mask && s.result==Result::Ongoing,
                    "freed mover did not regain a turn");
}

inline void last_mover_and_team_result(const Rules& rules) {
  const auto& b=Board::get();
  auto s=selfplay_fixture();
  place_single_case_piece(s,1,b.seat_node(0,4,3),Bomb);
  place_single_case_piece(s,2,b.seat_node(0,4,0),Mine);
  place_single_case_piece(s,26,b.seat_node(1,0,0),Brigade);
  s.out[2]=false;
  place_single_case_piece(s,50,b.seat_node(2,5,1),Flag);
  place_single_case_piece(s,51,b.seat_node(1,4,1),Engineer);
  const Move home{23,28},opponent{30,31},capture{51,56};
  single_case_check(is_legal(s,home),"last movable piece must be able to enter headquarters");
  const auto retired=step(s,home,rules);
  single_case_check(retired.eliminated_mask==1 && retired.exit_reason[0]==ExitReason::MovablePiecesExhausted &&
                    s.out[0] && !s.alive[0] && !s.alive[1] &&
                    !s.alive[2] && s.id_at[28]==kNone && s.result==Result::Ongoing &&
                    !team_score(s.result,0),"retirement did not clear the seat or ended the team early");
  single_case_check(is_legal(s,opponent),"opponent did not receive its scheduled turn");
  step(s,opponent,rules);
  single_case_check(s.to_move==2 && is_legal(s,capture),"partner did not receive its flag move");
  const auto win=step(s,capture,rules);
  single_case_check(win.eliminated_mask==2 && win.exit_reason[1]==ExitReason::FlagRemoved &&
                    win.end_reason==EndReason::TeamEliminated && s.result==Result::Team0Wins,
                    "retired learner failed to receive the later team win");
  OpeningEpisode trace;
  trace.learner_out=true;
  trace.decisions=1;
  trace.gradient[0]=2.0;
  trace.score=team_score(s.result,0);
  ShortWeights update{};
  single_case_check(add_team_return(update,trace) && update[0]==2.0,
                    "terminal team reward did not reach the retired learner's gradient");
  trace.score=team_score(Result::Truncated,0);
  single_case_check(!add_team_return(update,trace) && update[0]==2.0,"cutoff changed the training gradient");
}

inline void team_exits_without_two_flags(const Rules& rules, bool mixed) {
  auto s=selfplay_fixture();
  place_single_case_piece(s,1,4,Marshal);
  place_single_case_piece(s,26,mixed?50:30,Engineer);
  if (mixed) place_single_case_piece(s,2,51,Engineer);
  s.out[3]=false;
  place_single_case_piece(s,75,Board::get().seat_node(3,5,1),Flag);
  const Move first=mixed?Move{51,56}:Move{4,30};
  for (int from=0;from<kNodes;++from) {
    if (s.seat_at[from]>=0 || from==first.to) continue;
    auto sample=s;
    place_single_case_piece(sample,76,from,Engineer);
    if (!is_legal(sample,first)) continue;
    const auto departed=step(sample,first,rules);
    if (sample.to_move!=3 || sample.result!=Result::Ongoing) continue;
    const Move last{static_cast<uint8_t>(from),static_cast<uint8_t>(mixed?4:30)};
    if (!is_legal(sample,last)) continue;
    single_case_check(departed.exit_reason[1]==(mixed?ExitReason::FlagRemoved:ExitReason::MovablePiecesExhausted) &&
                      departed.eliminated_mask==2 && sample.alive[75],
                      "first exit event did not retain its cause before the final enemy acted");
    const auto won=step(sample,last,rules);
    single_case_check(won.exit_reason[3]==ExitReason::MovablePiecesExhausted &&
                      won.eliminated_mask==8 && won.end_reason==EndReason::TeamEliminated &&
                      sample.out[1] && sample.out[3] && team_score(sample.result,0)==1.0,
                      "losing the last mover with the flag alive did not settle the team");
    return;
  }
  single_case_check(false,"missing legal double exhaustion or mixed-win fixture");
}

inline void exhaustion_boundaries(const Rules& rules) {
  auto both=selfplay_fixture();
  place_single_case_piece(both,1,4,Brigade);
  place_single_case_piece(both,26,30,Brigade);
  single_case_check(is_legal(both,{4,30}),"mutual last-mover fixture invalid");
  const auto mutual=step(both,{4,30},rules);
  single_case_check(mutual.eliminated_mask==3 && !team_score(both.result,0) &&
                    mutual.exit_reason[0]==ExitReason::MovablePiecesExhausted &&
                    mutual.exit_reason[1]==ExitReason::MovablePiecesExhausted &&
                    mutual.end_reason==EndReason::SimultaneousExhaustion &&
                    both.result==Result::Truncated,"simultaneous exhaustion invented a winner");

  auto stuck=selfplay_fixture();
  block_last_mover(stuck,0);
  block_last_mover(stuck,1);
  place_single_case_piece(stuck,3,23,Bomb);
  single_case_check(is_legal(stuck,{23,28}),"full round with no moves fixture invalid");
  const auto event=step(stuck,{23,28},rules);
  single_case_check(stuck.result==Result::Truncated && event.end_reason==EndReason::NoLegalActions &&
                    event.skipped_mask==3 && !event.eliminated_mask &&
                    !stuck.out[0] && !stuck.out[1] && !team_score(stuck.result,0),
                    "complete immobilization must stop without making up an elimination");

  auto flag=selfplay_fixture();
  place_single_case_piece(flag,1,51,Bomb);
  place_single_case_piece(flag,26,30,Brigade);
  single_case_check(is_legal(flag,{51,56}),"flag priority fixture invalid");
  const auto captured=step(flag,{51,56},rules);
  single_case_check(captured.eliminated_mask==2 && captured.exit_reason[1]==ExitReason::FlagRemoved &&
                    captured.end_reason==EndReason::TeamEliminated && !flag.alive[26] && team_score(flag.result,0)==1.,
                    "last flag capture lost its terminal precedence");
}

inline void selfplay_elimination_cases() {
  Rules rules;
  rules.eliminate_exhausted_movers=true;
  temporary_block_can_clear(rules);
  last_mover_and_team_result(rules);
  team_exits_without_two_flags(rules,false);
  team_exits_without_two_flags(rules,true);
  exhaustion_boundaries(rules);
  Rules archive;
  archive.check_involved_no_move=true;
  single_case_check(rules_hash(rules)!=rules_hash(Rules{}) && rules_hash(rules)!=rules_hash(archive),
                    "training rules identity was reused");
  std::cout<<"PASS selfplay temporary blockage, permanent retirement, flag priority and team score\n";
}
}
