#pragma once
#include "single_rules_cases.h"
#include "../sim/replay_deadline.h"

namespace jq {
inline void public_flag_finish_cases() {
  auto initial=single_case_state();
  initial.flag_revealed[1]=true;
  place_single_case_piece(initial,1,51,Engineer);
  place_single_case_piece(initial,26,50,Engineer);
  const Move finish{51,56};
  const auto public_view=seat_decision(initial);
  const auto fallback=std::find_if(public_view.actions.begin(),public_view.actions.end(),
                                  [&](Move move){return !(move==finish);});
  single_case_check(fallback!=public_view.actions.end() && is_legal(initial,finish),"flag case actions");
  Rules rules;
  single_case_check(public_flag_finish(public_view,*fallback,rules)==finish,"visible last flag was missed");
  auto won=initial;
  step(won,finish,rules);
  single_case_check(won.result==Result::Team0Wins,"last flag did not end the team game");

  auto hidden=initial;
  hidden.flag_revealed[1]=false;
  single_case_check(!public_flag_finish(seat_decision(hidden),*fallback,rules),"hidden flag was exposed");
  auto two=initial;
  two.out[3]=false;
  place_single_case_piece(two,75,Board::get().seat_node(3,5,1),Flag);
  single_case_check(!public_flag_finish(seat_decision(two),*fallback,rules),"non-final flag was promoted");
  for (bool survives:{false,true}) {
    auto bomb=initial;
    place_single_case_piece(bomb,1,51,Bomb);
    Rules training=rules;
    training.eliminate_exhausted_movers=true;
    training.bomb_captures_flag=survives;
    const auto view=seat_decision(bomb);
    single_case_check(is_legal(bomb,finish) && public_flag_finish(view,view.actions.front(),training)==finish,
                      "bomb on last flag was not proved");
    const auto event=step(bomb,finish,training);
    single_case_check(bomb.result==Result::Team0Wins && bomb.alive[1]==survives &&
                      event.end_reason==EndReason::TeamEliminated,"bomb final flag has wrong terminal priority");
  }
  rules.draw_before_win=true;
  single_case_check(!public_flag_finish(public_view,*fallback,rules),"draw-priority rule was overridden");
  std::cout<<"PASS public last flag, hidden flag, remaining enemy and draw priority\n";
}
}
