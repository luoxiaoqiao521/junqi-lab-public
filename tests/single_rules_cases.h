#pragma once
#include "../sim/seat_episode.h"
#include <algorithm>
#include <iostream>

namespace jq {
// Use two opposing seats in the existing four-seat topology to isolate rules
// before the actual north-south team mapping is configurable.
inline void place_single_case_piece(State& state, int id, int node, int rank) {
  state.id_at[node]=int8_t(id);
  state.seat_at[node]=int8_t(id/kPiecesPerSeat);
  state.rank_at[node]=int8_t(rank);
  state.rank_of[id]=int8_t(rank);
  state.node_of[id]=uint8_t(node);
  state.alive[id]=true;
}

inline State single_case_state() {
  State state{};
  std::fill(std::begin(state.id_at),std::end(state.id_at),int8_t(kNone));
  std::fill(std::begin(state.seat_at),std::end(state.seat_at),int8_t(kNone));
  std::fill(std::begin(state.rank_at),std::end(state.rank_at),int8_t(Unknown));
  state.out[2]=state.out[3]=true;
  state.ply=state.no_progress=69;
  place_single_case_piece(state,0,26,Flag);
  place_single_case_piece(state,25,56,Flag);
  return state;
}

inline void single_case_check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

inline void single_quiet_clock_cases() {
  struct ClockCase { int defender; Outcome outcome; };
  const ClockCase cases[]={{Unknown,Outcome::Move},{Corps,Outcome::DefenderWins},
                           {Engineer,Outcome::AttackerWins},{Brigade,Outcome::BothRemoved}};
  for (const auto& item : cases) for (bool capture_only : {false,true}) {
    auto state=single_case_state();
    place_single_case_piece(state,1,4,Brigade);
    place_single_case_piece(state,26,50,Engineer);
    if (item.defender!=Unknown) place_single_case_piece(state,27,30,item.defender);
    Rules rules;
    rules.reset_on_capture_only=capture_only;
    const Move move{4,30};
    single_case_check(is_legal(state,move),"clock fixture needs a legal action");
    const auto event=step(state,move,rules);
    const bool reset=item.outcome!=Outcome::Move &&
                     (!capture_only || item.outcome!=Outcome::DefenderWins);
    single_case_check(event.outcome==item.outcome && state.ply==70 &&
                      state.no_progress==(reset ? 0 : 70) &&
                      state.result==(reset ? Result::Ongoing : Result::Draw),
                      "quiet clock/reset does not match the rule mode");
  }
  std::cout << "PASS 69-ply clock: quiet and three collision classes, two reset modes\n";
}

inline void single_terminal_cases() {
  for (bool draw_first : {false,true}) {
    Rules rules;
    rules.truncation_cap=70;
    rules.draw_before_win=draw_first;
    auto state=single_case_state();
    place_single_case_piece(state,1,4,Brigade);
    State undo;
    StepInfo event{};
    single_case_check(!apply_checked(state,{26,25},rules,undo,event) &&
                      state.no_progress==69 && state.ply==69,
                      "rejected action advanced the clock");
    single_case_check(apply_checked(state,{4,30},rules,undo,event) && state.out[1] &&
                      state.result==(draw_first ? Result::Draw : Result::Team0Wins),
                      "simultaneous no-legal-move/draw priority or cap");
    state=single_case_state();
    place_single_case_piece(state,1,51,Engineer);
    place_single_case_piece(state,26,50,Engineer);
    single_case_check(apply_checked(state,{51,56},rules,undo,event) &&
                      state.no_progress==0 && state.result==Result::Team0Wins &&
                      team_score(state.result,0)==1.0 && team_score(state.result,1)==0.0,
                      "capturing a flag at the external cap must remain a known win");
    state=single_case_state();
    place_single_case_piece(state,1,4,Engineer);
    place_single_case_piece(state,26,50,Engineer);
    rules.no_progress_limit=0;
    single_case_check(apply_checked(state,{4,30},rules,undo,event) &&
                      state.result==Result::Truncated && !team_score(state.result,0),
                      "external cutoff acquired a game score");
  }
  std::cout << "PASS terminal priority, invalid move, flag and external cutoff\n";
}

inline void single_wing_and_identity_cases() {
  auto state=single_case_state();
  place_single_case_piece(state,1,4,Brigade);
  single_case_check(is_legal(state,{4,30}) && is_legal(state,{4,50}) && !is_legal(state,{4,34}),
                    "ordinary piece empty wing path");
  auto blocked=state;
  place_single_case_piece(blocked,2,40,Mine);
  single_case_check(is_legal(blocked,{4,30}) && !is_legal(blocked,{4,50}),"rail blocker ignored");
  place_single_case_piece(state,1,4,Engineer);
  single_case_check(is_legal(state,{4,34}),"engineer wing corner missing");
  blocked=state;
  int id=2;
  for (const int node : Board::get().rail_adj(4)) place_single_case_piece(blocked,id++,node,Mine);
  single_case_check(!is_legal(blocked,{4,34}),"engineer jumped over every rail entry blocker");
  const auto original=state_hash(state);
  blocked=state; ++blocked.no_progress;
  single_case_check(original!=state_hash(blocked),"clock not represented in state hash");
  blocked=state; blocked.to_move=1;
  single_case_check(original!=state_hash(blocked),"turn not represented in state hash");
  blocked=state; blocked.out[2]=false;
  single_case_check(original!=state_hash(blocked),"active seats not represented in state hash");
  Rules other;
  other.reset_on_capture_only=true;
  single_case_check(rules_hash(Rules{})!=rules_hash(other),"draw mode not represented in rule hash");
  place_single_case_piece(state,50,64,Engineer);
  single_case_check(team_of(0)==team_of(2) && !can_enter(StateView{state},0,64),
                    "legacy seat pairing is part of this test contract");
  std::cout << "PASS empty-wing travel and blockers, state identity, north-south team dependency\n";
}

inline void single_dark_rule_contract() {
  single_quiet_clock_cases();
  single_terminal_cases();
  single_wing_and_identity_cases();
}
}
