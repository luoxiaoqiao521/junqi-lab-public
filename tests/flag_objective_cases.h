#pragma once
#include "single_rules_cases.h"
#include "../sim/decision.h"

namespace jq {
inline void mutual_marshal_reveal_cases() {
  const auto& board=Board::get();
  const Move clash{4,30};
  for (bool reveals:{true,false}) {
    auto state=single_case_state();
    place_single_case_piece(state,1,clash.from,Marshal);
    place_single_case_piece(state,26,clash.to,Marshal);
    for (int seat=0;seat<kSeats;++seat) {
      state.out[seat]=false;
      if (seat>=2) place_single_case_piece(state,seat*25,board.seat_node(seat,5,1),Flag);
      place_single_case_piece(state,seat*25+2,board.seat_node(seat,2,0),Engineer);
    }
    const auto initial=state_hash(state);
    Rules rules;
    rules.eliminate_exhausted_movers=true;
    rules.marshal_reveals_flag=reveals;
    State undo;
    StepInfo event{};
    single_case_check(apply_checked(state,clash,rules,undo,event) && event.outcome==Outcome::BothRemoved &&
                      event.revealed_mask==(reveals?3:0) && !event.eliminated_mask &&
                      !state.alive[1] && !state.alive[26] && state.result==Result::Ongoing && state.to_move==1,
                      "mutual marshal removal did not reveal both flags or changed the next seat");
    for (int viewer=0;viewer<kSeats;++viewer) {
      const auto visible=observe(state,viewer);
      for (int seat=0;seat<kSeats;++seat) {
        const auto flag=board.seat_node(seat,5,1);
        const bool known=seat==viewer || (reveals && seat<2);
        single_case_check(visible.rank_at(flag)==(known?Flag:Unknown),"wrong public visibility after a double reveal");
      }
    }
    restore(state,undo);
    single_case_check(state_hash(state)==initial,"undo kept a double reveal in the position");
  }
  std::cout<<"PASS mutual marshals: both flags revealed only in the enabled profile, next seat and undo\n";
}

inline void flag_objective_cases() {
  mutual_marshal_reveal_cases();
  const auto& board=Board::get();
  const int flag=board.seat_node(1,5,1), fake=board.seat_node(1,5,3);
  const Move take{uint8_t(board.seat_node(1,4,1)),uint8_t(flag)};
  Rules rules;
  rules.eliminate_exhausted_movers=true;
  auto game=single_case_state();
  place_single_case_piece(game,1,take.from,Bomb);
  place_single_case_piece(game,26,fake,Platoon);
  const auto dark=observe(game,0);
  const FlagObjective unknown(dark);
  single_case_check(is_legal(game,take) && unknown.targets_flag(flag) && unknown.targets_flag(fake),
                    "unrevealed flag should leave both occupied headquarters possible");
  auto alternate=game;
  alternate.rank_at[flag]=alternate.rank_of[25]=Platoon;
  alternate.rank_at[fake]=alternate.rank_of[26]=Flag;
  const auto swapped=observe(alternate,0);
  Rng first(7),second(7);
  Move before,after;
  single_case_check(DecisionScorer(dark,rules,DecisionKind::FlagRoute).choose(first,before) &&
                    DecisionScorer(swapped,rules,DecisionKind::FlagRoute).choose(second,after) && before==after,
                    "unrevealed identities changed the preferred move");
  game.flag_revealed[1]=true;
  const auto visible=observe(game,0);
  DecisionScorer scorer(visible,rules,DecisionKind::FlagRoute);
  single_case_check(scorer.objective->targets_flag(flag) && !scorer.objective->targets_flag(fake) &&
                    !scorer.objective->targets_flag(board.seat_node(3,5,1)) && scorer.choose(first,after) && after==take,
                    "known last flag should replace the other goals and all ordinary moves");
  const auto event=step(game,take,rules);
  single_case_check(event.outcome==Outcome::BothRemoved && game.result==Result::Team0Wins,
                    "last attacking bomb should still settle the final flag as a win");

  game=single_case_state();
  // The mine is deployed at home; use two real adjacent posts in the rear row.
  const Move contact{uint8_t(board.seat_node(1,4,1)),uint8_t(board.seat_node(1,4,0))};
  place_single_case_piece(game,1,contact.from,Marshal);
  place_single_case_piece(game,26,contact.to,Mine);
  auto full=observe(game,0,IdentityVisibility::AllIdentities);
  single_case_check(is_legal(game,contact) && FlagObjective(full).score(full,contact,rules).objective<0,
                    "known mine must not count as a lower rank defeated by a marshal");
  place_single_case_piece(game,1,contact.from,Engineer);
  full=observe(game,0,IdentityVisibility::AllIdentities);
  single_case_check(FlagObjective(full).score(full,contact,rules).objective>0,
                    "an engineer should be credited for clearing a known mine");

  game=single_case_state();
  const Move lock{uint8_t(board.seat_node(1,4,3)),uint8_t(fake)};
  place_single_case_piece(game,1,lock.from,Marshal);
  place_single_case_piece(game,26,fake,Platoon);
  place_single_case_piece(game,27,board.seat_node(1,3,0),Company);
  full=observe(game,0,IdentityVisibility::AllIdentities);
  single_case_check(is_legal(game,lock) && FlagObjective(full).score(full,lock,rules).total()<0,
                    "winning the collision must not hide that the last mover would be locked");
  const auto locked=step(game,lock,rules);
  single_case_check(locked.outcome==Outcome::AttackerWins && game.result==Result::Team1Wins &&
                    locked.exit_reason[0]==ExitReason::MovablePiecesExhausted,"false headquarters must not win");
  std::cout<<"PASS flag routes: dark headquarters, no rank leak, last bomb, mines and last-mover lock\n";
}
}
