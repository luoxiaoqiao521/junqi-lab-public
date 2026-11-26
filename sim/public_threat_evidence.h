#pragma once
#include "replay_deadline.h"
#include "rank_inventory.h"
#include <optional>
#include <utility>

namespace jq {
inline unsigned collision_casualty_mask(unsigned mask,int seat,const Observation& before,
                                        const Observation& after,const Rules& rules) {
  if (!rules.marshal_reveals_flag) return mask;
  const auto revealed=after.revealed_flags & ~before.revealed_flags;
  return mask & ((revealed&(1u<<seat)) ? 1u<<Marshal : ~(1u<<Marshal));
}

inline std::pair<unsigned,unsigned> supported_collision_ranks(unsigned att,unsigned def,
                                                               Outcome outcome,const Rules& rules) {
  unsigned supported_att=0, supported_def=0;
  for(int a=Bomb;a<RankCount;++a) if(att&(1u<<a))
    for(int b=Flag;b<RankCount;++b) if((def&(1u<<b)) && resolve(a,b,rules)==outcome) {
      supported_att|=1u<<a;
      supported_def|=1u<<b;
    }
  return {supported_att,supported_def};
}

// A complete native step makes the flag-removal retirement reason public.
inline std::optional<bool> settled_flag_capture(const Observation& before,const StepInfo& event,
                                                 const Rules& rules) {
  if (!rules.flag_eliminates ||
      (event.outcome!=Outcome::AttackerWins && event.outcome!=Outcome::BothRemoved) ||
      Board::get().node(event.move.to).kind!=NodeKind::HQ) return std::nullopt;
  const int defender=before.seat_at(event.move.to);
  expect_value(defender>=0 && defender<kSeats,"flag settlement has no defending seat");
  return event.exit_reason[defender]==ExitReason::FlagRemoved;
}

// Only the supplied observer's rank evidence and the recorded result are used.
inline void advance_contact_ranks(OpeningMarks& marks,const Observation& before,
                                  const Observation& after,Move move,Outcome outcome,const Rules& rules,
                                  std::optional<bool> flag_capture=std::nullopt) {
  if(outcome==Outcome::Move) {
    advance_public_ranks(marks,before,after,move,outcome);
    return;
  }
  const int attacker=before.seat_at(move.from), defender=before.seat_at(move.to);
  expect_value(attacker>=0 && defender>=0 && attacker!=defender,"collision evidence requires two different seats");
  if (marks.removed[attacker].size) constrain_rank_inventory(before,marks,attacker);
  if (marks.removed[defender].size) constrain_rank_inventory(before,marks,defender);
  auto att=compatible_rank_mask(before,marks,move.from) & public_route_ranks(before,move);
  auto def=compatible_rank_mask(before,marks,move.to);
  const bool attacker_lost=outcome!=Outcome::AttackerWins, defender_lost=outcome!=Outcome::DefenderWins;
  if (attacker_lost) att=collision_casualty_mask(att,attacker,before,after,rules);
  if (defender_lost) def=collision_casualty_mask(def,defender,before,after,rules);
  if (flag_capture.has_value()) def &= *flag_capture ? (1u<<Flag) : ~(1u<<Flag);
  const auto [supported_att,supported_def]=supported_collision_ranks(att,def,outcome,rules);
  expect_value(supported_att!=0,"public route and collision have no compatible identities");
  if (attacker_lost) marks.removed[attacker].add(supported_att);
  if (defender_lost) marks.removed[defender].add(supported_def);
  marks.advance(move,outcome,after);
  if(after.seat_at(move.to)>=0)
    marks.ranks[move.to]=outcome==Outcome::AttackerWins?supported_att:supported_def;
}

struct PublicThreat {
  int origins=0, forced=0, target=kNone;
};

inline int next_visible_actor(const Observation& view,const Rules& rules) {
  unsigned active=0;
  for(int n=0;n<kNodes;++n) if(view.seat_at(n)>=0) active|=1u<<view.seat_at(n);
  for(int seat=next_seat(view.viewer,rules);seat!=view.viewer;seat=next_seat(seat,rules))
    if(active&(1u<<seat)) return seat;
  return view.viewer;
}

inline PublicThreat visible_flag_threat(const Observation& view,const OpeningMarks& marks,
                                        const Rules& rules) {
  PublicThreat result;
  const int enemy=next_visible_actor(view,rules);
  if(view.team(enemy)==view.team(view.viewer)) return result;
  std::vector<int> flags;
  for(int node=0;node<kNodes;++node) {
    const int owner=view.seat_at(node);
    if(owner>=0 && view.team(owner)==view.team(view.viewer) && view.rank_at(node)==Flag) flags.push_back(node);
  }
  for(int n=0;n<kNodes;++n) {
    if(view.seat_at(n)!=enemy) continue;
    const auto ranks=compatible_rank_mask(view,marks,n);
    auto trial=view;
    unsigned reaches=0;
    for(int rank:{Platoon,Engineer}) {
      const unsigned supported=ranks & (rank==Engineer ? 1u<<Engineer :
                              all_evidence_ranks & ~((1u<<Flag)|(1u<<Mine)|(1u<<Engineer)));
      if(!supported) continue;
      trial.ranks[n]=static_cast<int8_t>(rank);
      std::vector<Move> legal;
      gen_piece_moves(trial,enemy,n,legal);
      if(std::any_of(legal.begin(),legal.end(),[&](Move m) {
        return std::find(flags.begin(),flags.end(),m.to)!=flags.end();
      })) reaches|=supported;
    }
    if(reaches) {++result.origins; result.target=n; result.forced+=ranks==reaches;}
  }
  return result;
}

inline std::optional<Move> visible_intercept(const SeatDecision& decision,const OpeningMarks& marks,
                                             const Rules& rules,Move fallback,const PublicThreat& threat) {
  if(threat.origins!=1 || !threat.forced || rules.draw_before_win || !rules.flag_eliminates)
    return std::nullopt;
  const auto& view=decision.observation;
  const auto removes=[&](Move move) {
    if(move.to!=threat.target) return false;
    const auto outcomes=collision_outcome_mask(view.rank_at(move.from),compatible_rank_mask(view,marks,move.to),rules);
    return outcomes!=0 && !(outcomes&(1u<<static_cast<unsigned>(Outcome::DefenderWins)));
  };
  if(removes(fallback) || public_flag_finish(decision,fallback,rules)) return std::nullopt;
  std::optional<Move> best;
  for(auto move:decision.actions)
    if(removes(move) && (!best || score_move(view,move).total()>score_move(view,*best).total())) best=move;
  return best;
}
}
