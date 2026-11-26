#pragma once
#include "rank_inventory.h"
#include "regional_arrival.h"

namespace jq {
// This claim covers the first legal reply under the exhausted-mover rules.
// Rank domains are an over-approximation, so Risk may be a false positive.
enum class FlagReply { Clear, Risk, Unresolved };

struct ReplyMobility {
  bool certain=false, flag_capture=false;
};

inline unsigned friendly_flag_slots(const Observation& view) {
  unsigned known=0;
  for (int node=0; node<kNodes; ++node)
    if (view.seat_at(node)>=0 && view.rank_at(node)==Flag)
      known|=1u<<view.seat_at(node);
  return known;
}

inline bool threatens_team_flag(const Observation& view,int node,unsigned known) {
  const int owner=view.seat_at(node);
  if (owner<0 || view.team(owner)!=view.team(view.viewer)) return false;
  if (view.rank_at(node)==Flag) return true;
  const auto& square=Board::get().node(node);
  return view.rank_at(node)==Unknown && !(known&(1u<<owner)) &&
         square.kind==NodeKind::HQ && square.seat==owner;
}

inline ReplyMobility reply_mobility(Observation& view,const OpeningMarks& marks,int seat,
                                    unsigned known,const Rules& rules) {
  constexpr unsigned ordinary=all_evidence_ranks & ~((1u<<Flag)|(1u<<Mine)|(1u<<Engineer));
  ReplyMobility result;
  std::vector<Move> actions;
  for (int node=0; node<kNodes; ++node) {
    if (view.seat_at(node)!=seat) continue;
    const auto support=compatible_rank_mask(view,marks,node);
    const auto domain=support?support:all_evidence_ranks;
    const auto saved=view.ranks[node];
    unsigned movable=0;
    for (const auto& group : {std::pair<int,unsigned>{Platoon,ordinary},{Engineer,1u<<Engineer}}) {
      if (!(domain&group.second)) continue;
      view.ranks[node]=static_cast<int8_t>(group.first);
      actions.clear();
      gen_piece_moves(view,seat,node,actions);
      if (!actions.empty()) movable|=domain&group.second;
      if (view.team(seat)==view.team(view.viewer)) continue;
      for (const auto action:actions) {
        if (!threatens_team_flag(view,action.to,known)) continue;
        const auto hits=domain&group.second;
        for (int rank=Bomb; rank<RankCount; ++rank)
          if ((hits&(1u<<rank)) && resolve(rank,Flag,rules)!=Outcome::DefenderWins)
            result.flag_capture=true;
      }
    }
    view.ranks[node]=saved;
    result.certain|=domain==movable;
  }
  return result;
}

inline bool keeps_intrinsic_mover(const Observation& view,const OpeningMarks& marks,int seat) {
  constexpr unsigned immobile=(1u<<Flag)|(1u<<Mine);
  for (int node=0; node<kNodes; ++node) {
    if (view.seat_at(node)!=seat || !can_leave_node(Platoon,node)) continue;
    const auto mask=compatible_rank_mask(view,marks,node);
    if (mask && !(mask&immobile)) return true;
  }
  return false;
}

inline FlagReply projected_flag_reply(Observation view,const OpeningMarks& before,Move move,
                                      Outcome outcome,const Rules& rules,int defender) {
  auto marks=before;
  const int actor=view.viewer;
  if (outcome!=Outcome::DefenderWins) {
    view=project_quiet(view,move);
    if (outcome==Outcome::BothRemoved) {
      view.seats[move.to]=kNone;
      view.ranks[move.to]=Unknown;
    }
  } else {
    view.seats[move.from]=kNone;
    view.ranks[move.from]=Unknown;
  }
  marks.advance(move,outcome,view);
  if (!keeps_intrinsic_mover(view,marks,actor) ||
      (defender>=0 && !keeps_intrinsic_mover(view,marks,defender)))
    return FlagReply::Unresolved;
  const int quiet=outcome==Outcome::Move || (rules.reset_on_capture_only && outcome==Outcome::DefenderWins)
    ?view.quiet_clock+1:0;
  if (view.ply+1>=rules.truncation_cap || (rules.no_progress_limit>0 && quiet>=rules.no_progress_limit))
    return FlagReply::Unresolved;
  const auto known=friendly_flag_slots(view);
  for (int i=0,seat=next_seat(actor,rules); i<kSeats; ++i,seat=next_seat(seat,rules)) {
    const auto reply=reply_mobility(view,marks,seat,known,rules);
    if (reply.flag_capture) return FlagReply::Risk;
    if (reply.certain) return FlagReply::Clear;
  }
  return FlagReply::Unresolved;
}

inline FlagReply public_flag_reply(const Observation& view,const OpeningMarks& marks,
                                   Move legal,const Rules& rules) {
  if (view.match!=MatchKind::FourSeats || view.viewer!=view.turn ||
      !rules.eliminate_exhausted_movers || !rules.flag_eliminates ||
      Board::get().node(legal.to).kind==NodeKind::HQ) return FlagReply::Unresolved;
  const int defender=view.seat_at(legal.to);
  if (defender<0) return projected_flag_reply(view,marks,legal,Outcome::Move,rules,kNone);
  const auto outcomes=collision_outcome_mask(view.rank_at(legal.from),
      compatible_rank_mask(view,marks,legal.to),rules);
  if (!outcomes) return FlagReply::Unresolved;
  bool unresolved=false;
  for (auto outcome:{Outcome::AttackerWins,Outcome::DefenderWins,Outcome::BothRemoved}) {
    if (!(outcomes&(1u<<static_cast<unsigned>(outcome)))) continue;
    const auto branch=projected_flag_reply(view,marks,legal,outcome,rules,defender);
    if (branch==FlagReply::Risk) return branch;
    unresolved|=branch==FlagReply::Unresolved;
  }
  return unresolved?FlagReply::Unresolved:FlagReply::Clear;
}
}
