#pragma once
#include "heuristic.h"
#include "../core/rules.h"
#include <array>

namespace jq {
inline unsigned occupied_enemies(const Observation& view) {
  unsigned result=0;
  for (int seat : view.seats)
    if (seat>=0 && view.team(seat)!=view.team(view.viewer)) result|=1u<<seat;
  return result;
}

// The caller supplies a legal action. Once the last opposing flag is removed,
// the referee settles the match before retiring an exhausted attacker.
inline bool public_terminal_flag(const Observation& view, Move action, const Rules& rules, unsigned enemies) {
  const int target=view.seat_at(action.to);
  if (!rules.flag_eliminates || rules.draw_before_win || target<0 || enemies!=(1u<<target) ||
      view.rank_at(action.to)!=Flag) return false;
  const auto outcome=resolve(view.rank_at(action.from),Flag,rules);
  return outcome==Outcome::AttackerWins || outcome==Outcome::BothRemoved;
}

// Road hops are a positional hint. Legal first moves still come from the
// referee's move generator, with railway, occupancy and camp constraints.
class FlagObjective {
  std::array<int,kNodes> hops_;
  unsigned enemies_;
  int movers_=0;
  bool ally_alive_=false;
public:
  explicit FlagObjective(const Observation& view):enemies_(occupied_enemies(view)) {
    const auto& board=Board::get();
    std::array<uint8_t,kNodes> queue{};
    int head=0,tail=0;
    hops_.fill(kNodes);
    for (int n=0;n<kNodes;++n) {
      const int owner=view.seat_at(n);
      if (owner==view.viewer) movers_+=can_leave_node(view.rank_at(n),n);
      else if (owner>=0 && view.team(owner)==view.team(view.viewer)) ally_alive_=true;
      if (owner<0 || !(enemies_&(1u<<owner)) || board.node(n).kind!=NodeKind::HQ || board.node(n).seat!=owner)
        continue;
      const int rank=view.rank_at(n);
      if (rank!=Flag && rank!=Unknown) continue;
      if (rank==Unknown && (view.rank_at(board.seat_node(owner,5,1))==Flag ||
                            view.rank_at(board.seat_node(owner,5,3))==Flag)) continue;
      hops_[n]=0;
      queue[tail++]=static_cast<uint8_t>(n);
    }
    while (head<tail) {
      const int node=queue[head++];
      for (int next:board.road(node)) if (hops_[next]==kNodes && board.node(next).kind!=NodeKind::HQ) {
        hops_[next]=hops_[node]+1;
        queue[tail++]=static_cast<uint8_t>(next);
      }
    }
  }

  bool targets_flag(int node) const {return hops_[node]==0;}

  MoveScore score(const Observation& view, Move move, const Rules& rules) const {
    if (public_terminal_flag(view,move,rules,enemies_)) return {10000,0,0};
    const auto& board=Board::get();
    const int attacker=view.rank_at(move.from), target=view.rank_at(move.to);
    const bool contact=view.seat_at(move.to)>=0;
    const bool locked=board.node(move.to).kind==NodeKind::HQ;
    const auto outcome=contact && target!=Unknown ? resolve(attacker,target,rules):Outcome::Move;
    MoveScore result;
    if (target==Flag) result.objective=120;
    else if (outcome==Outcome::AttackerWins) result.objective=24;
    else if (outcome==Outcome::BothRemoved) result.objective=4;
    else if (outcome==Outcome::DefenderWins) result.objective=-80;
    if (contact && target==Unknown) result.exposure=unknown_contact_preference(attacker);
    if (locked) result.position=targets_flag(move.to) ? -4 : -24;
    else if (outcome==Outcome::Move || outcome==Outcome::AttackerWins)
      result.position=3*(hops_[move.from]-hops_[move.to]);
    const bool consumed=locked || outcome==Outcome::DefenderWins || outcome==Outcome::BothRemoved;
    const bool possible_final=locked && target==Unknown && targets_flag(move.to) &&
                              enemies_==(1u<<view.seat_at(move.to));
    if (rules.eliminate_exhausted_movers && movers_==1 && !ally_alive_ && consumed && !possible_final)
      result.objective-=500;
    return result;
  }
};
}
