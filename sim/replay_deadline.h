#pragma once
#include "exchange_counterplay.h"
#include "../strategy/flag_objective.h"

namespace jq {
struct FlagDeadline {
  bool team_win = false;
  bool uncertain_finish = false;
  int enemy_turn = kSeats;
  int unchanged_seat_turn = kSeats;
  int flag = kNone;
  int threat = kNone;
};

inline FlagDeadline road_flag_deadline(const Observation& view,const MovedSquares& moved,
                                      const Rules& rules,unsigned active,int cleared) {
  int window=0;
  for (int offset=1; offset<kSeats; ++offset) {
    const int seat=(view.viewer+offset*(rules.ccw?1:3))%kSeats;
    if (!(active&(1u<<seat))) continue;
    ++window;
    if (view.team(seat)==view.team(view.viewer)) continue;
    for (int flag=0; flag<kNodes; ++flag) {
      const int owner=view.seat_at(flag);
      if (owner<0 || view.team(owner)!=view.team(view.viewer) || view.rank_at(flag)!=Flag) continue;
      for (const int threat : Board::get().road(flag))
        if (view.seat_at(threat)==seat && publicly_mobile(view,moved,threat) && threat!=cleared)
          return {false,false,window,window,flag,threat};
    }
  }
  return {};
}

// Measure direct road threats from pieces known to be mobile. A later seat's
// deadline is conditional on the intervening seats leaving that route open.
inline FlagDeadline replay_flag_deadline(const Observation& view, const MovedSquares& moved,
                                         Move action, const Rules& rules) {
  FlagDeadline out;
  if (!rules.flag_eliminates || rules.draw_before_win) return out;
  unsigned active = 0, enemies = 0;
  for (int node=0; node<kNodes; ++node) {
    const int owner=view.seat_at(node);
    if (owner<0) continue;
    active|=1u<<owner;
    if (view.team(owner)!=view.team(view.viewer)) enemies|=1u<<owner;
  }
  const int target=view.seat_at(action.to);
  const int rank=view.rank_at(action.to);
  const auto removes=[&](int defender) {
    const auto result=resolve(view.rank_at(action.from),defender,rules);
    return result==Outcome::AttackerWins || result==Outcome::BothRemoved;
  };
  if (target>=0 && rank==Flag && removes(Flag)) {
    out.team_win=public_terminal_flag(view,action,rules,enemies);
    active&=~(1u<<target);
    if (out.team_win) return out;
  }
  const bool dark_hq=target>=0 && rank==Unknown && Board::get().node(action.to).kind==NodeKind::HQ;
  const bool clears_threat=target>=0 && (rank!=Unknown ? removes(rank) :
      publicly_mobile(view,moved,action.to) && guarantees_interception(
          view,{kNone,action.to,action,action},rules));
  const int cleared=clears_threat ? action.to : kNone;
  out=road_flag_deadline(view,moved,rules,active,cleared);
  if (dark_hq && removes(Flag)) {
    out.uncertain_finish=enemies==(1u<<target);
    const auto if_flag=road_flag_deadline(view,moved,rules,active&~(1u<<target),cleared);
    if (!out.uncertain_finish && if_flag.enemy_turn<out.enemy_turn) {
      out.enemy_turn=if_flag.enemy_turn;
      out.flag=if_flag.flag;
      out.threat=if_flag.threat;
    }
  }
  return out;
}

// Experimental ordering. Preserve frozen preference within each class and
// defer a contest between an unconfirmed final flag and an immediate defense.
inline size_t replay_deadline_choice(const std::vector<FlagDeadline>& deadlines,
                                     const std::vector<double>& preference) {
  const auto best=static_cast<size_t>(std::max_element(preference.begin(),preference.end())-preference.begin());
  std::optional<size_t> better;
  for (size_t i=0; i<deadlines.size(); ++i) {
    const auto& candidate=deadlines[i];
    const bool saves=candidate.enemy_turn>1 && deadlines[best].enemy_turn==1 &&
                     !deadlines[best].uncertain_finish && !candidate.uncertain_finish;
    if (!candidate.team_win && !saves) continue;
    if (!better || (candidate.team_win && !deadlines[*better].team_win) ||
        (candidate.team_win==deadlines[*better].team_win && preference[i]>preference[*better]))
      better=i;
  }
  return better.value_or(best);
}

inline std::optional<Move> public_flag_finish(const SeatDecision& decision, Move fallback, const Rules& rules) {
  const auto& view=decision.observation;
  const auto enemies=occupied_enemies(view);
  const auto wins=[&](Move m){return public_terminal_flag(view,m,rules,enemies);};
  if (wins(fallback)) return fallback;
  const auto choice=std::find_if(decision.actions.begin(),decision.actions.end(),wins);
  return choice==decision.actions.end()?std::nullopt:std::optional<Move>(*choice);
}

}
