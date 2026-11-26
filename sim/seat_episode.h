#pragma once
#include <optional>
#include <stdexcept>
#include "../observe/filter.h"

namespace jq {
// Only a decision boundary at the learner's turn exposes a player view.
struct SeatDecision {
  Observation observation;
  std::vector<Move> actions;
};

inline SeatDecision seat_decision(const State& state) {
  SeatDecision next{observe(state, state.to_move), {}};
  gen_moves(next.observation, state.to_move, next.actions);
  return next;
}

inline std::optional<double> team_score(Result result, int seat, MatchKind match = MatchKind::FourSeats) {
  if (!participates(match,seat) || result == Result::Ongoing || result == Result::Truncated) return std::nullopt;
  if (result == Result::Draw) return 0.5;
  return (result == Result::Team0Wins ? 0 : 1) == team_of(seat,match) ? 1.0 : 0.0;
}

// State stays in the referee. The response callback gets its own seat's view.
// If elimination removes the learner's turns, continue to the team result.
template<class Respond>
std::optional<SeatDecision> advance_to_seat(State& state, const Rules& rules,
                                            int learner, Respond&& respond) {
  if (!participates(state.match,learner)) throw std::invalid_argument("invalid learner seat");
  while (state.result == Result::Ongoing) {
    auto next = seat_decision(state);
    if (state.to_move == learner) return next;
    const Move move = respond(next.observation, next.actions);
    if (!is_legal(state, move)) throw std::runtime_error("invalid response move");
    step(state, move, rules);
  }
  return std::nullopt;
}
}
