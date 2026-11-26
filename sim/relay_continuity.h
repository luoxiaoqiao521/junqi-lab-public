#pragma once
#include <algorithm>
#include <array>
#include <functional>
#include "decision_branches.h"

namespace jq {

struct RelayEvent {
  StepInfo step;
  uint64_t after;
};

struct RelayTrace {
  State state;
  std::array<Rng,kSeats> streams;
  std::vector<RelayEvent> events;
  bool changed = false;
};

struct RelayCheckpoint {
  RelayTrace trace;
  int receiver;

  bool can_decide() const {
    const auto& s=trace.state;
    return s.result==Result::Ongoing && !s.out[receiver] && s.to_move==receiver;
  }
};

using RelayPolicy = std::function<Move(const SeatDecision&,Rng&)>;
using RelaySelection = std::function<std::optional<Move>(const SeatDecision&,Move)>;

inline void relay_apply(RelayTrace& trace,const SeatDecision& d,Move move,const Rules& rules) {
  if(std::find(d.actions.begin(),d.actions.end(),move)==d.actions.end())
    throw std::invalid_argument("relay action is outside the seat's legal set");
  auto info=step(trace.state,move,rules);
  trace.events.push_back({info,state_hash(trace.state)});
}

inline void relay_default(RelayTrace& trace,const Rules& rules,const RelayPolicy& policy) {
  const auto d=seat_decision(trace.state);
  const auto move=policy(d,trace.streams[d.observation.viewer]);
  relay_apply(trace,d,move,rules);
}

inline RelayCheckpoint relay_checkpoint(const State& root,Move preparation,int receiver,const Rules& rules,
                                        uint64_t seed,const RelayPolicy& policy) {
  if(root.result!=Result::Ongoing || !participates(root.match,receiver) || root.out[receiver] ||
     team_of(root.to_move,root.match)!=team_of(receiver,root.match))
    throw std::invalid_argument("relay requires a live receiver on the acting team");
  RelayCheckpoint checkpoint{{root,review_streams(seed),{}},receiver};
  const auto d=seat_decision(root);
  // Proposed and frozen turns consume the same baseline random opportunity.
  const auto fallback=policy(d,checkpoint.trace.streams[d.observation.viewer]);
  if(std::find(d.actions.begin(),d.actions.end(),fallback)==d.actions.end())
    throw std::invalid_argument("relay root baseline action is illegal");
  relay_apply(checkpoint.trace,d,preparation,rules);
  auto& trace=checkpoint.trace;
  while(trace.state.result==Result::Ongoing && !trace.state.out[receiver] && !checkpoint.can_decide())
    relay_default(trace,rules,policy);
  return checkpoint;
}

inline RelayTrace relay_finish(const RelayCheckpoint& checkpoint,const Rules& rules,
                               const RelayPolicy& policy,const RelaySelection& select = {}) {
  auto trace=checkpoint.trace;
  if(checkpoint.can_decide()) {
    const auto d=seat_decision(trace.state);
    const auto fallback=policy(d,trace.streams[d.observation.viewer]);
    if(std::find(d.actions.begin(),d.actions.end(),fallback)==d.actions.end())
      throw std::invalid_argument("relay fallback action is illegal");
    const auto choice=select ? select(d,fallback) : std::nullopt;
    const auto move=choice.value_or(fallback);
    trace.changed=!(move==fallback);
    relay_apply(trace,d,move,rules);
  }
  while(trace.state.result==Result::Ongoing) relay_default(trace,rules,policy);
  return trace;
}

}
