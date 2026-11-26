#pragma once
#include "log.h"
#include "visible_terminal.h"
#include "../strategy/flag_objective.h"
#include <optional>
namespace jq {
enum class DecisionKind { Random, Heuristic, FlagRoute, CollisionHeuristic, TerminalHeuristic };

struct SimulationPolicy {
  int heuristic_seats=0;
  bool flag_route=false;
  int heuristic_version=1;
  bool records_decision() const { return flag_route || heuristic_seats; }
  DecisionKind at(int seat) const {
    if (flag_route) return DecisionKind::FlagRoute;
    if (!(heuristic_seats&(1<<seat))) return DecisionKind::Random;
    if (heuristic_version==3) return DecisionKind::TerminalHeuristic;
    return heuristic_version==2 ? DecisionKind::CollisionHeuristic : DecisionKind::Heuristic;
  }
};

inline SimulationPolicy simulation_policy(const std::string& policy) {
  if (policy == "random") return {};
  if (policy == "flag-route") return {0,true};
  const int version=policy.rfind("heuristic-3-",0)==0?3:policy.rfind("heuristic-2-",0)==0?2:1;
  const auto name=version>1?"heuristic-"+policy.substr(12):policy;
  if (name == "heuristic-ns") return {5,false,version};
  if (name == "heuristic-ew") return {10,false,version};
  if (name == "heuristic-all") return {15,false,version};
  if (name == "heuristic-south") return {1,false,version};
  if (name == "heuristic-east") return {2,false,version};
  throw std::runtime_error("unsupported policy");
}

struct DecisionScorer {
  const Observation& view;
  const Rules& rules;
  DecisionKind kind;
  std::optional<FlagObjective> objective;
  std::optional<State> visible_board;
  DecisionScorer(const Observation& o, const Rules& r, DecisionKind type):view(o),rules(r),kind(type) {
    if (kind==DecisionKind::FlagRoute) objective.emplace(view);
    if (kind==DecisionKind::TerminalHeuristic && rules.flag_eliminates)
      visible_board=visible_terminal_board(view);
  }
  MoveScore score(Move m) const {
    if (objective) return objective->score(view,m,rules);
    if (kind==DecisionKind::Random) return {};
    if (const auto result=visible_result(visible_board,m,rules)) {
      if (*result==Result::Team0Wins || *result==Result::Team1Wins) {
        const int winner=*result==Result::Team0Wins?0:1;
        return {winner==team_of(view.turn,view.match)?20000:-20000,0,0};
      }
    }
    const bool corrected=kind==DecisionKind::CollisionHeuristic || kind==DecisionKind::TerminalHeuristic;
    if (corrected && view.rank_at(m.to)==Flag &&
        public_terminal_flag(view,m,rules,occupied_enemies(view)))
      return {10000,0,0};
    return score_move(view,m,corrected?&rules:nullptr);
  }
  bool choose(Rng& rng, Move& selected) const {
    if (kind==DecisionKind::Random) return random_move(view,rng,selected);
    if (view.viewer!=view.turn) return false;
    std::vector<Move> moves;
    gen_moves(view,view.viewer,moves);
    return preferred_move(moves,rng,selected,[&](Move m){return score(m).total();});
  }
};

inline Json decision_record(const DecisionScorer& scorer, Move m) {
  const auto& o=scorer.view;
  const auto score=scorer.score(m);
  const char* name=scorer.objective ? "flag-route-1" :
      scorer.kind==DecisionKind::TerminalHeuristic ? "heuristic-3" :
      scorer.kind==DecisionKind::CollisionHeuristic ? "heuristic-2" :
      scorer.kind==DecisionKind::Heuristic ? "heuristic-1" : "random-1";
  std::ostringstream out;
  out << "{\"policy\":\"" << name
      << "\",\"viewer\":" << o.viewer << ",\"based_on_seq\":" << o.ply
      << ",\"objective\":" << score.objective << ",\"position\":" << score.position
      << ",\"exposure\":" << score.exposure << ",\"total\":" << score.total() << '}';
  return parse_json(out.str());
}
inline std::string decision_frame(const State& state, const StepInfo& event, const Json& decision,
                                  IdentityVisibility visibility, bool settlement = false, bool review = false) {
  auto frame = parse_json(frame_json(state,&event,visibility,settlement,review));
  frame.object["decision"] = decision;
  return dump_json(frame);
}
}
