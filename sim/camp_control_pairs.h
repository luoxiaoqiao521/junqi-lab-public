#pragma once
#include "camp_control.h"

namespace jq {
inline int next_tracked_node(int node, int owner, const StepInfo& event, const Observation& after) {
  if (node==kNone) return kNone;
  if (node==event.move.from) node=event.move.to;
  return after.seat_at(node)==owner ? node : kNone;
}

inline bool public_flag_exposure(const Observation& view, const MovedSquares& moved, int actor) {
  if (team_of(actor)==team_of(view.viewer)) return false;
  for (int flag=0; flag<kNodes; ++flag) {
    if (view.rank_at(flag)!=Flag || team_of(view.seat_at(flag))!=team_of(view.viewer)) continue;
    for (const auto from : Board::get().road(flag))
      if (view.seat_at(from)==actor && publicly_mobile(view,moved,from)) return true;
  }
  return false;
}

inline std::optional<Move> toward_visible_flag(const SeatDecision& decision) {
  const auto& view=decision.observation;
  int best=0;
  std::optional<Move> choice;
  for (int flag=0; flag<kNodes; ++flag) {
    if (view.rank_at(flag)!=Flag || team_of(view.seat_at(flag))!=team_of(view.viewer)) continue;
    for (const auto move : decision.actions) {
      if (view.seat_at(move.to)!=kNone) continue;
      const auto gain=distance(move.from,flag)-distance(move.to,flag);
      if (gain>best) { best=gain; choice=move; }
    }
  }
  return choice;
}

struct CampContinuation {
  int score=-1, next_group=0;
  bool escape=false, flag_exposure=false, ally_turn=false, ally_contact=false, target_turn=false;
  int pursue_score=-1, switch_score=-1;
};

inline void compare_camp_revisit(CampContinuation& result, const State& state, int moved_piece,
                                int target, bool changed, const Rules& rules, int response,
                                const ShortWeights& weights, uint64_t seed) {
  const auto d=seat_decision(state);
  const auto chase=std::find_if(d.actions.begin(),d.actions.end(),[=](Move action) {
    return action.from==moved_piece && action.to==target;
  });
  const auto defense=toward_visible_flag(d);
  result.next_group=1;
  if (chase==d.actions.end() || !defense) return;
  result.next_group=changed ? 3 : 2;
  const auto a=review_rollout(state,*chase,rules,response,true,weights,seed);
  const auto b=review_rollout(state,*defense,rules,response,true,weights,seed);
  result.pursue_score=review_outcome(a,state.to_move);
  result.switch_score=review_outcome(b,state.to_move);
}

inline CampContinuation camp_continue(State state, const CampPoint& point, Move first,
                                       const Rules& rules, int response, const ShortWeights& weights,
                                       uint64_t seed) {
  CampContinuation result;
  const int learner=state.to_move;
  const int ally=(learner+2)%4;
  const auto before=observe(state,learner);
  const int target_owner=before.seat_at(point.choice.target);
  step(state,first,rules);
  int target=point.choice.target, follower=first.to;
  const auto after_first=observe(state,learner);
  const int starting_outlet=next_empty_camp(after_first,target,kNone);
  auto streams=review_streams(seed);
  auto moved=point.moved;
  moved[first.from]=false; moved[first.to]=true;
  while (state.result==Result::Ongoing && !state.out[learner] && state.to_move!=learner) {
    const auto d=seat_decision(state);
    result.flag_exposure |= public_flag_exposure(observe(state,learner),moved,state.to_move);
    result.ally_turn |= state.to_move==ally;
    result.target_turn |= state.to_move==target_owner;
    const auto move=review_move(d,learner,response,true,weights,streams[state.to_move]);
    result.ally_contact |= state.to_move==ally && move.to==target;
    const auto e=step(state,move,rules);
    const auto view=observe(state,learner);
    target=next_tracked_node(target,target_owner,e,view);
    follower=next_tracked_node(follower,learner,e,view);
    observe_movement(moved,e,view);
    if (target!=kNone && Board::get().node(target).kind==NodeKind::Camp) result.escape=true;
  }
  if (state.result==Result::Ongoing && state.to_move==learner) {
    const auto view=observe(state,learner);
    const bool changed=target!=point.choice.target || result.flag_exposure ||
      (target!=kNone && next_empty_camp(view,target,kNone)!=starting_outlet);
    compare_camp_revisit(result,state,follower,target,changed,rules,response,weights,mix_seed(seed,901));
  }
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,response,true,weights,streams[state.to_move]);
    step(state,action,rules);
  }
  result.score=review_outcome(state.result,learner);
  return result;
}

inline void print_camp_continuation(const CampContinuation& result) {
  std::cout << '[' << result.score << ',' << result.next_group << ',' << int(result.escape)
            << ',' << int(result.flag_exposure) << ',' << int(result.ally_turn) << ','
            << int(result.ally_contact) << ',' << int(result.target_turn) << ','
            << result.pursue_score << ',' << result.switch_score << ']';
}

inline void camp_control_pairs(const Rules& rules, uint64_t seed, int learner, int variant,
                                const ShortWeights& weights, size_t kind) {
  const auto trace=camp_trace(rules,seed,learner,variant,weights);
  expect_value(bool(trace.points[kind]),"camp control point must exist in the frozen source");
  const auto& point=*trace.points[kind];
  const std::vector<Move> prefix(trace.history.begin(),trace.history.begin()+point.state.ply);
  const auto decision=seat_decision(point.state);
  const auto worlds=credit_worlds(trace.opening,point.state,decision,rules,prefix,weights,true);
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant
            << ",\"kind\":" << kind << ",\"worlds\":" << worlds.size() << ",\"point\":";
  print_camp_point(point);
  std::cout << ",\"cells\":[";
  bool comma=false;
  for (size_t world=0; world<worlds.size(); ++world)
    for (int response : {0,2})
      for (int stream : {0,1}) {
        std::cout << (comma ? "," : "") << "{\"world\":" << world << ",\"response\":" << response
                  << ",\"stream\":" << stream << ",\"pairs\":[";
        comma=true;
        for (uint64_t sample=0; sample<8; ++sample) {
          const auto common=mix_seed(mix_seed(seed,static_cast<uint64_t>(point.state.ply)),
                                     sample+(stream==0 ? 20 : 120));
          const auto a=camp_continue(worlds[world],point,point.choice.seal,rules,response,weights,common);
          const auto b=camp_continue(worlds[world],point,point.choice.press,rules,response,weights,common);
          std::cout << (sample ? ",[" : "[");
          print_camp_continuation(a);
          std::cout << ',';
          print_camp_continuation(b);
          std::cout << ']';
        }
        std::cout << "]}";
      }
  std::cout << "]}\n";
}
}
