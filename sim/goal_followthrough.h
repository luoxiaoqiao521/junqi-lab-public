#pragma once
#include "flag_guard_transfer.h"
#include "regional_arrival.h"

namespace jq {
struct GoalPosition {
  const State& initial;
  const State& current;
  const Rules& rules;
  const std::vector<Move>& prefix;
  const MovedSquares& moved;
  const ShortWeights& weights;
};

inline bool goal_vacancy_matters(const SeatDecision& d, Move previous, Move guard) {
  const auto& view=d.observation;
  if (view.seat_at(previous.from)!=kNone || view.seat_at(previous.to)!=(view.viewer+2)%4)
    return false;
  const auto blocked=project_quiet(view,{previous.to,previous.from});
  std::vector<Move> actions;
  gen_piece_moves(blocked,view.viewer,guard.from,actions);
  return std::find(actions.begin(),actions.end(),guard)==actions.end();
}

inline std::optional<State> goal_swap(const GoalPosition& p, const State& initial,
                                      const State& before, int owner, State& changed_initial) {
  const auto d=seat_decision(p.current);
  std::vector<std::pair<int,int>> pairs;
  for (int a=0; a<25; ++a) for (int b=a+1; b<25; ++b) {
    const auto hidden_live=[&](int slot) {
      const int id=initial.id_at[layout_slot_node(owner,slot)];
      return p.current.alive[id] && d.observation.rank_at(p.current.node_of[id])==Unknown;
    };
    if (hidden_live(a) || hidden_live(b)) pairs.emplace_back(a,b);
  }
  Rng rng(305746);
  for (int try_index=0; try_index<64 && !pairs.empty(); ++try_index) {
    const auto index=rng.below(static_cast<uint32_t>(pairs.size()));
    const auto [a,b]=pairs[index];
    pairs[index]=pairs.back();
    pairs.pop_back();
    if (!swap_in_layout(initial,changed_initial,owner,a,b)) continue;
    auto alternate=changed_initial;
    if (!replay_alternative(p.initial,alternate,p.rules,p.prefix,d.observation.viewer)) continue;
    const auto checked=seat_decision(alternate);
    expect_value(d.actions==checked.actions &&
                 review_learned(d,p.weights)==review_learned(checked,p.weights),
                 "goal worlds changed first-step policy");
    if (live_hidden_change(before,alternate)) return alternate;
  }
  return std::nullopt;
}

inline void verify_goal_groups(const std::vector<std::pair<int,State>>& worlds, int viewer) {
  for (size_t a=0; a<worlds.size(); ++a) for (size_t b=a+1; b<worlds.size(); ++b) {
    const auto ally_a=observe(worlds[a].second,viewer,IdentityVisibility::AllyAndPublic);
    const auto ally_b=observe(worlds[b].second,viewer,IdentityVisibility::AllyAndPublic);
    const bool same=std::equal(std::begin(ally_a.ranks),std::end(ally_a.ranks),std::begin(ally_b.ranks));
    expect_value(same==(worlds[a].first==worlds[b].first) &&
                 live_hidden_change(worlds[a].second,worlds[b].second),
                 "worlds did not refine the ally-visible identity partition");
  }
}

inline std::vector<std::pair<int,State>> goal_worlds(const GoalPosition& p) {
  const int viewer=p.current.to_move;
  const int enemy=(viewer+(p.rules.ccw ? 1 : 3))%4;
  State ally_initial, unused;
  std::vector<std::pair<int,State>> worlds{{0,p.current}};
  if (auto state=goal_swap(p,p.initial,p.current,enemy,unused)) worlds.emplace_back(0,*state);
  if (auto state=goal_swap(p,p.initial,p.current,(viewer+2)%4,ally_initial)) {
    worlds.emplace_back(1,*state);
    if (auto other=goal_swap(p,ally_initial,*state,enemy,unused)) worlds.emplace_back(1,*other);
  }
  verify_goal_groups(worlds,viewer);
  return worlds;
}

inline std::pair<int,bool> goal_trial(State state, Move first, const GoalPosition& p, int response,
                                      uint64_t seed, bool follow) {
  const int viewer=state.to_move;
  auto moved=p.moved;
  auto streams=review_streams(seed);
  auto event=step(state,first,p.rules);
  observe_movement(moved,event,observe(state,viewer));
  bool next=true, changed=false;
  while (state.result==Result::Ongoing) {
    const int seat=state.to_move;
    const auto d=seat_decision(state);
    auto action=review_move(d,viewer,response,true,p.weights,streams[seat]);
    if (seat==viewer && next) {
      next=false;
      if (follow) {
        const auto guard=urgent_public_guard(d,moved,p.rules);
        if (guard && !(action==guard->intercept)) {
          action=guard->intercept;
          changed=true;
        }
      }
    }
    event=step(state,action,p.rules);
    if (next) observe_movement(moved,event,observe(state,viewer));
  }
  return {review_outcome(state.result,viewer),changed};
}

inline std::vector<Move> goal_candidates(const SeatDecision& d, Move played, const FlagThreat& threat,
                                         uint64_t seed) {
  std::vector<Move> candidates{played,threat.intercept};
  if (!(played==threat.advance)) candidates.push_back(threat.advance);
  if (const auto attack=public_hq_attack(d); attack &&
      std::find(candidates.begin(),candidates.end(),*attack)==candidates.end())
    candidates.push_back(*attack);
  std::vector<Move> outside;
  for (auto move : d.actions)
    if (std::find(candidates.begin(),candidates.end(),move)==candidates.end()) outside.push_back(move);
  Rng rng(mix_seed(seed,907));
  if (!outside.empty()) candidates.push_back(outside[rng.below(static_cast<uint32_t>(outside.size()))]);
  return candidates;
}

inline void print_goal_cell(const State& state, const GoalPosition& p, const std::vector<Move>& moves,
                            int group, int response, uint64_t seed) {
  std::cout << "{\"group\":" << group << ",\"response\":" << response << ",\"samples\":[";
  for (uint64_t i=0; i<14; ++i) {
    std::cout << (i ? ",[" : "[");
    const auto common=mix_seed(mix_seed(seed,p.current.ply),81170+i);
    for (size_t a=0; a<moves.size(); ++a) {
      if (a) std::cout << ',';
      std::cout << goal_trial(state,moves[a],p,response,common,false).first;
    }
    for (size_t a=0; a<2; ++a) {
      const auto [score,changed]=goal_trial(state,moves[a],p,response,common,true);
      std::cout << ",[" << score << ',' << changed << ']';
    }
    std::cout << ']';
  }
  std::cout << "]}";
}

inline void print_goal_point(const GoalPosition& p, const SeatDecision& d, const FlagThreat& threat,
                             Move played, uint64_t seed, bool vacancy) {
  const auto moves=goal_candidates(d,played,threat,seed);
  const auto worlds=goal_worlds(p);
  std::cout << "{\"ply\":" << p.current.ply << ",\"vacancy\":" << vacancy
            << ",\"legal\":" << d.actions.size() << ",\"same_features\":"
            << (short_features(d.observation,played)==short_features(d.observation,threat.intercept))
            << ",\"moves\":[";
  for (size_t a=0; a<moves.size(); ++a) {
    if (a) std::cout << ',';
    print_camp_move(moves[a]);
  }
  std::cout << "],\"cells\":[";
  for (size_t i=0; i<worlds.size(); ++i) for (int response : {0,2}) {
    if (i || response) std::cout << ',';
    print_goal_cell(worlds[i].second,p,moves,worlds[i].first,response,seed);
  }
  std::cout << "]}";
}

inline void goal_followthrough(const Rules& rules, uint64_t seed, int viewer, int variant,
                                const ShortWeights& weights) {
  Rng rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,rng);
  const int first=rules.random_first_seat ? int(rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  std::vector<Move> prefix;
  MovedSquares moved{};
  std::optional<Move> ally_quiet;
  std::array<int,2> offers{};
  int decisions=0;
  bool comma=false;
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << viewer << ",\"variant\":" << variant
            << ",\"first\":" << first << ",\"points\":[";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto move=review_move(d,viewer,variant,true,weights,streams[state.to_move]);
    if (state.to_move==viewer) {
      ++decisions;
      const auto threat=public_flag_threats(d,moved,rules)[0];
      if (threat && !(move==threat->intercept)) {
        const bool vacancy=ally_quiet && goal_vacancy_matters(d,*ally_quiet,threat->intercept);
        if (++offers[vacancy]==1) {
          if (comma) std::cout << ',';
          comma=true;
          print_goal_point({initial,state,rules,prefix,moved,weights},d,*threat,move,seed,vacancy);
        }
      }
      ally_quiet.reset();
    }
    prefix.push_back(move);
    const auto event=step(state,move,rules);
    observe_movement(moved,event,observe(state,viewer));
    if (event.seat==(viewer+2)%4 && event.outcome==Outcome::Move) ally_quiet=move;
  }
  std::cout << "],\"decisions\":" << decisions << ",\"offers\":[" << offers[0] << ',' << offers[1]
            << "],\"score\":" << review_outcome(state.result,viewer) << "}\n";
}
}
