#pragma once
#include <array>
#include <optional>
#include "replay.h"
#include "../strategy/random.h"

namespace jq {
using MovedSquares = std::array<bool,kNodes>;

inline void observe_movement(MovedSquares& moved, const StepInfo& event,
                             const Observation& after) {
  moved[event.move.from]=false;
  if (event.outcome==Outcome::Move || event.outcome==Outcome::AttackerWins)
    moved[event.move.to]=true;
  for (int node=0; node<kNodes; ++node)
    if (after.seat_at(node)<0) moved[node]=false;
}
inline bool same_seat_view(const State& a, const State& b, int seat) {
  return a.result==b.result && same_observation(observe(a,seat),observe(b,seat));
}

inline bool replay_alternative(State original, State& alternative, const Rules& rules,
                               const std::vector<Move>& prefix, int viewer) {
  if (!same_seat_view(original,alternative,viewer)) return false;
  for (const Move action : prefix) {
    if (!is_legal(alternative,action) || alternative.result!=Result::Ongoing) return false;
    const auto a=step(original,action,rules), b=step(alternative,action,rules);
    if (a.seat!=b.seat || a.outcome!=b.outcome || a.eliminated_mask!=b.eliminated_mask ||
        a.revealed_mask!=b.revealed_mask || !same_seat_view(original,alternative,viewer)) return false;
  }
  return true;
}

inline bool swap_in_layout(const State& initial, State& alternative, int seat, int a, int b) {
    int8_t layout[25];
    for (int i=0; i<25; ++i) layout[i]=initial.rank_at[layout_slot_node(seat,i)];
    if (layout[a]==layout[b]) return false;
    std::swap(layout[a],layout[b]);
    if (!valid_layout(layout)) return false;
    alternative=initial;
    for (const int i : {a,b}) {
      const int node=layout_slot_node(seat,i);
      alternative.rank_at[node]=layout[i];
      alternative.rank_of[alternative.id_at[node]]=layout[i];
    }
    return true;
}

inline std::vector<State> draw_worlds(const State& initial, const State& at,
                                      const Rules& rules, const std::vector<Move>& prefix) {
  std::vector<State> worlds{at};
  Rng rng(67017);
  for (int attempt=0; attempt<96 && worlds.size()<3; ++attempt) {
    const int seat=(at.to_move+1+2*int(rng.below(2)))%4;
    const int a=int(rng.below(25)), b=int(rng.below(25));
    State alternative;
    if (!swap_in_layout(initial,alternative,seat,a,b)) continue;
    if (!replay_alternative(initial,alternative,rules,prefix,at.to_move)) continue;
    bool distinct=true;
    for (const auto& world : worlds)
      distinct &= state_hash(world)!=state_hash(alternative);
    if (distinct) worlds.push_back(alternative);
  }
  return worlds;
}

inline void report_draw_branch(const State& at, Move action, const Rules& rules,
                               int world, bool heuristic, bool evidence = false,
                               bool opponents_only = false) {
  int totals[5]{};
  State after=at;
  step(after,action,rules);
  const int samples=after.result==Result::Ongoing ? 8 : 0;
  if (!samples) totals[int(after.result)]=8;
  for (uint64_t seed=0; seed<static_cast<uint64_t>(samples); ++seed) {
    State branch=after;
    std::array<Rng,4> streams{Rng(mix_seed(seed,10)),Rng(mix_seed(seed,11)),
                              Rng(mix_seed(seed,12)),Rng(mix_seed(seed,13))};
    while (branch.result==Result::Ongoing) {
      const auto o=observe(branch,branch.to_move);
      Move move;
      auto& rng=streams[o.viewer];
      const bool use_heuristic=heuristic &&
        (!opponents_only || team_of(o.viewer)!=team_of(at.to_move));
      const bool found=use_heuristic ? heuristic_move(o,rng,move) : random_move(o,rng,move);
      expect_value(found,"missing legal continuation");
      step(branch,move,rules);
    }
    ++totals[int(branch.result)];
  }
  if (evidence)
    std::cout << "FIRST " << world << ' ' << heuristic << ' '
              << int(action.from) << ' ' << int(action.to) << ' '
              << int(after.result) << ' ' << samples << '\n';
  std::cout << "BRANCH " << world << ' ' << heuristic << ' '
            << int(action.from) << ' ' << int(action.to) << ' '
            << totals[team_of(at.to_move)+1] << ' ' << totals[int(Result::Draw)] << ' '
            << totals[2-team_of(at.to_move)] << ' ' << totals[int(Result::Truncated)] << '\n';
}

inline void report_target_worlds(const State& initial, const State& last,
                                 const Rules& rules, const std::vector<Move>& prefix,
                                 Move quiet, const std::vector<Move>& legal) {
  const auto o=observe(last,last.to_move);
  const Move attack{59,58};
  expect_value(o.viewer==2 && last.ply==700 &&
               std::find(legal.begin(),legal.end(),attack)!=legal.end() &&
               o.rank_at(attack.to)==Unknown && last.seat_at[attack.to]>=0,
               "unexpected target fixture");
  const int owner=last.seat_at[attack.to], slot=last.id_at[attack.to]%25;
  int world=0;
  for (int b=-1; b<25; ++b) {
    State alternative=last;
    if (b>=0 && (!swap_in_layout(initial,alternative,owner,slot,b) ||
        !replay_alternative(initial,alternative,rules,prefix,o.viewer))) continue;
    std::cout << "TARGET " << world << ' ' << b << ' ' << int(alternative.rank_at[attack.to]) << '\n';
    for (bool heuristic : {false,true})
      for (Move action : {quiet,attack}) report_draw_branch(alternative,action,rules,world,heuristic,true);
    ++world;
  }
}

// Two rounds of reversible quiet moves increase pressure while restoring occupancy.
inline std::vector<Move> pressure_cycle(const State& low, const Rules& rules) {
  const int live = int(std::count(std::begin(low.out),std::end(low.out),false));
  if (low.no_progress+2*live+1!=rules.no_progress_limit) return {};
  for (uint64_t seed=0; seed<64; ++seed) {
    State high=low;
    Rng rng(seed);
    std::array<Move,4> forwards{};
    std::vector<Move> path;
    for (int i=0; i<2*live && high.result==Result::Ongoing; ++i) {
      const int seat=high.to_move;
      Move action;
      if (i<live) {
        std::vector<Move> quiet;
        legal_moves(high,quiet);
        quiet.erase(std::remove_if(quiet.begin(),quiet.end(),
                    [&](Move a){return high.seat_at[a.to]>=0;}),quiet.end());
        if (quiet.empty()) break;
        action=quiet[rng.below(static_cast<uint32_t>(quiet.size()))];
        forwards[seat]=action;
      } else action={forwards[seat].to,forwards[seat].from};
      if (!is_legal(high,action) || high.seat_at[action.to]>=0) break;
      step(high,action,rules);
      path.push_back(action);
    }
    if (path.size()!=static_cast<size_t>(2*live) || high.to_move!=low.to_move ||
        high.result!=Result::Ongoing || std::memcmp(high.id_at,low.id_at,sizeof low.id_at)!=0) continue;
    const auto quiet=path.front();
    State a=low, b=high;
    step(a,quiet,rules);
    step(b,quiet,rules);
    if (a.result!=Result::Ongoing || b.result!=Result::Draw) continue;
    return path;
  }
  return {};
}

inline void print_pressure(const State& low, const State& high, const std::vector<Move>& cycle) {
  const auto quiet=cycle.front();
  std::cout << "PRESSURE " << low.ply << ' ' << int(low.to_move) << ' '
            << low.no_progress << ' ' << high.no_progress << ' '
            << int(quiet.from) << ' ' << int(quiet.to);
  for (auto move : cycle) std::cout << ' ' << int(move.from) << ' ' << int(move.to);
  std::cout << '\n';
}

inline bool pressure_successor(State& high, const State& low, const Rules& rules,
                               const std::vector<Move>& cycle) {
  high=low;
  for (auto move : cycle) {
    if (high.result!=Result::Ongoing || !is_legal(high,move) || high.seat_at[move.to]>=0) return false;
    step(high,move,rules);
  }
  return high.result==Result::Ongoing && high.to_move==low.to_move &&
    high.no_progress+1==rules.no_progress_limit &&
    std::memcmp(high.id_at,low.id_at,sizeof low.id_at)==0;
}

inline void report_pressure_ranking(const State& initial, const State& low, const Rules& rules,
                                    const std::vector<Move>& prefix, const std::vector<Move>& cycle,
                                    bool opponents_only = false, const MovedSquares* moved = nullptr) {
  const auto observation=observe(low,low.to_move);
  std::vector<Move> legal, quiet{cycle.front()}, attack;
  gen_moves(observation,observation.viewer,legal);
  for (auto move : legal) {
    if (observation.seat_at(move.to)<0) {
      if (quiet.size()<2 && !(move==quiet.front())) quiet.push_back(move);
    } else if (attack.size()<2 && team_of(observation.seat_at(move.to))!=team_of(observation.viewer) &&
               (!moved || (*moved)[move.to] || std::none_of(cycle.begin(),cycle.end(),
                  [&](Move step){return step.from==move.to;}))) {
      attack.push_back(move);
    }
  }
  if (attack.empty()) return;
  auto worlds=draw_worlds(initial,low,rules,prefix);
  for (auto move : attack) {
    const int owner=low.seat_at[move.to], slot=low.id_at[move.to]%25;
    for (int b=0; b<25; ++b) {
      State other;
      if (!swap_in_layout(initial,other,owner,slot,b) ||
          !replay_alternative(initial,other,rules,prefix,low.to_move)) continue;
      bool distinct=true;
      for (const auto& world : worlds) distinct &= state_hash(world)!=state_hash(other);
      if (distinct) worlds.push_back(other);
      break;
    }
  }
  State high;
  expect_value(pressure_successor(high,low,rules,cycle),"invalid pressure cycle");
  print_pressure(low,high,cycle);
  quiet.insert(quiet.end(),attack.begin(),attack.end());
  for (auto move : quiet)
    std::cout << "ACTION " << int(move.from) << ' ' << int(move.to) << ' '
              << (low.seat_at[move.to]>=0) << '\n';
  if (opponents_only)
    for (auto move : attack)
      std::cout << "HQ " << int(move.from) << ' ' << int(move.to) << ' '
                << (Board::get().node(move.to).kind==NodeKind::HQ) << '\n';
  if (moved)
    for (auto move : attack)
      std::cout << "MOVED " << int(move.from) << ' ' << int(move.to) << ' ' << (*moved)[move.to] << '\n';
  int count=0;
  for (const auto& world : worlds) {
    if (!pressure_successor(high,world,rules,cycle)) continue;
    if (moved)
      for (auto move : attack)
        if ((*moved)[move.to])
          expect_value(world.rank_at[move.to]!=Flag && world.rank_at[move.to]!=Mine,
                       "public movement cannot belong to flag or mine");
    const int cell=count++*2;
    for (bool profile : {false,true})
      for (auto move : quiet) {
        expect_value(is_legal(world,move) && is_legal(high,move),"inconsistent candidates");
        report_draw_branch(world,move,rules,cell,profile,true,opponents_only);
        report_draw_branch(high,move,rules,cell+1,profile,true,opponents_only);
      }
  }
  std::cout << "WORLDS " << count << '\n';
}

inline void draw_branches(const std::string& path, bool targeted = false, bool ranking = false,
                          bool opponents_only = false, bool track_movement = false) {
  std::istringstream input(read_text(path));
  std::string line;
  std::getline(input,line);
  Rules rules;
  const auto initial=replay_initial(parse_json(line),rules);
  std::vector<Move> moves;
  State last=initial;
  std::optional<State> pressure;
  MovedSquares moved{}, pressure_moved{};
  replay(path,false,[&](const State& state, const StepInfo& event, const State& after){
    last=state; moves.push_back(event.move);
    const auto live=std::count(std::begin(state.out),std::end(state.out),false);
    if ((targeted || ranking) && state.no_progress+2*live+1==rules.no_progress_limit) {
      pressure=state; pressure_moved=moved;
    }
    if (track_movement) observe_movement(moved,event,observe(after,state.to_move));
  });
  const auto quiet=moves.back();
  moves.pop_back();
  State final=last;
  step(final,quiet,rules);
  if (final.result!=Result::Draw || last.seat_at[quiet.to]>=0 ||
      last.no_progress+1!=rules.no_progress_limit) return;
  const auto o=observe(last,last.to_move);
  std::vector<Move> legal;
  gen_moves(o,o.viewer,legal);
  if (targeted || ranking) {
    if (targeted) report_target_worlds(initial,last,rules,moves,quiet,legal);
    if (!pressure) return;
    const auto cycle=pressure_cycle(*pressure,rules);
    if (cycle.empty()) return;
    if (ranking) {
      moves.resize(static_cast<size_t>(pressure->ply));
      report_pressure_ranking(initial,*pressure,rules,moves,cycle,opponents_only,
                              track_movement ? &pressure_moved : nullptr);
    } else {
      State high;
      expect_value(pressure_successor(high,*pressure,rules,cycle),"invalid pressure cycle");
      print_pressure(*pressure,high,cycle);
      for (bool profile : {false,true}) {
        report_draw_branch(*pressure,cycle.front(),rules,0,profile);
        report_draw_branch(high,cycle.front(),rules,1,profile);
      }
    }
    return;
  }
  const auto worlds=draw_worlds(initial,last,rules,moves);
  std::cout << "DRAW " << last.ply << ' ' << int(last.to_move) << ' '
            << last.no_progress << ' ' << worlds.size() << ' '
            << int(quiet.from) << ' ' << int(quiet.to) << '\n';
  std::vector<Move> actions{quiet};
  for (const auto a : legal)
    if (o.seat_at(a.to)>=0 && team_of(o.seat_at(a.to))!=team_of(o.viewer) && actions.size()<5)
      actions.push_back(a);
  for (size_t world=0; world<worlds.size(); ++world)
    for (const auto action : actions)
      for (bool heuristic : {false,true})
        report_draw_branch(worlds[world],action,rules,static_cast<int>(world),heuristic);
}
}
