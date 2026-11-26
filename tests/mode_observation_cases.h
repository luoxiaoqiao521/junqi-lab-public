#pragma once
#include "../sim/goal_followthrough.h"

namespace jq {
inline void mode_rank_contract(const State& state, int viewer) {
  const auto dark=observe(state,viewer);
  const auto ally=observe(state,viewer,IdentityVisibility::AllyAndPublic);
  const auto full=observe(state,viewer,IdentityVisibility::AllIdentities);
  for (int node=0; node<kNodes; ++node) {
    const int owner=state.seat_at[node], rank=state.rank_at[node];
    const bool public_flag=owner>=0 && state.flag_revealed[owner] && rank==Flag;
    const bool available=owner==viewer || public_flag;
    const bool allied=owner>=0 && (viewer+2)%4==owner;
    expect_value(dark.rank_at(node)==(available ? rank : Unknown) &&
                 ally.rank_at(node)==(available || allied ? rank : Unknown) &&
                 full.rank_at(node)==rank &&
                 dark.seat_at(node)==ally.seat_at(node) && dark.seat_at(node)==full.seat_at(node),
                 "wrong identity visibility or occupancy");
  }
  std::vector<Move> dark_moves, ally_moves, full_moves;
  gen_moves(dark,viewer,dark_moves);
  gen_moves(ally,viewer,ally_moves);
  gen_moves(full,viewer,full_moves);
  expect_value(dark_moves==ally_moves && dark_moves==full_moves,
               "disclosed identities changed own legal move order");
}

inline void mode_observation_contract() {
  Rules rules;
  Rng rng(49012);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,rng);
  State initial;
  init_state(initial,layouts,0,rules);
  auto state=initial;
  std::vector<Move> prefix;
  MovedSquares moved{};
  ShortWeights weights{};
  for (int ply=0; ply<16; ++ply) {
    const auto d=seat_decision(state);
    const int viewer=d.observation.viewer;
    if (ply>=12) {
      mode_rank_contract(state,viewer);
      auto revealed=state;
      revealed.flag_revealed[(viewer+1)%4]=true;
      mode_rank_contract(revealed,viewer);
      const auto worlds=goal_worlds({initial,state,rules,prefix,moved,weights});
      expect_value(worlds.size()==4,"fixture requires changes in ally and enemy identities");
      expect_value(worlds[0].first==0 && worlds[1].first==0 && worlds[2].first==1 && worlds[3].first==1,
                   "fixture needs both enemy alternatives within each ally observation");
    }
    const auto move=d.actions[rng.below(static_cast<uint32_t>(d.actions.size()))];
    prefix.push_back(move);
    step(state,move,rules);
  }
  std::cout << "PASS identity visibility: four viewers, public flag, two ally groups, four worlds\n";
}
}
