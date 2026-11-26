#pragma once
#include "selfplay_elimination_cases.h"
#include "../sim/relay_continuity.h"

namespace jq {

inline uint8_t relay_node(int seat,int row,int col) {
  return static_cast<uint8_t>(Board::get().seat_node(seat,row,col));
}

inline bool same_relay_event(const RelayEvent& x,const RelayEvent& y) {
  const auto& a=x.step;
  const auto& b=y.step;
  return x.after==y.after && a.move==b.move && a.seat==b.seat && a.outcome==b.outcome &&
         a.eliminated_mask==b.eliminated_mask && a.skipped_mask==b.skipped_mask &&
         a.revealed_mask==b.revealed_mask && a.end_reason==b.end_reason &&
         std::equal(std::begin(a.exit_reason),std::end(a.exit_reason),std::begin(b.exit_reason));
}

inline bool same_relay_trace(const RelayTrace& a,const RelayTrace& b) {
  const auto& x=a.state;
  const auto& y=b.state;
  const auto equal=[](const auto& left,const auto& right) {
    return std::equal(std::begin(left),std::end(left),std::begin(right));
  };
  return state_hash(x)==state_hash(y) && x.match==y.match && x.result==y.result &&
         equal(x.seat_at,y.seat_at) && equal(x.rank_at,y.rank_at) && equal(x.alive,y.alive) &&
         equal(x.node_of,y.node_of) && equal(x.rank_of,y.rank_of) &&
         a.changed==b.changed && a.events.size()==b.events.size() &&
         std::equal(a.events.begin(),a.events.end(),b.events.begin(),same_relay_event) &&
         std::equal(a.streams.begin(),a.streams.end(),b.streams.begin(),
                    [](const Rng& v,const Rng& w) { return v.s==w.s; });
}

inline Rules relay_test_rules() {
  Rules rules;
  rules.eliminate_exhausted_movers=true;
  rules.truncation_cap=30;
  return rules;
}

inline RelayPolicy relay_random_policy() {
  return [](const SeatDecision& d,Rng& rng) {
    return d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
  };
}

inline State relay_generated(uint64_t seed,const Rules& rules) {
  Rng rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for(auto& layout:layouts) generate_layout(layout,rng);
  State initial;
  init_state(initial,layouts,0,rules);
  return initial;
}

inline void relay_matching_and_override() {
  auto rules=relay_test_rules();
  auto policy=relay_random_policy();
  int changed=0;
  for(uint64_t seed=601;seed<625;++seed) {
    const auto root=relay_generated(seed,rules);
    const auto preparation=seat_decision(root).actions.front();
    const auto common=relay_checkpoint(root,preparation,2,rules,seed,policy);
    single_case_check(common.can_decide(),"generated fixture never reached the receiver");
    const auto baseline=relay_finish(common,rules,policy);
    int called=0;
    const auto unused=relay_finish(common,rules,policy,[&](const SeatDecision& d,Move) -> std::optional<Move> {
      ++called;
      single_case_check(d.observation.viewer==2,"selector was given the wrong viewer");
      return std::nullopt;
    });
    single_case_check(called==1 && same_relay_trace(baseline,unused),"untriggered relay diverged");
    const auto same=relay_finish(common,rules,policy,
        [](const SeatDecision&,Move fallback) -> std::optional<Move> { return fallback; });
    single_case_check(same_relay_trace(baseline,same),"selecting the fallback changed the trajectory");
    int positives=0;
    const auto planned=relay_finish(common,rules,policy,[&](const SeatDecision& d,Move fallback) -> std::optional<Move> {
      ++positives;
      for(auto move:d.actions) if(!(move==fallback)) return move;
      return std::nullopt;
    });
    single_case_check(positives==1,"relay proposed more than one later decision");
    if(!planned.changed) continue;
    ++changed;
    const auto first=std::mismatch(baseline.events.begin(),baseline.events.end(),planned.events.begin(),
                                  planned.events.end(),same_relay_event);
    single_case_check(size_t(first.first-baseline.events.begin())==common.trace.events.size() &&
                      first.first->step.seat==2 && first.second->step.seat==2,
                      "relay diverged before the receiver took its turn");
  }
  single_case_check(changed==24,"expected a legal single intervention at each generated checkpoint");
  std::cout<<"PASS relay paired trajectories: 24 no-trigger, 24 same-fallback, 24 receiver overrides\n";
}

inline State relay_sparring_position() {
  auto state=selfplay_fixture();
  state.out[2]=false;
  place_single_case_piece(state,50,relay_node(2,5,1),Flag);
  return state;
}

inline void relay_skip_then_act() {
  auto root=relay_sparring_position();
  auto rules=relay_test_rules();
  place_single_case_piece(root,51,relay_node(2,5,0),Platoon);
  place_single_case_piece(root,1,relay_node(2,4,0),Engineer);
  place_single_case_piece(root,2,relay_node(0,4,2),Platoon);
  place_single_case_piece(root,26,relay_node(1,0,1),Brigade);
  const Move setup{relay_node(0,4,2),relay_node(0,4,3)};
  const Move enemy{relay_node(1,0,1),relay_node(1,0,2)};
  const Move opens{relay_node(2,4,0),relay_node(2,4,1)};
  RelayPolicy policy=[&](const SeatDecision& d,Rng& rng) {
    rng.next();
    const int ply=d.observation.ply;
    if(ply==0) return setup;
    if(ply==1) return enemy;
    if(ply==2) return opens;
    if(ply==3) return Move{enemy.to,enemy.from};
    return d.actions.front();
  };
  const auto pinned=relay_checkpoint(root,setup,2,rules,777,policy);
  const auto& trace=pinned.trace.events;
  single_case_check(pinned.can_decide() && trace.size()==4 && (trace[1].step.skipped_mask&4) &&
                    trace[2].step.seat==0 && trace[3].step.seat==1,"receiver was not restored after a genuine skip");
  auto expected=review_streams(777);
  for(const auto& event:trace) expected[event.step.seat].next();
  for(int seat=0;seat<kSeats;++seat) single_case_check(pinned.trace.streams[seat].s==expected[seat].s,"checkpoint lost RNG consumption");
  const auto baseline=relay_finish(pinned,rules,policy);
  const auto hold=relay_finish(pinned,rules,policy,[](const SeatDecision&,Move) -> std::optional<Move> { return std::nullopt; });
  single_case_check(same_relay_trace(baseline,hold),"post-skip no-trigger trace diverged");
  std::cout<<"PASS relay actual turn: skip, opener acts again, receiver regains choice\n";
}

inline void relay_terminal_and_retirement() {
  auto rules=relay_test_rules();
  auto root=relay_sparring_position();
  const Move setup{relay_node(0,4,0),relay_node(0,4,1)};
  const Move enemy_capture{relay_node(2,4,1),relay_node(2,5,1)};
  const Move winner{relay_node(1,4,1),relay_node(1,5,1)};
  place_single_case_piece(root,1,setup.from,Engineer);
  place_single_case_piece(root,2,winner.from,Engineer);
  place_single_case_piece(root,26,enemy_capture.from,Engineer);
  place_single_case_piece(root,51,relay_node(2,0,2),Brigade);
  RelayPolicy policy=[&](const SeatDecision& d,Rng& rng) {
    rng.next();
    return d.observation.ply==0 ? setup : d.observation.viewer==1 ? enemy_capture : winner;
  };
  int calls=0;
  const RelaySelection forbidden=[&](const SeatDecision&,Move) -> std::optional<Move> { ++calls; return std::nullopt; };
  // With no reserve, the last enemy mover locks itself inside the captured HQ.
  const auto won_early=relay_checkpoint(root,setup,2,rules,931,policy);
  const auto early_result=relay_finish(won_early,rules,policy,forbidden);
  single_case_check(early_result.events.size()==2 && team_score(early_result.state.result,2)==1.,
                    "early natural terminal did not retain the team win");
  place_single_case_piece(root,27,relay_node(1,0,0),Platoon);
  const auto retired=relay_checkpoint(root,setup,2,rules,931,policy);
  single_case_check(!retired.can_decide() && retired.trace.state.out[2] &&
                    retired.trace.state.result==Result::Ongoing,"expected only the receiver to retire");
  const auto b=relay_finish(retired,rules,policy);
  const auto c=relay_finish(retired,rules,policy,forbidden);
  single_case_check(same_relay_trace(b,c) && team_score(c.state.result,2)==1.,"receiver retirement lost team result");
  for(const bool draw:{false,true}) {
    auto ending=rules;
    if(draw) ending.no_progress_limit=1;
    else ending.truncation_cap=1;
    const auto early=relay_checkpoint(root,setup,2,ending,931,policy);
    const auto result=relay_finish(early,ending,policy,forbidden);
    single_case_check(result.events.size()==1 && result.state.result==(draw?Result::Draw:Result::Truncated) &&
                      team_score(result.state.result,2)==(draw?std::optional<double>(0.5):std::nullopt),
                      "early draw or cutoff incorrectly reached a decision");
  }
  single_case_check(calls==0,"selector called after the receiver or game had ended");
  std::cout<<"PASS relay settlement: receiver retires and team wins; early draw and cutoff do not call selector\n";
}

inline void relay_reject_illegal_selection() {
  const auto rules=relay_test_rules();
  const auto policy=relay_random_policy();
  const auto root=relay_generated(601,rules);
  const auto common=relay_checkpoint(root,seat_decision(root).actions.front(),2,rules,601,policy);
  bool caught=false;
  try {
    relay_finish(common,rules,policy,[](const SeatDecision&,Move) -> std::optional<Move> { return Move{0,0}; });
  } catch(const std::invalid_argument&) { caught=true; }
  single_case_check(caught,"illegal receiver action was accepted");
  std::cout<<"PASS relay invalid override rejected\n";
}

inline void relay_hidden_worlds() {
  const auto rules=relay_test_rules();
  const auto root=relay_generated(602,rules);
  const auto preparation=seat_decision(root).actions.front();
  const auto policy=relay_random_policy();
  const auto original=relay_checkpoint(root,preparation,2,rules,602,policy);
  const auto root_view=seat_decision(root);
  State altered;
  for(int a=0;a<kPiecesPerSeat;++a) for(int b=a+1;b<kPiecesPerSeat;++b) {
    const auto left=root.rank_at[layout_slot_node(1,a)];
    const auto right=root.rank_at[layout_slot_node(1,b)];
    if(left<Platoon || right<Platoon || !swap_in_layout(root,altered,1,a,b)) continue;
    const auto other=relay_checkpoint(altered,preparation,2,rules,602,policy);
    if(!other.can_decide() || other.trace.events.size()!=original.trace.events.size()) continue;
    const bool same_steps=std::equal(original.trace.events.begin(),original.trace.events.end(),
        other.trace.events.begin(),[](auto x,auto y) { y.after=x.after; return same_relay_event(x,y); });
    if(!same_steps) continue;
    single_case_check(state_hash(root)!=state_hash(altered) &&
                      same_seat_view(root,altered,0) && root_view.actions==seat_decision(altered).actions &&
                      same_seat_view(original.trace.state,other.trace.state,2) &&
                      seat_decision(original.trace.state).actions==seat_decision(other.trace.state).actions,
                      "equivalent public roots leaked hidden identity");
    std::cout<<"PASS relay hidden-world boundary: same public history and ordered actions\n";
    return;
  }
  single_case_check(false,"missing compatible hidden-world fixture");
}

inline void relay_continuity_cases() {
  relay_matching_and_override();
  relay_skip_then_act();
  relay_terminal_and_retirement();
  relay_reject_illegal_selection();
  relay_hidden_worlds();
}

}
