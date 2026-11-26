#pragma once
#include "flag_urgency.h"

namespace jq {
struct ExchangeOffer {
  int flag, threat;
  Move guard;
  Move capture;
};

inline std::optional<ExchangeOffer> exchange_offer(const SeatDecision& decision,
                                                    const MovedSquares& moved,const Rules& rules,
                                                    bool only_later,
                                                    std::optional<Move> capture_override=std::nullopt) {
  const auto& view=decision.observation;
  const auto& board=Board::get();
  const int early=(view.viewer+(rules.ccw ? 1 : 3))%4;
  const int late=(view.viewer+(rules.ccw ? 3 : 1))%4;
  for (int flag=0; flag<kNodes; ++flag) {
    if (view.seat_at(flag)!=view.viewer || view.rank_at(flag)!=Flag) continue;
    const auto& adjacent=board.road(flag);
    const bool early_threat=std::any_of(adjacent.begin(),adjacent.end(),[&](int node) {
      return view.seat_at(node)==early && publicly_mobile(view,moved,node);
    });
    if (early_threat && only_later) return std::nullopt;
    const int enemy=early_threat ? early : late;
    for (const auto guard : decision.actions) {
      if (view.seat_at(guard.to)!=enemy || !publicly_mobile(view,moved,guard.to) ||
          std::find(adjacent.begin(),adjacent.end(),guard.to)==adjacent.end()) continue;
      if (capture_override && capture_override->to!=guard.to)
        return ExchangeOffer{flag,guard.to,guard,*capture_override};
      std::optional<Move> capture;
      for (const auto move : decision.actions) {
        const auto target=view.seat_at(move.to);
        if (move.to==guard.to || target<0 || team_of(target)==team_of(view.viewer)) continue;
        if (view.rank_at(move.to)==Flag) return ExchangeOffer{flag,guard.to,guard,move};
        if (!capture && moved[move.to]) capture=move;
      }
      if (capture) return ExchangeOffer{flag,guard.to,guard,*capture};
    }
  }
  return std::nullopt;
}

inline std::optional<Move> guard_friendly_hq(const SeatDecision& ally,const MovedSquares& moved) {
  const auto& view=ally.observation;
  const auto& board=Board::get();
  for (const auto action : ally.actions) {
    const int enemy=view.seat_at(action.to);
    if (enemy<0 || team_of(enemy)==team_of(view.viewer) || !publicly_mobile(view,moved,action.to)) continue;
    for (const auto node : board.road(action.to)) {
      const int owner=view.seat_at(node);
      if (board.node(node).kind==NodeKind::HQ && owner>=0 && team_of(owner)==team_of(view.viewer))
        return action;
    }
  }
  return std::nullopt;
}

inline int guard_share(const SeatDecision& d,const ExchangeOffer& offer,const ShortWeights& weights) {
  const auto probabilities=review_learned(d,weights);
  const auto guard=std::find(d.actions.begin(),d.actions.end(),offer.guard);
  const auto capture=std::find(d.actions.begin(),d.actions.end(),offer.capture);
  expect_value(guard!=d.actions.end() && capture!=d.actions.end(),"exchange actions missing");
  const double a=probabilities[guard-d.actions.begin()], b=probabilities[capture-d.actions.begin()];
  return static_cast<int>(std::lround(10000*a/(a+b)));
}

struct ExchangeTrial {
  int score=-1, lost_flag=0, reply_used=0, helper_used=0, helped_threat=0;
  int next_turn=0, next_pair=0, next_guard_share=-1, next_capture_score=-1, next_guard_score=-1;
  int capture_won=0, changed_threat=0;
  int enemy_window=-1, ally_window=-1, next_window=-1;
  bool ally_moved_elsewhere=false, pressure_changed=false;
};

inline std::vector<std::pair<int,int>> flag_road_pressure(const Observation& view,
                                                         const MovedSquares& moved,int flag) {
  std::vector<std::pair<int,int>> enemies;
  for (const auto node : Board::get().road(flag)) {
    const int owner=view.seat_at(node);
    if (publicly_mobile(view,moved,node) && team_of(owner)!=team_of(view.viewer))
      enemies.emplace_back(node,owner);
  }
  return enemies;
}

inline bool guarantees_interception(const Observation& view,const ExchangeOffer& offer,const Rules& rules) {
  const auto attacker=view.rank_at(offer.guard.from), target=view.rank_at(offer.threat);
  const auto removes=[&](int rank) {
    const auto outcome=resolve(attacker,rank,rules);
    return outcome==Outcome::AttackerWins || outcome==Outcome::BothRemoved;
  };
  if (target!=Unknown) return removes(target);
  // The offer requires a visibly mobile enemy, so this conservative set excludes mines and flags.
  for (int rank=Bomb; rank<RankCount; ++rank)
    if (!removes(rank)) return false;
  return true;
}

inline void review_exchange_next(ExchangeTrial& out,const State& state,const MovedSquares& moved,
                                 const ExchangeOffer& initial,const Rules& rules,
                                 const ShortWeights& weights,uint64_t seed) {
  const auto next=seat_decision(state);
  out.next_turn=1;
  out.next_window=state.ply;
  const auto pair=exchange_offer(next,moved,rules,false);
  if (!pair) return;
  out.next_pair=1;
  out.changed_threat=pair->threat!=initial.threat;
  out.next_guard_share=guard_share(next,*pair,weights);
  out.next_capture_score=review_outcome(review_rollout(state,pair->capture,rules,0,true,weights,seed),state.to_move);
  out.next_guard_score=review_outcome(review_rollout(state,pair->guard,rules,0,true,weights,seed),state.to_move);
}

inline ExchangeTrial exchange_trial(State state,const ExchangeOffer& pair,Move first,
                                     MovedSquares moved,const Rules& rules,int profile,
                                     const ShortWeights& weights,uint64_t seed) {
  const int learner=state.to_move, ally=(learner+2)%4, threat=state.seat_at[pair.threat];
  auto streams=review_streams(seed);
  ExchangeTrial out;
  const auto initial_move=step(state,first,rules);
  out.capture_won=initial_move.outcome==Outcome::AttackerWins;
  observe_movement(moved,initial_move,observe(state,learner));
  const auto initial_pressure=flag_road_pressure(observe(state,learner),moved,pair.flag);
  while (state.result==Result::Ongoing && state.to_move!=learner && !state.out[learner]) {
    const auto d=seat_decision(state);
    const int actor=state.to_move;
    std::optional<Move> response;
    if (actor==threat) {
      if (std::any_of(d.actions.begin(),d.actions.end(),[&](Move action){return action.to==pair.flag;}))
        out.enemy_window=state.ply;
      if (profile>=1) {
        response=public_hq_attack(d);
        out.reply_used=bool(response);
      }
    }
    if (actor==ally) {
      const auto defense=guard_friendly_hq(d,moved);
      if (defense) out.ally_window=state.ply;
      if (profile==2) {
        response=defense;
        out.helper_used=bool(response);
        out.helped_threat=response && response->to==pair.threat;
      }
    }
    const auto random_action=review_move(d,learner,0,true,weights,streams[actor]);
    const auto action=response.value_or(random_action);
    if (actor==ally) out.ally_moved_elsewhere=action.to!=pair.threat;
    const auto event=step(state,action,rules);
    out.lost_flag=out.lost_flag || (action.to==pair.flag &&
        (event.outcome==Outcome::AttackerWins || event.outcome==Outcome::BothRemoved));
    observe_movement(moved,event,observe(state,learner));
  }
  if (state.result==Result::Ongoing && state.to_move==learner) {
    out.pressure_changed=flag_road_pressure(observe(state,learner),moved,pair.flag)!=initial_pressure;
    review_exchange_next(out,state,moved,pair,rules,weights,mix_seed(seed,291));
  }
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto action=review_move(d,learner,0,true,weights,streams[state.to_move]);
    step(state,action,rules);
  }
  out.score=review_outcome(state.result,learner);
  return out;
}

inline void print_exchange_trial(const ExchangeTrial& a,bool extended) {
  std::cout << '[' << a.score << ',' << a.lost_flag << ',' << a.reply_used << ',' << a.helper_used
            << ',' << a.helped_threat << ',' << a.next_turn << ',' << a.next_pair << ','
            << a.next_guard_share << ',' << a.next_capture_score << ',' << a.next_guard_score
            << ',' << a.capture_won << ',' << a.changed_threat << ',' << a.enemy_window
            << ',' << a.ally_window << ',' << a.next_window;
  if (extended) std::cout << ',' << a.ally_moved_elsewhere << ',' << a.pressure_changed;
  std::cout << ']';
}

inline void print_exchange_case(const State& initial,const State& point,const ExchangeOffer& pair,
                                 const MovedSquares& moved,const Rules& rules,
                                 const std::vector<Move>& history,const ShortWeights& weights,uint64_t seed,
                                 bool extended) {
  const auto d=seat_decision(point);
  const auto worlds=credit_worlds(initial,point,d,rules,history,weights,true);
  std::cout << "{\"ply\":" << point.ply << ",\"guard_share\":" << guard_share(d,pair,weights)
            << ",\"capture\":";
  print_camp_move(pair.capture);
  std::cout << ",\"guard\":";
  print_camp_move(pair.guard);
  if (extended) {
    const auto delta=urgency_delta(d.observation,{pair.flag,pair.threat,pair.guard,pair.capture});
    const auto guaranteed=guarantees_interception(d.observation,pair,rules);
    const auto visible_attack=d.observation.rank_at(pair.capture.to)==Flag;
    for (const auto& world : worlds) {
      const auto alternate=seat_decision(world);
      const auto offer=exchange_offer(alternate,moved,rules,true);
      expect_value(offer && offer->guard==pair.guard && offer->capture==pair.capture &&
                   urgency_delta(alternate.observation,{pair.flag,pair.threat,pair.guard,pair.capture})==delta &&
                   guarantees_interception(alternate.observation,*offer,rules)==guaranteed &&
                   (alternate.observation.rank_at(pair.capture.to)==Flag)==visible_attack,
                   "compatible world changed public exchange features");
    }
    std::cout << ",\"feature_delta\":[" << delta[0];
    for (size_t i=1; i<delta.size(); ++i) std::cout << ',' << delta[i];
    std::cout << "],\"guaranteed_interception\":" << guaranteed
              << ",\"visible_flag_attack\":" << visible_attack;
  }
  std::cout << ",\"worlds\":" << worlds.size() << ",\"cells\":[";
  for (size_t world=0; world<worlds.size(); ++world)
    for (int profile : {0,1,2}) {
      std::cout << (world || profile ? ",[" : "[");
      for (uint64_t sample=0; sample<8; ++sample) {
        const auto common=mix_seed(mix_seed(seed,static_cast<uint64_t>(point.ply)),sample+444);
        const auto a=exchange_trial(worlds[world],pair,pair.capture,moved,rules,profile,weights,common);
        const auto b=exchange_trial(worlds[world],pair,pair.guard,moved,rules,profile,weights,common);
        std::cout << (sample ? ",[" : "[");
        print_exchange_trial(a,extended); std::cout << ','; print_exchange_trial(b,extended);
        std::cout << ']';
      }
      std::cout << ']';
    }
  std::cout << "]}";
}

inline void exchange_counterplay(const Rules& rules,uint64_t seed,int learner,int variant,
                                 const ShortWeights& weights,bool extended=false) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  MovedSquares moved{};
  std::vector<Move> history;
  bool found=false;
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant << ",\"case\":";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner && !found) {
      if (const auto pair=exchange_offer(d,moved,rules,true)) {
        print_exchange_case(initial,state,*pair,moved,rules,history,weights,seed,extended);
        found=true;
      }
    }
    const auto action=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    history.push_back(action);
    const auto info=step(state,action,rules);
    observe_movement(moved,info,observe(state,learner));
  }
  if (!found) std::cout << "null";
  std::cout << ",\"source_score\":" << review_outcome(state.result,learner) << "}\n";
}
}
