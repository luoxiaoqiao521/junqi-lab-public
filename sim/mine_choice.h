#pragma once
#include "mine_evidence.h"

namespace jq {
enum class MineStudy { AttackPair, InformationValue, QuietRecheck };

struct MineOffer {
  Move engineer, ordinary;
};

inline std::optional<MineOffer> mine_offer(const SeatDecision& d, const OpeningMarks& marks) {
  const auto& view=d.observation;
  for (const auto engineer : d.actions) {
    const auto owner=view.seat_at(engineer.to);
    if (view.rank_at(engineer.from)!=Engineer || owner<0 || view.rank_at(engineer.to)!=Unknown ||
        owner!=Board::get().node(engineer.to).seat || marks.moved[engineer.to] ||
        !(marks.ranks[engineer.to] & (1u << Mine))) continue;
    for (const auto ordinary : d.actions)
      if (ordinary.to==engineer.to && view.rank_at(ordinary.from)>=Company)
        return MineOffer{engineer,ordinary};
  }
  return std::nullopt;
}

inline std::optional<State> flip_target_mine(const State& initial,const State& point,
                                             int target,const Rules& rules,const std::vector<Move>& prefix) {
  const int owner=point.seat_at[target], slot=point.id_at[target]%kPiecesPerSeat;
  const auto original=point.rank_at[target];
  for (int other=0; other<kPiecesPerSeat; ++other) {
    const auto replacement=initial.rank_at[layout_slot_node(owner,other)];
    if ((replacement==Mine)==(original==Mine)) continue;
    State world;
    if (swap_in_layout(initial,world,owner,slot,other) &&
        replay_alternative(initial,world,rules,prefix,point.to_move)) {
      if ((world.rank_at[target]==Mine)!=(original==Mine)) return world;
    }
  }
  return std::nullopt;
}

inline void mine_offer_results(const std::array<State,2>& worlds, const MineOffer& offer,
                               const Rules& rules,const ShortWeights& weights,uint64_t seed) {
  for (size_t world=0; world<worlds.size(); ++world)
    for (int response : {0,2}) {
      std::cout << (world || response ? ",[" : "[");
      const auto& point=worlds[world];
      for (uint64_t sample=0; sample<8; ++sample) {
        const auto common=mix_seed(seed,sample+551);
        const auto a=review_rollout(point,offer.engineer,rules,response,true,weights,common);
        const auto b=review_rollout(point,offer.ordinary,rules,response,true,weights,common);
        std::cout << (sample ? ",[" : "[") << review_outcome(a,point.to_move)
                  << ',' << review_outcome(b,point.to_move) << ']';
      }
      std::cout << ']';
    }
}

inline void report_mine_offer(const State& initial,const State& point,const MineOffer& offer,
                              const Rules& rules,const std::vector<Move>& prefix,
                              const ShortWeights& weights,uint64_t seed) {
  const auto target=offer.engineer.to;
  std::cout << "{\"ply\":" << point.ply << ",\"target\":" << int(target)
            << ",\"engineer_from\":" << int(offer.engineer.from) << ",\"ordinary_from\":"
            << int(offer.ordinary.from);
  if (const auto alternative=flip_target_mine(initial,point,target,rules,prefix)) {
    const auto a=seat_decision(point), b=seat_decision(*alternative);
    expect_value(a.actions==b.actions && review_learned(a,weights)==review_learned(b,weights),
                 "mine-compatible world changed the player policy");
    const std::array<State,2> worlds=point.rank_at[target]==Mine ?
      std::array<State,2>{point,*alternative} : std::array<State,2>{*alternative,point};
    std::cout << ",\"immediate\":[";
    for (size_t index=0; index<worlds.size(); ++index) {
      const auto& world=worlds[index];
      const int defender=world.rank_at[target];
      std::cout << (index ? ",[" : "[") << int(resolve(Engineer,defender,rules))
                << ',' << int(resolve(world.rank_at[offer.ordinary.from],defender,rules)) << ']';
    }
    std::cout << "],\"cells\":[";
    mine_offer_results(worlds,offer,rules,weights,mix_seed(seed,static_cast<uint64_t>(point.ply)));
    std::cout << ']';
  } else {
    std::cout << ",\"immediate\":[],\"cells\":[]";
  }
  std::cout << "}";
}

inline std::vector<Move> mine_information_moves(const SeatDecision& d,const MineOffer& offer,
                                                 Move played,const ShortWeights& weights,uint64_t seed) {
  std::vector<Move> moves{played};
  const auto add=[&](Move move) {
    if (std::find(moves.begin(),moves.end(),move)==moves.end()) moves.push_back(move);
  };
  add(offer.engineer);
  add(offer.ordinary);
  const auto probabilities=review_learned(d,weights);
  std::optional<size_t> quiet;
  for (size_t i=0;i<d.actions.size();++i)
    if (d.observation.seat_at(d.actions[i].to)==kNone &&
        (!quiet || probabilities[i]>probabilities[*quiet])) quiet=i;
  if (quiet) add(d.actions[*quiet]);
  std::vector<Move> outside;
  for (const auto move : d.actions)
    if (std::find(moves.begin(),moves.end(),move)==moves.end()) outside.push_back(move);
  Rng rng(mix_seed(seed,330));
  if (!outside.empty()) add(outside[rng.below(static_cast<uint32_t>(outside.size()))]);
  return moves;
}

inline void mine_information_panel(const State& point,const std::vector<Move>& moves,const Rules& rules,
                                   const ShortWeights& weights,uint64_t seed,int response,bool recheck) {
  std::cout << '[';
  for (int sample=0;sample<(recheck ? 24 : 20);++sample) {
    const auto common=mix_seed(seed,(recheck ? 31000 : sample<8 ? 11000 : 21000)+sample);
    if (sample) std::cout << ',';
    std::cout << '[';
    for (size_t arm=0;arm<moves.size();++arm) {
      if (arm) std::cout << ',';
      std::cout << review_outcome(review_rollout(point,moves[arm],rules,response,true,weights,common),
                                   point.to_move);
    }
    std::cout << ']';
  }
  std::cout << ']';
}

inline void report_mine_information(const State& initial,const State& point,const MineOffer& offer,
                                     const Rules& rules,const std::vector<Move>& prefix,
                                     const ShortWeights& weights,uint64_t seed,Move played,bool recheck) {
  const auto d=seat_decision(point);
  const auto target=offer.engineer.to;
  const auto moves=mine_information_moves(d,offer,played,weights,seed);
  std::cout << "{\"ply\":" << point.ply << ",\"target\":" << int(target)
            << ",\"legal\":" << d.actions.size() << ",\"engineer\":" << int(offer.engineer.from)
            << ",\"ordinary\":" << int(offer.ordinary.from) << ",\"moves\":[";
  for (size_t i=0;i<moves.size();++i) {
    if (i) std::cout << ',';
    const auto move=moves[i];
    std::cout << '[' << int(move.from) << ',' << int(move.to) << ','
              << (d.observation.seat_at(move.to)==kNone) << ']';
  }
  std::cout << "],\"cells\":[";
  if (const auto other=flip_target_mine(initial,point,target,rules,prefix)) {
    const auto b=seat_decision(*other);
    expect_value(b.actions==d.actions && review_learned(b,weights)==review_learned(d,weights) &&
                 mine_information_moves(b,offer,played,weights,seed)==moves,
                 "compatible mine world changed the public policy");
    const std::array<State,2> worlds=point.rank_at[target]==Mine ?
      std::array<State,2>{point,*other} : std::array<State,2>{*other,point};
    for (size_t world=0;world<worlds.size();++world)
      for (int response : {0,1,2}) {
        if (response==1 && !recheck) continue;
        if (world || response) std::cout << ',';
        mine_information_panel(worlds[world],moves,rules,weights,
                               mix_seed(seed,static_cast<uint64_t>(point.ply)),response,recheck);
      }
  }
  std::cout << "]}";
}

inline void mine_choice_probe(const Rules& rules,uint64_t seed,int learner,int variant,
                              const ShortWeights& weights,MineStudy study=MineStudy::AttackPair) {
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,rules);
  auto state=initial;
  auto streams=review_streams(seed);
  OpeningMarks marks(observe(state,learner),true);
  std::vector<Move> prefix;
  bool found=false;
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant << ",\"case\":";
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto move=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    if (state.to_move==learner && !found) {
      if (const auto offer=mine_offer(d,marks)) {
        if (study==MineStudy::AttackPair)
          report_mine_offer(initial,state,*offer,rules,prefix,weights,seed);
        else report_mine_information(initial,state,*offer,rules,prefix,weights,seed,move,
                                     study==MineStudy::QuietRecheck);
        found=true;
      }
    }
    prefix.push_back(move);
    step_with_public_ranks(state,rules,learner,move,&marks);
  }
  if (!found) std::cout << "null";
  std::cout << ",\"score\":" << review_outcome(state.result,learner) << "}\n";
}
}
