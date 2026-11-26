#pragma once
#include <chrono>
#include "learning_probe.h"
#include "replay.h"
#include "replay_deadline.h"
#include "../strategy/short_policy.h"
#include "../strategy/heuristic.h"
namespace jq {
struct FrozenSeat {
  std::string policy;
  ShortWeights weights{};
};

inline Move frozen_seat_move(const SeatDecision& decision, const FrozenSeat& seat, Rng& rng) {
  if (seat.policy=="short") return short_move(decision.observation,decision.actions,seat.weights,rng);
  if (seat.policy=="random")
    return decision.actions.at(rng.below(static_cast<uint32_t>(decision.actions.size())));
  if (seat.policy=="contact") {
    const auto& view=decision.observation;
    std::vector<Move> contacts;
    for (const auto action : decision.actions) {
      const int target=view.seat_at(action.to);
      if (target!=kNone && team_of(target)!=team_of(view.viewer)) contacts.push_back(action);
    }
    if (!contacts.empty() && rng.below(4)==0)
      return contacts.at(rng.below(static_cast<uint32_t>(contacts.size())));
  }
  Move action;
  if (!heuristic_move(decision.observation,rng,action)) throw std::runtime_error("no frozen heuristic move");
  return action;
}

inline void transfer_probe(const Rules& rules, const ShortWeights& weights,
                           uint64_t start=40000, uint64_t count=32,
                           const std::string& policy="short", int variant=0,
                           const FrozenSeat* partner=nullptr, const FrozenSeat* opponents=nullptr,
                           const FrozenSeat* later_opponent=nullptr) {
  const bool final_flag=policy=="short-final-flag";
  for (uint64_t seed=start; seed<start+count; ++seed) {
    for (int learner=0; learner<4; ++learner) {
      Rng layout_rng(seed);
      int8_t layouts[kSeats][kPiecesPerSeat];
      for (auto& layout : layouts) generate_layout(layout,layout_rng);
      const int first = rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
      State s; init_state(s,layouts,first,rules);
      std::array<Rng,4> streams{Rng(mix_seed(seed,10)),Rng(mix_seed(seed,11)),
                                Rng(mix_seed(seed,12)),Rng(mix_seed(seed,13))};
      int decisions=0, opportunities=0, evidence_contacts=0;
      int flag_chances=0, flag_overrides=0;
      std::optional<OpeningMarks> marks;
      if (policy=="evidence") marks.emplace(observe(s,learner),true);
      uint64_t learner_ns=0;
      while (s.result==Result::Ongoing) {
        const auto d=seat_decision(s);
        auto& rng=streams[s.to_move];
        Move action;
        if (s.to_move==learner) {
          ++decisions;
          if (marks)
            evidence_contacts += std::any_of(d.actions.begin(),d.actions.end(),[&](Move action) {
              return short_features(d.observation,action,false,&marks->ranks)[5] != 0;
            });
          bool opportunity=false;
          for (auto a : d.actions) {
            State branch=s; step(branch,a,rules);
            if (team_score(branch.result,learner)==1.0) opportunity=true;
          }
          opportunities += opportunity;
        }
        const bool learned = s.to_move==learner &&
          (policy=="short" || final_flag || policy=="position" || policy=="evidence");
        const bool heuristic = (s.to_move==learner && policy=="heuristic") ||
          (variant==1 && s.to_move==(learner+2)%4) ||
          (variant==2 && (s.to_move%2)!=(learner%2));
        const auto began=std::chrono::steady_clock::now();
        const int later_seat=(learner+(rules.ccw ? 3 : 1))%4;
        const auto* frozen = later_opponent && s.to_move==later_seat ? later_opponent :
          s.to_move==learner ? nullptr :
          team_of(s.to_move)==team_of(learner) ? partner : opponents;
        if (frozen) action=frozen_seat_move(d,*frozen,rng);
        else if (learned) action=short_move(d.observation,d.actions,weights,rng,nullptr,policy=="position",
                                       marks ? &marks->ranks : nullptr);
        else if (heuristic) {
          if (!heuristic_move(d.observation,rng,action)) throw std::runtime_error("no heuristic move");
        } else action=d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
        if (final_flag && s.to_move==learner) {
          if (const auto win=public_flag_finish(d,action,rules)) {
            ++flag_chances;
            flag_overrides+=!(action==*win);
            action=*win;
          }
        }
        if (s.to_move==learner)
          learner_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-began).count());
        step_with_public_ranks(s,rules,learner,action,marks ? &*marks : nullptr);
        expect_value(!flag_chances || team_score(s.result,learner)==1.,"public flag recommendation did not end in a win");
      }
      std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner
                << ",\"first\":" << first << ",\"plies\":" << s.ply
                << ",\"decisions\":" << decisions << ",\"opportunities\":" << opportunities
                << (marks ? ",\"evidence_contacts\":" + std::to_string(evidence_contacts) : "")
                << (final_flag ? ",\"public_flag_chances\":" + std::to_string(flag_chances) +
                     ",\"public_flag_overrides\":" + std::to_string(flag_overrides) : "")
                << ",\"learner_ns\":" << learner_ns
                << ",\"score\":";
      const auto score=team_score(s.result,learner);
      if (score) std::cout << *score; else std::cout << "null";
      std::cout << "}\n";
    }
  }
}
}
