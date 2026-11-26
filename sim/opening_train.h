#pragma once
#include <iomanip>
#include "transfer_probe.h"
#include "settlement.h"

namespace jq {
struct OpeningEpisode {
  std::optional<double> score;
  ShortWeights gradient{};
  int decisions = 0;
  bool learner_out = false;
  TrainingSettlement settlement;
  bool first_contact=false, first_hq_pressure=false;
};

inline bool hq_pressure_contact(const Observation& view, Move move) {
  if (view.seat_at(move.to)<0) return false;
  const auto& target=Board::get().node(move.to);
  return target.seat>=0 && view.team(target.seat)!=view.team(view.viewer) &&
      (target.kind==NodeKind::HQ || (target.r==4 && (target.c==1 || target.c==3)));
}

inline OpeningEpisode continue_opening(State state, const Rules& rules, uint64_t seed, int learner,
                                       const ShortWeights& weights, bool position=false,
                                       bool evidence=false, int context=0) {
  std::array<Rng,4> streams{Rng(mix_seed(seed,10)),Rng(mix_seed(seed,11)),
                            Rng(mix_seed(seed,12)),Rng(mix_seed(seed,13))};
  OpeningEpisode episode;
  std::optional<OpeningMarks> marks;
  const int contact_seat=context==4 ? (learner+(rules.ccw ? 1 : 3))%4 :
                         context==5 ? (learner+(rules.ccw ? 3 : 1))%4 : kNone;
  const FrozenSeat contact{"contact",{}};
  if (evidence) marks.emplace(observe(state,learner),true);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    auto& rng=streams[d.observation.viewer];
    Move action;
    if (d.observation.viewer==learner) {
      ++episode.decisions;
      action=short_move(d.observation,d.actions,weights,rng,&episode.gradient,position,
                        marks ? &marks->ranks : nullptr);
      if (episode.decisions==1) {
        episode.first_contact=d.observation.seat_at(action.to)>=0;
        episode.first_hq_pressure=hq_pressure_contact(d.observation,action);
      }
    } else if (d.observation.viewer==contact_seat) {
      action=frozen_seat_move(d,contact,rng);
    } else if ((context==1 && d.observation.viewer==(learner+2)%4) ||
               (context==2 && team_of(d.observation.viewer)!=team_of(learner))) {
      if (!heuristic_move(d.observation,rng,action)) throw std::runtime_error("no frozen response");
    } else action=d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
    const auto event=step_with_public_ranks(state,rules,learner,action,marks ? &*marks : nullptr);
    if (rules.eliminate_exhausted_movers) episode.settlement.record(event,state.ply);
  }
  episode.score = team_score(state.result,learner);
  episode.learner_out = state.out[learner];
  return episode;
}

inline OpeningEpisode opening_episode(const Rules& rules, uint64_t seed, int learner,
                                      const ShortWeights& weights, bool position=false,
                                      bool evidence=false, int context=0) {
  Rng layouts_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layouts_rng);
  const int first = rules.random_first_seat ? int(layouts_rng.below(kSeats)) : 0;
  State state;
  init_state(state,layouts,first,rules);
  return continue_opening(state,rules,seed,learner,weights,position,evidence,context);
}

inline bool add_team_return(ShortWeights& batch, const OpeningEpisode& episode, double baseline=0) {
  if (!episode.score || !episode.decisions) return false;
  const double centered = 2 * *episode.score - 1 - baseline;
  for (size_t j=0; j<batch.size(); ++j) batch[j] += centered * episode.gradient[j];
  return true;
}

inline void check_position_fixture(const Rules& rules, uint64_t seed) {
  Rng rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,rng);
  State state;
  init_state(state,layouts,0,rules);
  const auto d = seat_decision(state);
  for (const auto a : d.actions)
    for (const auto b : d.actions)
      if (a.from == b.from && a.to != b.to &&
          d.observation.seat_at(a.to)<0 && d.observation.seat_at(b.to)<0 &&
          short_features(d.observation,a)==short_features(d.observation,b) &&
          short_features(d.observation,a,true)!=short_features(d.observation,b,true)) {
        std::cout << "{\"seed\":" << seed << ",\"from\":" << int(a.from)
                  << ",\"targets\":[" << int(a.to) << ',' << int(b.to)
                  << "],\"position_values\":[" << short_features(d.observation,a,true)[5]
                  << ',' << short_features(d.observation,b,true)[5] << "]}";
        return;
      }
  throw std::runtime_error("opening fixture lacks distinguishable routes");
}

inline int opening_context(int profile, int index) {
  if (profile==3) return index%3;
  if (profile==4) {
    // The learner seat is index % 4. Vary context across rounds of all four seats.
    const int rotation[4]={0,4,0,5};
    return rotation[(index/4)%4];
  }
  return profile;
}

inline void print_opening_weights(const ShortWeights& weights, size_t width) {
  std::cout << std::setprecision(17);
  for (size_t index=0; index<width; ++index) {
    if (!std::isfinite(weights[index])) throw std::runtime_error("nonfinite training weight");
    std::cout << (index ? "," : "") << weights[index];
  }
  std::cout << ']';
}

inline void print_opening_contexts(const std::array<int,6>& contexts) {
  std::cout << ",\"contexts\":[";
  for (size_t index=0; index<contexts.size(); ++index)
    std::cout << (index ? "," : "") << contexts[index];
  std::cout << ']';
}

struct OpeningCounts {
  int natural=0, truncated=0, no_decisions=0, decisions=0, delayed_wins=0;
  int wins=0, draws=0, losses=0, first_contacts=0, first_pressure=0;

  void record(const OpeningEpisode& episode) {
    decisions+=episode.decisions;
    first_contacts+=episode.first_contact;
    first_pressure+=episode.first_hq_pressure;
    if (!episode.score) { ++truncated; return; }
    ++natural;
    wins+=episode.score==1.0; draws+=episode.score==.5; losses+=episode.score==0.0;
    no_decisions+=episode.decisions==0;
    delayed_wins+=episode.learner_out && episode.decisions>1 && episode.score==1.0;
  }

  void print() const {
    std::cout << "{\"natural\":" << natural << ",\"truncated\":" << truncated
              << ",\"no_decisions\":" << no_decisions << ",\"decisions\":" << decisions
              << ",\"delayed_wins\":" << delayed_wins << ",\"weights\":[";
  }
};

inline void train_openings(const Rules& rules, ShortWeights weights, uint64_t first_seed=50000,
                           bool position=false, bool verify_fixture=false, bool evidence=false,
                           double rate=.015, int batches=32, bool return_baseline=false,
                           bool update_report=false, int training_profile=0,
                           const std::vector<State>* archive_starts=nullptr) {
  OpeningCounts counts;
  double baseline=0, average_gradient_variance=0;
  int scored_with_decisions=0;
  TrainingSettlement settlement;
  std::array<int,6> contexts{};
  for (int batch=0; batch<batches; ++batch) {
    ShortWeights gradient{};
    double return_sum=0, squared_gradient=0;
    int usable=0;
    for (int sample=0; sample<16; ++sample) {
      const int index=batch*16+sample;
      const int context=opening_context(training_profile,index);
      ++contexts[context];
      const auto seed=first_seed+uint64_t(index);
      const auto* root=archive_starts ? &(*archive_starts)[index%archive_starts->size()] : nullptr;
      const int learner=root ? root->to_move : index%4;
      const auto e=root ? continue_opening(*root,rules,seed,learner,weights,false,false,context)
                        : opening_episode(rules,seed,learner,weights,position,evidence,context);
      if (rules.eliminate_exhausted_movers) settlement.merge(e.settlement,seed,learner);
      counts.record(e);
      ShortWeights sample_gradient{};
      if (add_team_return(sample_gradient,e,baseline)) {
        ++scored_with_decisions;
        ++usable; return_sum += 2 * *e.score - 1;
        for (size_t j=0; j<weights.size(); ++j) {
          gradient[j] += sample_gradient[j];
          squared_gradient += sample_gradient[j]*sample_gradient[j];
        }
      }
    }
    if (usable)
      for (size_t j=0; j<weights.size(); ++j) {
        const double mean=gradient[j]/usable;
        weights[j] += rate*gradient[j]/usable;
        squared_gradient -= usable*mean*mean;
      }
    if (usable) average_gradient_variance += std::max(0.0,squared_gradient/usable)/batches;
    if (return_baseline && usable) baseline=.9*baseline+.1*return_sum/usable;
  }
  counts.print();
  print_opening_weights(weights,(position || evidence) ? weights.size() : weights.size()-1);
  if (update_report)
    std::cout << ",\"rate\":" << rate << ",\"batches\":" << batches
              << ",\"return_baseline\":" << (return_baseline ? "true" : "false")
              << ",\"gradient_variance_proxy\":" << average_gradient_variance
              << ",\"training_profile\":" << training_profile << ",\"wins\":" << counts.wins
              << ",\"draws\":" << counts.draws << ",\"losses\":" << counts.losses;
  if (training_profile==4) print_opening_contexts(contexts);
  if (rules.eliminate_exhausted_movers) settlement.print(std::cout,scored_with_decisions);
  if (archive_starts) std::cout << ",\"first_contacts\":" << counts.first_contacts
                               << ",\"first_hq_pressure\":" << counts.first_pressure;
  if (verify_fixture) {
    std::cout << ",\"position_fixture\":";
    check_position_fixture(rules,first_seed);
  }
  std::cout << "}\n";
}
}
