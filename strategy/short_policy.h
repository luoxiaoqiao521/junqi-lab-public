#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include "random.h"
#include "heuristic.h"
#include "../core/rng.h"
namespace jq {
using ShortWeights = std::array<double,6>;
using PublicRanks = std::array<unsigned,kNodes>;

inline int enemy_hq_distance(int viewer, int node, MatchKind match = MatchKind::FourSeats) {
  int best = 100;
  for (int seat=0; seat<kSeats; ++seat)
    if (participates(match,seat) && team_of(seat,match)!=team_of(viewer,match))
      for (int col : {1,3})
        best = std::min(best,distance(node,Board::get().seat_node(seat,5,col)));
  return best;
}

inline ShortWeights short_features(const Observation& o, Move a, bool position=false, const PublicRanks* evidence=nullptr) {
  const double occupied = o.seat_at(a.to) >= 0;
  const double hq = Board::get().node(a.to).kind == NodeKind::HQ;
  const double progress = position ?
    (enemy_hq_distance(o.viewer,a.from,o.match)-enemy_hq_distance(o.viewer,a.to,o.match))/12.0 : 0;
  return {occupied,hq,double(o.rank_at(a.to)==Flag),o.rank_at(a.from)/11.0,occupied*hq,
          evidence ? double(occupied && (*evidence)[a.to]==(1u << Engineer)) : progress};
}

inline std::vector<double> short_action_masses(const Observation& o, const std::vector<Move>& actions,
                                                const ShortWeights& w, bool position=false,
                                                const PublicRanks* evidence=nullptr) {
  std::vector<double> logits;
  for (auto a : actions) {
    const auto f = short_features(o,a,position,evidence);
    logits.push_back(f[0]*w[0] + f[1]*w[1] + f[2]*w[2] + f[3]*w[3] + f[4]*w[4] + f[5]*w[5]);
  }
  const double maximum = *std::max_element(logits.begin(),logits.end());
  for (auto& x : logits) x = std::exp(x-maximum);
  return logits;
}

inline Move short_move(const Observation& o, const std::vector<Move>& actions,
                       const ShortWeights& w, Rng& rng, ShortWeights* gradient = nullptr,
                       bool position=false, const PublicRanks* evidence=nullptr) {
  const auto logits=short_action_masses(o,actions,w,position,evidence);
  double total = 0;
  for (double x : logits) total+=x;
  double sample = static_cast<double>(rng.next() >> 11) * 0x1.0p-53 * total;
  size_t chosen = actions.size()-1;
  for (size_t i=0; i<logits.size(); ++i) {
    sample -= logits[i];
    if (sample < 0) { chosen=i; break; }
  }
  if (gradient) {
    const auto selected = short_features(o, actions[chosen],position,evidence);
    auto estimate = selected;
    for (size_t i=0; i<actions.size(); ++i) {
      const auto f = short_features(o, actions[i],position,evidence);
      for (size_t j=0; j<w.size(); ++j) estimate[j] -= logits[i]/total*f[j];
    }
    for (size_t j=0; j<w.size(); ++j) (*gradient)[j] += estimate[j];
  }
  return actions[chosen];
}
}
