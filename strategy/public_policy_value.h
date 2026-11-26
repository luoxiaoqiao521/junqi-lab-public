#pragma once
#include "../core/moves.h"
#include "../core/rng.h"
#include "../observe/observation.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jq {

// v1: 32 channels per rotated node, followed by 22 public global fields.
constexpr int pv_node_width = 4 + 1 + RankCount + 1 + 2 + RankCount;
constexpr int pv_width = kNodes * pv_node_width + 22;
constexpr int pv_action_width = 10, pv_first = 24, pv_hidden = 12;
constexpr uint64_t pv_limit = 10000000;
constexpr char pv_magic[] = "JQPV0001";

struct PublicHistory {
  bool complete = false;
  std::array<bool, kNodes> moved{};
  std::array<uint8_t, kNodes> collision{};
  std::array<unsigned, kNodes> excluded{};
  uint8_t last_collision = 0, skipped = 0, departed = 0;
};

struct PublicDecision {
  std::array<float, pv_width> board{};
  std::vector<Move> legal;
  std::vector<std::array<float, pv_action_width>> action;
};

inline int relative_seat(int seat, int viewer) { return (seat + kSeats - viewer) % kSeats; }

inline int pv_node(int node, int viewer) {
  const auto& board = Board::get();
  const auto& point = board.node(node);
  if (point.seat >= 0) return board.seat_node(relative_seat(point.seat,viewer),point.r,point.c);
  int row = point.gr, col = point.gc;
  for (int rotation=0; rotation<viewer; ++rotation) {
    const int new_col = 16-row;
    row = col;
    col = new_col;
  }
  return board.at_grid(row,col);
}

inline auto pv_action(const Observation& view, Move move) {
  const auto& board = Board::get();
  const auto& from = board.node(pv_node(move.from,view.viewer));
  const auto& to = board.node(pv_node(move.to,view.viewer));
  return std::array<float,pv_action_width>{
      1.f, from.gr/16.f, from.gc/16.f, to.gr/16.f, to.gc/16.f,
      float(view.seat_at(move.to)>=0), float(from.kind==NodeKind::Camp),
      float(to.kind==NodeKind::Camp), float(view.rank_at(move.from)==Engineer), float(to.rail)};
}

inline PublicDecision encode_public_decision(const Observation& view, const PublicHistory& history,
                                             bool ccw) {
  if (view.match != MatchKind::FourSeats || view.viewer < 0 || view.viewer >= kSeats ||
      view.turn < 0 || view.turn >= kSeats) throw std::invalid_argument("four-dark decision required");
  PublicDecision input;
  auto* global = input.board.data() + kNodes*pv_node_width;
  global[4+relative_seat(view.turn,view.viewer)] = 1.f;
  global[ccw ? 8 : 9] = 1.f;
  global[21] = float(history.complete);
  if (history.complete) {
    for (int owner=0; owner<kSeats; ++owner) {
      const int seat = relative_seat(owner,view.viewer);
      global[10+seat] = float((history.skipped>>owner)&1u);
      global[14+seat] = float((history.departed>>owner)&1u);
    }
    if (history.last_collision > 3) throw std::invalid_argument("invalid public collision");
    if (history.last_collision) global[17+history.last_collision] = 1.f;
  }
  for (int node=0; node<kNodes; ++node) {
    const int owner = view.seat_at(node), rank = view.rank_at(node);
    if (owner < 0) continue;
    if (owner >= kSeats || rank < Unknown || rank >= RankCount ||
        (owner != view.viewer && rank != Unknown && rank != Flag))
      throw std::invalid_argument("observation rank/owner outside four-dark visibility");
    auto* channels = input.board.data() + pv_node(node,view.viewer)*pv_node_width;
    const int seat = relative_seat(owner,view.viewer);
    channels[seat] = global[seat] = 1.f;
    channels[rank == Unknown ? 4 : 5+rank] = 1.f;
    if (!history.complete) continue;
    channels[17] = float(history.moved[node]);
    const auto collision = history.collision[node];
    if (collision > 2) throw std::invalid_argument("invalid public survivor evidence");
    if (collision) channels[17+collision] = 1.f;
    for (int r=0; r<RankCount; ++r) channels[20+r] = float((history.excluded[node]>>r)&1u);
  }
  if (view.viewer == view.turn) gen_moves(view,view.viewer,input.legal);
  for (const auto move : input.legal) input.action.push_back(pv_action(view,move));
  return input;
}

constexpr int pv_w1 = 0, pv_b1 = pv_width*pv_first;
constexpr int pv_w2 = pv_b1+pv_first, pv_b2 = pv_w2+pv_first*pv_hidden;
constexpr int pv_policy = pv_b2+pv_hidden, pv_policy_bias = pv_policy+pv_hidden*pv_action_width;
constexpr int pv_value = pv_policy_bias+pv_action_width, pv_value_bias = pv_value+pv_hidden;
constexpr int pv_parameters = pv_value_bias+1;

struct PublicModel {
  std::vector<float> weights = std::vector<float>(pv_parameters);
};

struct PublicActivation {
  std::array<float,pv_first> first{};
  std::array<float,pv_hidden> shared{};
};

inline PublicActivation public_encode(const PublicModel& model, const PublicDecision& input,
                                      unsigned* encoding_calls=nullptr) {
  if (encoding_calls) ++*encoding_calls;
  const auto& w = model.weights;
  PublicActivation output;
  for (int j=0; j<pv_first; ++j) {
    float sum=w[pv_b1+j];
    for (int i=0; i<pv_width; ++i) if (input.board[i]) sum += input.board[i]*w[pv_w1+j*pv_width+i];
    if (!std::isfinite(sum)) throw std::runtime_error("non-finite public input projection");
    output.first[j]=std::tanh(sum);
  }
  for (int j=0; j<pv_hidden; ++j) {
    float sum=w[pv_b2+j];
    for (int i=0; i<pv_first; ++i) sum += output.first[i]*w[pv_w2+j*pv_first+i];
    if (!std::isfinite(sum)) throw std::runtime_error("non-finite public shared projection");
    output.shared[j]=std::tanh(sum);
  }
  return output;
}

inline std::vector<float> public_distribution(const PublicDecision& input,
                                               const std::array<float,pv_action_width>& context) {
  std::vector<float> p;
  for (const auto& action : input.action) {
    float score = 0.f;
    for (int i=0; i<pv_action_width; ++i) score += context[i]*action[i];
    if (!std::isfinite(score)) throw std::runtime_error("non-finite public policy logit");
    p.push_back(score);
  }
  if (p.empty()) return p;
  const float maximum = *std::max_element(p.begin(),p.end());
  float total=0.f;
  for (auto& probability : p) { probability=std::exp(probability-maximum); total+=probability; }
  for (auto& probability : p) probability/=total;
  return p;
}

struct PublicPrediction {
  std::vector<float> policy;
  float value = 0.f;
  float probability(const PublicDecision& input, Move move) const {
    const auto i=std::find(input.legal.begin(),input.legal.end(),move);
    return i==input.legal.end() ? 0.f : policy[static_cast<size_t>(i-input.legal.begin())];
  }
};

inline PublicPrediction public_heads(const PublicModel& model, const PublicDecision& input,
                                     const PublicActivation& encoded) {
  const auto& w = model.weights;
  std::array<float,pv_action_width> context{};
  // A constant action component cancels under softmax; the other components
  // interact with the board-dependent context and can change move ordering.
  for (int i=0; i<pv_action_width; ++i) {
    context[i]=w[pv_policy_bias+i];
    for (int j=0; j<pv_hidden; ++j) context[i]+=w[pv_policy+i*pv_hidden+j]*encoded.shared[j];
    if (!std::isfinite(context[i])) throw std::runtime_error("non-finite public policy projection");
  }
  float value=w[pv_value_bias];
  for (int j=0; j<pv_hidden; ++j) value+=w[pv_value+j]*encoded.shared[j];
  if (!std::isfinite(value)) throw std::runtime_error("non-finite public value logit");
  return {public_distribution(input,context),1.f/(1.f+std::exp(-value))};
}

inline PublicPrediction predict_public(const PublicModel& model, const PublicDecision& input,
                                       unsigned* encoding_calls=nullptr) {
  return public_heads(model,input,public_encode(model,input,encoding_calls));
}

struct PublicExample {
  PublicDecision input;
  size_t preferred = 0;
  float team_return = 0.f;
  bool settled = false;
};

inline PublicModel initialize_public_model(uint64_t seed) {
  PublicModel model;
  Rng rng(seed);
  for (auto& weight : model.weights)
    weight=(static_cast<int>(rng.below(20001))-10000)*0.00002f;
  return model;
}

inline auto public_policy_error(const PublicExample& row, const std::vector<float>& probabilities) {
  std::array<float,pv_action_width> gradient{};
  for (size_t i=0; i<probabilities.size(); ++i)
    for (int j=1; j<pv_action_width; ++j)
      gradient[j]+=(probabilities[i]-float(i==row.preferred))*row.input.action[i][j];
  return gradient;
}

// SGD on categorical cross-entropy and half squared team-return error.
// A truncated pair has no policy target and contributes no gradient to either head.
inline bool update_public_model(PublicModel& model, const PublicExample& row, float rate) {
  if (!row.settled) return false;
  if (row.preferred>=row.input.legal.size() || row.team_return<0.f || row.team_return>1.f ||
      !std::isfinite(row.team_return) || !std::isfinite(rate) || rate<=0.f)
    throw std::invalid_argument("invalid public training target or rate");
  const auto hidden=public_encode(model,row.input);
  const auto prediction=public_heads(model,row.input,hidden);
  const auto action_error=public_policy_error(row,prediction.policy);
  const float v=prediction.value, value_error=(v-row.team_return)*v*(1.f-v);
  std::array<float,pv_first> first_error{};
  auto& w=model.weights;
  w[pv_value_bias]-=rate*value_error;
  for (int i=1; i<pv_action_width; ++i) w[pv_policy_bias+i]-=rate*action_error[i];
  for (int j=0; j<pv_hidden; ++j) {
    float error=value_error*w[pv_value+j];
    w[pv_value+j]-=rate*value_error*hidden.shared[j];
    for (int i=1; i<pv_action_width; ++i) {
      auto& weight=w[pv_policy+i*pv_hidden+j];
      error+=action_error[i]*weight;
      weight-=rate*action_error[i]*hidden.shared[j];
    }
    error*=1.f-hidden.shared[j]*hidden.shared[j];
    w[pv_b2+j]-=rate*error;
    for (int i=0; i<pv_first; ++i) {
      auto& weight=w[pv_w2+j*pv_first+i];
      first_error[i]+=error*weight;
      weight-=rate*error*hidden.first[i];
    }
  }
  for (int j=0; j<pv_first; ++j) {
    const float change=rate*first_error[j]*(1.f-hidden.first[j]*hidden.first[j]);
    w[pv_b1+j]-=change;
    for (int i=0; i<pv_width; ++i) if (row.input.board[i])
      w[pv_w1+j*pv_width+i]-=change*row.input.board[i];
  }
  return true;
}

inline void pv_write_u32(std::ostream& out, uint32_t word) {
  for (int shift=0; shift<32; shift+=8) out.put(static_cast<char>((word>>shift)&255u));
}

inline uint32_t pv_read_u32(std::istream& in) {
  uint32_t word=0;
  for (int shift=0; shift<32; shift+=8) {
    const auto byte=in.get();
    if (byte==std::char_traits<char>::eof()) throw std::runtime_error("truncated public model");
    word|=static_cast<uint32_t>(byte)<<shift;
  }
  return word;
}

inline void save_public_model(const std::filesystem::path& path, const PublicModel& model, uint64_t rules) {
  static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"requires IEEE float32");
  if (model.weights.size()!=pv_parameters ||
      !std::all_of(model.weights.begin(),model.weights.end(),[](float w){return std::isfinite(w);}))
    throw std::invalid_argument("invalid public model parameters");
  std::ofstream out(path,std::ios::binary);
  out.write(pv_magic,8);
  for (const auto field : {pv_width,pv_first,pv_hidden,pv_action_width,pv_parameters})
    pv_write_u32(out,static_cast<uint32_t>(field));
  pv_write_u32(out,static_cast<uint32_t>(rules));
  pv_write_u32(out,static_cast<uint32_t>(rules>>32));
  for (const auto weight : model.weights) {
    uint32_t bits;
    std::memcpy(&bits,&weight,sizeof bits);
    pv_write_u32(out,bits);
  }
  out.close();
  if (!out) throw std::runtime_error("cannot write public model");
}

inline PublicModel load_public_model(const std::filesystem::path& path, uint64_t rules) {
  if (std::filesystem::file_size(path)>pv_limit) throw std::runtime_error("public model byte budget exceeded");
  std::ifstream in(path,std::ios::binary);
  char magic[8]{};
  in.read(magic,sizeof magic);
  if (!in || std::memcmp(magic,pv_magic,8)!=0) throw std::runtime_error("public model encoding mismatch");
  for (const auto field : {pv_width,pv_first,pv_hidden,pv_action_width,pv_parameters})
    if (pv_read_u32(in)!=static_cast<uint32_t>(field)) throw std::runtime_error("public model shape mismatch");
  const uint64_t low=pv_read_u32(in), high=pv_read_u32(in);
  if ((low|(high<<32))!=rules) throw std::runtime_error("public model rules mismatch");
  PublicModel model;
  for (auto& weight : model.weights) {
    const auto bits=pv_read_u32(in);
    std::memcpy(&weight,&bits,sizeof weight);
    if (!std::isfinite(weight)) throw std::runtime_error("non-finite public model weight");
  }
  if (in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("extra public model bytes");
  return model;
}

}  // namespace jq
