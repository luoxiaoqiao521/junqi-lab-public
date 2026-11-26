#pragma once
#include "public_threat_evidence.h"
#include "../strategy/public_policy_value.h"
#include <iomanip>

namespace jq {

// Constructed from a verified opening and advanced only with filtered views
// and publicly observed actions/results/turn events.
class PublicPrefix {
 public:
  explicit PublicPrefix(const Observation& opening)
      : marks_(opening,true) {
    expect_value(opening.ply==0,"public prefix requires an opening");
    history_.complete=true;
    update_exclusions(opening);
  }

  const PublicHistory& history() const { return history_; }

  void advance(const Observation& before, const Observation& after, Move move, Outcome outcome,
               uint8_t skipped, uint8_t departed, const Rules& rules) {
    advance_contact_ranks(marks_,before,after,move,outcome,rules);
    const auto old_evidence=history_.collision[move.from];
    history_.collision[move.from]=0;
    switch (outcome) {
      case Outcome::Move: history_.collision[move.to]=old_evidence; break;
      case Outcome::AttackerWins: history_.collision[move.to]=1; break;
      case Outcome::DefenderWins: history_.collision[move.to]=2; break;
      case Outcome::BothRemoved: history_.collision[move.to]=0; break;
    }
    if (outcome!=Outcome::Move) history_.last_collision=static_cast<uint8_t>(outcome);
    history_.skipped=skipped;
    history_.departed=departed;
    update_exclusions(after);
  }

 private:
  OpeningMarks marks_;
  PublicHistory history_;

  void update_exclusions(const Observation& view) {
    history_.moved=marks_.moved;
    for (int node=0; node<kNodes; ++node) {
      const int owner=view.seat_at(node), rank=view.rank_at(node);
      if (owner<0) history_.collision[node]=0;
      history_.excluded[node]=owner<0 ? 0 : (rank==Unknown ?
          all_evidence_ranks & ~marks_.ranks[node] : all_evidence_ranks & ~(1u<<rank));
    }
  }
};

inline void write_public_prediction(std::ostream& out, const Observation& view, const PublicHistory& history,
                                    const PublicModel& model, const Rules& rules) {
  const auto input=encode_public_decision(view,history,rules.ccw);
  unsigned encodings=0;
  const auto prediction=predict_public(model,input,&encodings);
  out << std::setprecision(std::numeric_limits<float>::max_digits10)
      << "{\"ply\":" << view.ply << ",\"seat\":" << view.viewer << ",\"encodings\":" << encodings
      << ",\"moved\":" << std::count(history.moved.begin(),history.moved.end(),true)
      << ",\"collision\":" << static_cast<int>(history.last_collision)
      << ",\"value\":" << prediction.value << ",\"policy\":[";
  for (size_t i=0; i<input.legal.size(); ++i) {
    const auto move=input.legal[i];
    out << (i ? ",[" : "[") << int(move.from) << ',' << int(move.to) << ',' << prediction.policy[i] << ']';
  }
  out << "]}\n";
}

inline void replay_public_model(const std::string& rules_path, const std::string& model_path,
                                const std::string& log_path, unsigned budget) {
  if (!budget || budget>128) throw std::invalid_argument("public replay decision budget must be 1..128");
  Rules expected;
  std::string error;
  if (!parse_rules(read_text(rules_path),expected,error)) throw std::runtime_error(error);
  const auto model=load_public_model(model_path,rules_hash(expected));
  const auto text=read_text(log_path);
  expect_value(!text.empty() && text.back()=='\n',"truncated log tail");
  std::istringstream input(text);
  std::string line;
  std::getline(input,line);
  const auto header=parse_json(line);
  Rules rules;
  const auto policy=header.object.count("policy")?simulation_policy(header.at("policy").str()):SimulationPolicy{};
  auto state=replay_initial(header,rules);
  expect_value(state.match==MatchKind::FourSeats && rules_hash(rules)==rules_hash(expected),
               "public replay profile mismatch");
  const bool settlement=header.object.count("settlement_detail")!=0;
  if (settlement)
    expect_value(header.at("settlement_detail").boolean() && rules.eliminate_exhausted_movers,
                 "invalid settlement detail profile");
  expect_value(bool(std::getline(input,line)) &&
               dump_json(parse_json(line))==dump_json(parse_json(frame_json(state,nullptr,IdentityVisibility::OwnAndPublic))),
               "initial frame mismatch");
  std::array<PublicPrefix,kSeats> prefix{PublicPrefix(observe(state,0)),PublicPrefix(observe(state,1)),
                                       PublicPrefix(observe(state,2)),PublicPrefix(observe(state,3))};
  std::ostringstream output;
  while (std::getline(input,line)) {
    expect_value(!line.empty() && state.result==Result::Ongoing,"extra/empty event");
    const auto view=observe(state,state.to_move);
    if (budget) write_public_prediction(output,view,prefix[view.viewer].history(),model,rules);
    State before;
    const auto event=replay_frame(parse_json(line),rules,state,before,
                                  IdentityVisibility::OwnAndPublic,policy,false,settlement);
    if (budget && --budget) for (int viewer=0; viewer<kSeats; ++viewer)
      prefix[viewer].advance(observe(before,viewer),observe(state,viewer),event.move,event.outcome,
                             event.skipped_mask,event.eliminated_mask,rules);
  }
  expect_value(state.result!=Result::Ongoing,"unfinished log");
  std::cout << output.str();
}

}  // namespace jq
