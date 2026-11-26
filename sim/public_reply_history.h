#pragma once
#include "capture_ledger.h"

namespace jq {

// The shared prefix is built from public observations and recorded events.
// Never seed another seat with the actor's privately inferred ranks.
class PublicReplyHistory {
 public:
  explicit PublicReplyHistory(const State& start, bool complete_opening, bool track_captures=false)
      : view_(observe(start,kNone)), marks_(view_,complete_opening) {
    if (track_captures) captures_.emplace(view_,marks_);
  }

  const OpeningMarks& marks() const { return marks_; }
  const std::array<Move,kSeats>& last_moves() const { return last_; }

  void advance(const State& before,const State& after,const StepInfo& event,const Rules& rules,
               std::optional<bool> flag_capture=std::nullopt) {
    const auto prior=observe(before,kNone);
    expect_value(same_observation(prior,view_) && after.ply==before.ply+1 && event.seat==before.to_move,
                 "reply history is not the predecessor of this event");
    const auto next=observe(after,kNone);
    advance_contact_ranks(marks_,prior,next,event.move,event.outcome,rules,flag_capture);
    if (captures_) captures_->advance(prior,next,event.move,event.outcome,rules,marks_);
    last_[event.seat]=event.move;
    view_=next;
  }

  // An archive control may rotate seats or clear pieces without a move.
  void after_control(const State& state) {
    const auto next=observe(state,kNone);
    expect_value(next.ply==view_.ply && next.quiet_clock==view_.quiet_clock &&
                 next.revealed_flags==view_.revealed_flags && next.match==view_.match,
                 "archive control changed reply history counters");
    for (int node=0;node<kNodes;++node) {
      if (next.seat_at(node)<0) {
        marks_.ranks[node]=0;
        marks_.moved[node]=false;
      } else {
        expect_value(next.seat_at(node)==view_.seat_at(node) && next.rank_at(node)==view_.rank_at(node),
                     "archive control added or changed a piece");
      }
    }
    if (captures_) captures_->after_control(next);
    view_=next;
  }

  // The caller provides the verified public result after all controls on this move.
  void record_flag_resolution(const StepInfo& event) {
    if (captures_) captures_->record_flag_resolution(event,view_.ply);
  }

  OpeningMarks for_actor(Observation view,const OpeningMarks& actor,
                         std::vector<CaptureEvidence>* capture_history=nullptr) const {
    expect_value(view.viewer==view.turn,"reply history requires the current actor's view");
    OpeningMarks evidence=marks_;
    for (int node=0;node<kNodes;++node) {
      const int seat=view.seat_at(node);
      if (seat<0) continue;
      evidence.ranks[node]&=compatible_rank_mask(view,actor,node);
      expect_value(evidence.ranks[node]!=0,"actor evidence contradicts the shared reply history");
      const bool public_flag=view.rank_at(node)==Flag && (view.revealed_flags&(1u<<seat));
      expect_value(view.rank_at(node)==Unknown || seat==view.viewer || public_flag,
                   "reply history received another seat's private rank");
      if (!public_flag) view.ranks[node]=Unknown;
    }
    view.viewer=kNone;
    expect_value(same_observation(view,view_),"reply history does not match this decision");
    for (int seat=0;seat<kSeats;++seat) {
      const auto& shared_losses=marks_.removed[seat];
      const auto& actor_losses=actor.removed[seat];
      expect_value(shared_losses.size==actor_losses.size,"reply history and actor losses have different lengths");
      RemovedRanks combined;
      for (size_t i=0;i<shared_losses.size;++i) combined.add(shared_losses.masks[i]&actor_losses.masks[i]);
      evidence.removed[seat]=combined;
      constrain_rank_inventory(view,evidence,seat);
    }
    if (captures_) captures_->refine(view,evidence,capture_history);
    else expect_value(!capture_history,"capture tracking must start with the replay prefix");
    return evidence;
  }

  CaptureCompatibility check_candidate(const Observation& view,const OpeningMarks& actor,
                                       const int8_t (&sampled_ranks)[kNodes],unsigned budget) const {
    expect_value(captures_.has_value(),"candidate proof requires capture history");
    const auto evidence=for_actor(view,actor);
    return captures_->check_candidate(view,evidence,sampled_ranks,budget);
  }

  CaptureCompatibility solve_candidate(const Observation& view,const OpeningMarks& actor,
      const Observation& opening,const std::array<unsigned,kNodes>& pins,CaptureSearch& work) const {
    expect_value(captures_.has_value(),"candidate construction requires capture history");
    const auto evidence=for_actor(view,actor);
    return captures_->solve_candidate(view,evidence,opening,pins,work);
  }

  std::array<unsigned,kNodes> opening_support(const Observation& view,const OpeningMarks& actor,
                                              const Observation& opening) const {
    expect_value(captures_.has_value(),"opening support requires capture history");
    return captures_->opening_support(view,for_actor(view,actor),opening);
  }

  int opening_node(int now) const {
    expect_value(captures_.has_value(),"piece origin requires capture history");
    return captures_->opening_node(now);
  }

 private:
  Observation view_;
  OpeningMarks marks_;
  std::array<Move,kSeats> last_{};
  std::optional<CaptureLedger> captures_;
};
}
