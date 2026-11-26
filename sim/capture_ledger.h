#pragma once
#include "public_threat_evidence.h"
#include "../core/rng.h"

namespace jq {

struct CaptureEvidence {
  int ply=0;
  uint8_t attacker=0, defender=0, attacker_seat=0, defender_seat=0;
  Outcome outcome=Outcome::Move;
  unsigned attacker_ranks=0, defender_ranks=0;
};

enum class CaptureCompatibility { Consistent, Contradicted, Unresolved };

struct CaptureSearch {
  unsigned remaining;
  Rng* random=nullptr;
  std::array<int8_t,kNodes>* opening=nullptr;
  int first_conflict_ply=0;
  bool inventory_conflict=false;
  const std::array<unsigned,kNodes>* initial_pins=nullptr;
};

struct AttributedLosses {
  size_t total=0;
  RankCountRange rank;
};

// A casualty is attributed to its collision counterpart, including mutual loss.
// Rank bounds use the actor's projected domains and the complete victim inventory.
inline AttributedLosses collision_losses(const Observation& view,const OpeningMarks& marks,
    const std::vector<CaptureEvidence>& rows,int counterpart,int victim,int rank) {
  expect_value(counterpart>=0 && counterpart<kSeats && counterpart!=victim,"invalid collision counterpart");
  SeatRankInventory inventory(view,marks,victim);
  size_t count=0,losses=0;
  for (const auto& row:rows) {
    if (row.outcome==Outcome::Move) continue;
    const bool attacker=row.attacker_seat==victim && row.outcome!=Outcome::AttackerWins;
    const bool defender=row.defender_seat==victim && row.outcome!=Outcome::DefenderWins;
    if (!attacker && !defender) continue;
    expect_value(losses<inventory.lost,"attributed casualty is missing from inventory");
    const int other=attacker ? row.defender_seat : row.attacker_seat;
    if (other==counterpart) {
      std::swap(inventory.matching.allowed[inventory.live+count],
                inventory.matching.allowed[inventory.live+losses]);
      ++count;
    }
    ++losses;
  }
  expect_value(losses==inventory.lost,"capture history is missing recorded casualties");
  return {count,category_rank_count(inventory,rank,inventory.live,inventory.live+count)};
}

// Piece keys are assigned from occupied nodes in the first tracked snapshot.
// They come from the public board and follow the piece across movement.
class CaptureLedger {
 public:
  CaptureLedger(const Observation& start,const OpeningMarks& marks) {
    at_.fill(absent);
    owner_.fill(kNone);
    for (int node=0;node<kNodes;++node) if (start.seat_at(node)>=0) {
      at_[node]=static_cast<uint8_t>(node);
      owner_[node]=static_cast<int8_t>(start.seat_at(node));
    }
    remember(start,marks);
  }

  void advance(const Observation& before,const Observation& after,Move move,Outcome outcome,
               const Rules& rules,const OpeningMarks& marks) {
    const auto attacker=at_[move.from],defender=at_[move.to];
    expect_value(attacker!=absent,"capture history lost the moving piece");
    if (outcome!=Outcome::Move) {
      expect_value(defender!=absent && matches_rules(rules),"capture history lacks a victim or changed collision rules");
      rules_=rules;
      const auto a=static_cast<uint8_t>(before.seat_at(move.from));
      const auto b=static_cast<uint8_t>(before.seat_at(move.to));
      events_.push_back({after.ply,attacker,defender,a,b,outcome,0,0});
      if (outcome!=Outcome::AttackerWins) register_loss(a,attacker,marks);
      if (outcome!=Outcome::DefenderWins) register_loss(b,defender,marks);
    }
    at_[move.from]=absent;
    if (outcome==Outcome::Move || outcome==Outcome::AttackerWins) at_[move.to]=attacker;
    if (outcome==Outcome::BothRemoved) at_[move.to]=absent;
    after_control(after);
    remember(after,marks);
  }

  void after_control(const Observation& view) {
    for (int node=0;node<kNodes;++node)
      if (view.seat_at(node)<0) at_[node]=absent;
  }

  void record_flag_resolution(const StepInfo& event,int ply) {
    if (event.outcome!=Outcome::AttackerWins && event.outcome!=Outcome::BothRemoved) return;
    if (Board::get().node(event.move.to).kind!=NodeKind::HQ) return;
    expect_value(!events_.empty() && events_.back().ply==ply,"flag resolution has no matching collision");
    const auto& contact=events_.back();
    const bool captured=event.exit_reason[contact.defender_seat]==ExitReason::FlagRemoved;
    auto& ranks=last_seen_[contact.defender];
    ranks&=captured?(1u<<Flag):all_evidence_ranks&~(1u<<Flag);
    expect_value(ranks!=0,"public flag resolution contradicts the collision's defender");
  }

  void refine(const Observation& view,OpeningMarks& marks,std::vector<CaptureEvidence>* entries) const {
    auto domains=last_seen_;
    read_domains(view,marks,domains);
    while (propagate(domains)) {
      write_domains(view,marks,domains);
      for (int seat=0;seat<kSeats;++seat) constrain_rank_inventory(view,marks,seat);
      read_domains(view,marks,domains);
    }
    if (!entries) return;
    *entries=events_;
    for (auto& event:*entries) {
      event.attacker_ranks=domains[event.attacker];
      event.defender_ranks=domains[event.defender];
    }
  }

  // Pins refer to occupied squares in this observation. Unknown leaves a piece free.
  // A proof covers the stored constraints and makes no probability claim.
  CaptureCompatibility check_candidate(const Observation& view,const OpeningMarks& marks,
                                       const int8_t (&sampled_ranks)[kNodes],unsigned budget) const {
    auto domains=last_seen_;
    read_domains(view,marks,domains);
    for (int node=0;node<kNodes;++node) if (at_[node]!=absent) {
      const int rank=sampled_ranks[node];
      if (rank==Unknown) continue;
      if (rank<Flag || rank>=RankCount) return CaptureCompatibility::Contradicted;
      domains[at_[node]]&=1u<<rank;
    }
    CaptureSearch work{budget};
    return complete(domains,work);
  }

  // The opening view is projected to the same observer. A zero mask leaves a live piece unpinned.
  CaptureCompatibility solve_candidate(const Observation& view,const OpeningMarks& marks,
      const Observation& opening,const std::array<unsigned,kNodes>& pins,CaptureSearch& work) const {
    auto domains=initial_domains(view,marks,opening);
    for (int node=0;node<kNodes;++node) {
      if (at_[node]!=absent && pins[node]) domains[at_[node]]&=pins[node];
      if (work.initial_pins && owner_[node]!=kNone && (*work.initial_pins)[node])
        domains[node]&=(*work.initial_pins)[node];
    }
    return complete(domains,work);
  }

  std::array<unsigned,kNodes> opening_support(const Observation& view,const OpeningMarks& marks,
                                              const Observation& opening) const {
    auto domains=initial_domains(view,marks,opening);
    bool changed=false;
    do {
      changed=false;
      expect_value(narrow(domains,changed,nullptr),"opening domains contradict the current observation");
    } while (changed);
    return domains;
  }

  int opening_node(int node) const {
    return node>=0 && node<kNodes && at_[node]!=absent?at_[node]:kNone;
  }

 private:
  static constexpr uint8_t absent=255;
  using Domains=std::array<unsigned,kNodes>;
  std::array<uint8_t,kNodes> at_{};
  std::array<int8_t,kNodes> owner_{};
  std::array<std::array<uint8_t,kPiecesPerSeat>,kSeats> lost_{};
  std::array<size_t,kSeats> loss_count_{};
  Domains last_seen_{};
  Rules rules_{};
  std::vector<CaptureEvidence> events_;

  Domains initial_domains(const Observation& view,const OpeningMarks& marks,const Observation& opening) const {
    expect_value(opening.ply==0 && opening.viewer==view.viewer,"candidate needs the observer's opening view");
    auto domains=last_seen_;
    read_domains(view,marks,domains);
    for (int node=0;node<kNodes;++node) {
      const auto initial=opening.rank_at(node);
      if (owner_[node]!=kNone && initial!=Unknown) {
        const int seat=opening.seat_at(node);
        expect_value(seat==owner_[node] && (seat==view.viewer ||
                     (initial==Flag && (opening.revealed_flags&(1u<<seat)))),"private opening rank in candidate query");
        domains[node]&=1u<<initial;
      }
    }
    return domains;
  }

  bool fits_inventory(const Domains& domains,CaptureSearch* solution=nullptr) const {
    for (int seat=0;seat<kSeats;++seat) {
      RankAssignment assignment;
      std::array<int,kPiecesPerSeat> keys{};
      size_t pieces=0;
      for (int key=0;key<kNodes;++key) if (owner_[key]==seat) {
        if (!domains[key] || pieces==assignment.allowed.size()) return false;
        keys[pieces]=key;
        assignment.allowed[pieces++]=domains[key];
      }
      if (solution && solution->random)
        for (int i=kPiecesPerSeat-1;i>0;--i)
          std::swap(assignment.order[i],assignment.order[solution->random->below(static_cast<uint32_t>(i+1))]);
      if (!assignment.match(pieces)) return false;
      if (solution && solution->opening)
        for (int i=0;i<kPiecesPerSeat;++i)
          if (assignment.piece_for_slot[i]>=0)
            (*solution->opening)[keys[assignment.piece_for_slot[i]]]=static_cast<int8_t>(assignment.slot_rank[i]);
    }
    return true;
  }

  int collision_split(const Domains& domains) const {
    for (const auto& event:events_) {
      const unsigned attackers=domains[event.attacker], defenders=domains[event.defender];
      for (int a=Bomb;a<RankCount;++a) if (attackers&(1u<<a))
        for (int b=Flag;b<RankCount;++b) if ((defenders&(1u<<b)) && resolve(a,b,rules_)!=event.outcome)
          return (attackers&(attackers-1)) ? event.attacker : event.defender;
    }
    return kNone;
  }

  CaptureCompatibility complete(Domains domains,CaptureSearch& work) const {
    using Result=CaptureCompatibility;
    if (!work.remaining) return Result::Unresolved;
    --work.remaining;
    bool changed=false;
    do {
      changed=false;
      if (!narrow(domains,changed,&work.first_conflict_ply)) return Result::Contradicted;
    } while (changed);
    if (!fits_inventory(domains)) {
      work.inventory_conflict=true;
      return Result::Contradicted;
    }
    const int split=collision_split(domains);
    if (split==kNone) {
      if (work.opening) expect_value(fits_inventory(domains,&work),"capture witness lost its inventory");
      return Result::Consistent;
    }
    const auto choices=domains[split];
    const int start=work.random?static_cast<int>(work.random->below(RankCount)):0;
    for (int offset=0;offset<RankCount;++offset) {
      const int rank=(start+offset)%RankCount;
      if (!(choices&(1u<<rank))) continue;
      domains[split]=1u<<rank;
      const auto result=complete(domains,work);
      if (result!=Result::Contradicted) return result;
    }
    return Result::Contradicted;
  }

  bool matches_rules(const Rules& other) const {
    return events_.empty() || (rules_.bomb_captures_flag==other.bomb_captures_flag &&
      rules_.bomb_mine_both_removed==other.bomb_mine_both_removed && rules_.mine_survives==other.mine_survives);
  }

  void register_loss(int seat,uint8_t piece,const OpeningMarks& marks) {
    expect_value(loss_count_[seat]<kPiecesPerSeat && loss_count_[seat]+1==marks.removed[seat].size,
                 "capture history and observer losses disagree");
    lost_[seat][loss_count_[seat]++]=piece;
  }

  void remember(const Observation& view,const OpeningMarks& marks) {
    for (int node=0;node<kNodes;++node) if (at_[node]!=absent) {
      const unsigned ranks=compatible_rank_mask(view,marks,node);
      expect_value(ranks!=0,"capture history retained an empty piece domain");
      auto& saved=last_seen_[at_[node]];
      saved=saved ? saved&ranks : ranks;
      expect_value(saved!=0,"same piece acquired conflicting public rank evidence");
    }
  }

  void read_domains(const Observation& view,const OpeningMarks& marks,Domains& domains) const {
    for (int node=0;node<kNodes;++node)
      if (at_[node]!=absent) domains[at_[node]]&=compatible_rank_mask(view,marks,node);
    for (int seat=0;seat<kSeats;++seat) {
      expect_value(loss_count_[seat]==marks.removed[seat].size,"capture history inventory length mismatch");
      for (size_t i=0;i<loss_count_[seat];++i) domains[lost_[seat][i]]&=marks.removed[seat].masks[i];
    }
  }

  void write_domains(const Observation& view,OpeningMarks& marks,const Domains& domains) const {
    for (int node=0;node<kNodes;++node)
      if (at_[node]!=absent) {
        expect_value(view.seat_at(node)>=0,"capture history still contains a cleared piece");
        marks.ranks[node]=domains[at_[node]];
      }
    for (int seat=0;seat<kSeats;++seat) {
      RemovedRanks updated;
      for (size_t i=0;i<loss_count_[seat];++i) updated.add(domains[lost_[seat][i]]);
      marks.removed[seat]=updated;
    }
  }

  bool narrow(Domains& domains,bool& changed,int* conflict_ply=nullptr) const {
    for (const auto& event:events_) {
      auto& att=domains[event.attacker];
      auto& def=domains[event.defender];
      const auto [a,b]=supported_collision_ranks(att,def,event.outcome,rules_);
      if (!a || !b) {
        if (conflict_ply && !*conflict_ply) *conflict_ply=event.ply;
        return false;
      }
      changed|=a!=att || b!=def;
      att=a;
      def=b;
    }
    return true;
  }

  bool propagate(Domains& domains) const {
    bool changed=false;
    expect_value(narrow(domains,changed),"past collision conflicts with the observer's rank evidence");
    return changed;
  }
};

}
