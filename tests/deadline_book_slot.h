#pragma once

#include <bitset>

// A preparatory move is worth nominating if it creates a concrete legal
// continuation in the opening player's information. Replies are handled
// later by the unchanged scenario evaluator.
class BookPreparation {
 public:
  BookPreparation(const jq::SeatDecision& decision,const jq::OpeningMarks& marks,
                  const std::vector<int>& flags) {
    using namespace jq;
    const auto& view=decision.observation;
    if (threatened_friendly_flag(view,marks)) return;
    const auto& board=Board::get();
    int count[kSeats]{}, last[kSeats]{};
    for (int n=0;n<kNodes;++n) {
      const int seat=view.seat_at(n);
      if (seat<0 || view.team(seat)==view.team(view.viewer) || !can_leave_node(Engineer,n) ||
          !(compatible_rank_mask(view,marks,n)&~immobile_ranks)) continue;
      ++count[seat];
      last[seat]=n;
    }
    for (int seat=0;seat<kSeats;++seat) if (count[seat]==1) goals_[last[seat]]=3;
    for (const int flag:flags) {
      goals_[flag]=4;
      for (const int gate:board.road(flag))
        if (view.seat_at(gate)>=0 && view.team(view.seat_at(gate))!=view.team(view.viewer))
          goals_[gate]=std::max(goals_[gate],2);
    }
    for(const auto action:decision.actions)
      if (view.seat_at(action.to)>=0) known_[action.from].set(action.to);
  }

  int score(const jq::Observation& view,const jq::OpeningMarks& marks,const jq::Rules& rules,
            jq::Move prepare,const std::vector<jq::Move>& followups) const {
    using namespace jq;
    int best=0;
    for (const auto reply:followups) {
      const int goal=goals_[reply.to];
      if (!goal || known_[prepare.from][reply.to] || view.seat_at(reply.to)<0) continue;
      const unsigned possible=collision_outcome_mask(view.rank_at(prepare.from),
          compatible_rank_mask(view,marks,reply.to),rules);
      const unsigned survives=outcome_bit(Outcome::AttackerWins);
      const unsigned removes=survives|(goal>=3?outcome_bit(Outcome::BothRemoved):0);
      if (possible&removes) best=std::max(best,4*goal+2*int((possible&~removes)==0));
    }
    return best;
  }

 private:
  std::array<int,jq::kNodes> goals_{};
  std::array<std::bitset<jq::kNodes>,jq::kNodes> known_{};
};

static void book_preparation_fixtures(const jq::Rules& rules) {
  using namespace jq;
  auto root=selfplay_fixture();
  place_single_case_piece(root,1,50,Marshal);
  place_single_case_piece(root,2,4,Engineer);
  place_single_case_piece(root,26,46,Bomb);
  place_single_case_piece(root,27,33,Platoon);
  const Move prepare{50,51};
  auto decide=seat_decision(root);
  auto marks=OpeningMarks(decide.observation,false);
  const auto score=[&](const SeatDecision& d) {
    const BookPreparation support(d,marks,flag_targets(d.observation,marks));
    const auto after=project_quiet(d.observation,prepare);
    std::vector<Move> followups;
    gen_piece_moves(after,0,prepare.to,followups);
    return support.score(d.observation,marks,rules,prepare,followups);
  };
  single_case_check(is_legal(root,prepare) && score(decide)>0,"valid flag approach was not nominated");
  auto answered=root;
  step(answered,prepare,rules);
  const Move reply{46,51};
  single_case_check(is_legal(answered,reply) && step(answered,reply,rules).outcome==Outcome::BothRemoved &&
                    !answered.alive[1],"enemy reply did not refute the prepared continuation");
  place_single_case_piece(root,28,25,Platoon);
  decide=seat_decision(root);
  marks=OpeningMarks(decide.observation,false);
  single_case_check(score(decide)==0,"free attack on own flag did not defer the preparation");
}
