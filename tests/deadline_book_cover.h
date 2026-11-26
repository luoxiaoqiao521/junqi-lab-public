#pragma once

#include "../sim/public_relay_screen.h"

// Nominate a quiet move only when it interrupts earlier enemy support
// while preserving the existing allied approach to an enemy headquarters.
class BookCover {
 public:
  BookCover(const jq::Observation& view,const jq::OpeningMarks& marks,
            const jq::Rules& rules,const std::vector<int>& flags) {
    using namespace jq;
    if (threatened_friendly_flag(view,marks)) return;
    const int ally=(view.viewer+2)%kSeats;
    const int reserve=next_seat(view.viewer,rules);
    const int target=(reserve+2)%kSeats;
    const auto& board=Board::get();
    for (int front=0;front<kNodes;++front) {
      if (view.seat_at(front)!=ally || board.node(front).seat!=target) continue;
      Front line{front,{},{}};
      for (const int flag:flags) {
        if (view.seat_at(flag)!=target) continue;
        if (reaches(view,marks,front,flag)) line.approaches.push_back(flag);
        for (const int gate:board.road(flag))
          if (view.seat_at(gate)==kNone && reaches(view,marks,front,gate))
            line.approaches.push_back(gate);
      }
      if (line.approaches.empty()) continue;
      for (int from=0;from<kNodes;++from)
        if (view.seat_at(from)==reserve && reaches(view,marks,from,front))
          line.attackers.push_back(from);
      if (!line.attackers.empty()) lines_.push_back(std::move(line));
    }
  }

  int score(const jq::Observation& after,const jq::OpeningMarks& marks) const {
    int best=0;
    for (const auto& line:lines_) {
      if (std::none_of(line.approaches.begin(),line.approaches.end(),
          [&](int to){ return reaches(after,marks,line.from,to); })) continue;
      // A different previously blocked route can open when the piece moves.
      const int reserve=after.seat_at(line.attackers.front());
      bool threatened=false;
      for (int from=0;from<jq::kNodes && !threatened;++from)
        threatened=after.seat_at(from)==reserve && reaches(after,marks,from,line.from);
      if (!threatened) best=std::max(best,static_cast<int>(line.attackers.size()));
    }
    return best;
  }

 private:
  struct Front { int from; std::vector<int> approaches, attackers; };
  std::vector<Front> lines_;

  static bool reaches(const jq::Observation& view,const jq::OpeningMarks& marks,int from,int to) {
    using namespace jq;
    const auto ranks=compatible_rank_mask(view,marks,from)&~immobile_ranks;
    const int owner=view.seat_at(from);
    const Move move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    return ((ranks&(1u<<Engineer)) && relay_has_move(relay_contact_moves(view,owner,from,Engineer),move)) ||
           ((ranks&~(1u<<Engineer)) && relay_has_move(relay_contact_moves(view,owner,from,Platoon),move));
  }
};

static void book_cover_fixtures(jq::Rules rules) {
  using namespace jq;
  auto root=selfplay_fixture();
  place_single_case_piece(root,1,108,Marshal);
  place_single_case_piece(root,2,4,Engineer);
  place_single_case_piece(root,26,114,Platoon);
  place_single_case_piece(root,50,86,Flag);
  place_single_case_piece(root,51,110,Platoon);
  place_single_case_piece(root,75,116,Flag);
  place_single_case_piece(root,76,102,Platoon);
  root.alive[2]=root.alive[3]=true;
  const Move cut{108,113};
  const auto view=seat_decision(root).observation;
  OpeningMarks marks(view,false);
  marks.ranks[110]=marks.ranks[114]=1u<<Platoon;
  const auto after=project_quiet(view,cut);
  const auto targets=flag_targets(view,marks);
  rules.ccw=true;
  single_case_check(is_legal(root,cut) && BookCover(view,marks,rules,targets).score(after,marks)>0,
                    "legal support cut with an allied approach was not nominated");
  marks.ranks[114]|=1u<<Engineer;
  single_case_check(BookCover(view,marks,rules,targets).score(after,marks)==0,
                    "an engineer could bypass the proposed support cut");
  rules.ccw=false;
  marks.ranks[114]=1u<<Platoon;
  single_case_check(BookCover(view,marks,rules,targets).score(after,marks)==0,
                    "reserve that acted after the ally triggered a support cut");
}
