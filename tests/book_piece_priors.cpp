#include "../sim/archive_review.h"
#include "../sim/public_threat_evidence.h"
#include <iostream>

using namespace jq;

static std::optional<Rules> strict_archive(const ArchiveRows& raw,const std::string& config,
                                           const std::string& visibility) {
  std::optional<Rules> result;
  for (const bool ccw:{false,true}) {
    try {
      archive_review_attempt(raw,config,visibility,ccw);
    } catch (const std::runtime_error&) {
      continue;
    }
    if (result) return std::nullopt;
    Rules rules;
    std::string error;
    if (!parse_rules(archive_review_rules(config,ccw),rules,error)) throw std::runtime_error(error);
    result=rules;
  }
  return result;
}

static void emit_move(const char* kind,const std::string& key,const State& before,
                      Move move,int rank,unsigned mask,int extra=0,int stratum=0) {
  expect_value(rank>=Flag && rank<RankCount && (mask&(1u<<rank)),"book audit excluded actual rank");
  std::cout<<kind<<' '<<key<<' '<<before.ply+1<<' '<<int(before.to_move)<<' '
           <<int(move.from)<<' '<<int(move.to)<<' '<<rank<<' '<<mask<<' '<<extra<<' '<<stratum<<'\n';
}

static void opening_layout(const std::string& key,const State& state) {
  const auto& b=Board::get();
  for (int seat=0;seat<kSeats;++seat) {
    for (int col:{1,3}) {
      const auto at=b.seat_node(seat,5,col);
      const int rank=state.rank_at[at];
      if (rank!=Flag) std::cout<<"H "<<key<<' '<<seat<<' '<<at<<' '<<rank<<'\n';
    }
    for (int col=0;col<5;++col) {
      const auto at=b.seat_node(seat,0,col);
      std::cout<<"F "<<key<<' '<<seat<<' '<<at<<' '<<int(state.rank_at[at])<<'\n';
    }
  }
}

static void quiet_camp(const std::string& key,const State& prior,Move move,
                       const std::array<bool,kPieces>& involved,const OpeningMarks& public_marks) {
  if (prior.ply>=40 || involved[prior.id_at[move.from]]) return;
  const auto& b=Board::get();
  const auto origin=b.node(move.from), target=b.node(move.to);
  if (origin.seat!=prior.to_move || origin.r==0 || target.seat!=prior.to_move) return;
  const auto& adjacent=b.road(move.from);
  if (std::find(adjacent.begin(),adjacent.end(),move.to)==adjacent.end()) return;
  bool camp=false, other=false;
  for (const int to:adjacent) if (prior.seat_at[to]<0 && b.node(to).seat==prior.to_move) {
    camp|=b.node(to).kind==NodeKind::Camp;
    other|=b.node(to).kind!=NodeKind::Camp;
  }
  if (!camp || !other) return;
  const auto view=observe(prior,kNone);
  const auto mask=compatible_rank_mask(view,public_marks,move.from)&public_route_ranks(view,move);
  if (!(mask&(1u<<Bomb))) return;
  emit_move("C",key,prior,move,prior.rank_at[move.from],mask,target.kind==NodeKind::Camp,
            5*origin.r+std::min<int>(origin.c,4-origin.c));
}

static unsigned casualty_support(const State& after,const OpeningMarks& marks,int viewer,int owner) {
  auto projected=marks;
  constrain_rank_inventory(observe(after,viewer),projected,owner);
  const auto& casualties=projected.removed[owner];
  expect_value(casualties.size>0,"missing casualty in book audit");
  return casualties.masks[casualties.size-1];
}

static void combat_notes(const std::string& key,const State& before,const State& after,const StepInfo& event,
                         const std::array<OpeningMarks,kSeats>& marks) {
  const auto move=event.move;
  const int a=event.seat, d=before.seat_at[move.to];
  const int attack=before.rank_at[move.from], defense=before.rank_at[move.to];
  if (event.outcome==Outcome::AttackerWins && defense==Division)
    emit_move("D",key,before,move,attack,compatible_rank_mask(observe(after,d),marks[d],move.to));
  if (event.outcome==Outcome::DefenderWins && attack==Division)
    emit_move("D",key,before,move,defense,compatible_rank_mask(observe(after,a),marks[a],move.to),1);
  if (event.outcome!=Outcome::BothRemoved) return;
  const auto& adjacent=Board::get().road(move.from);
  const bool rail=std::find(adjacent.begin(),adjacent.end(),move.to)==adjacent.end();
  if (attack==Engineer)
    emit_move("E",key,before,move,defense,casualty_support(after,marks[a],a,d),0,rail);
  if (defense==Engineer)
    emit_move("E",key,before,move,attack,casualty_support(after,marks[d],d,a),1,rail);
}

static void audit_game(const std::string& key,const ArchiveRows& raw,const Rules& rules) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  std::cout<<"G "<<key<<' '<<raw.entries<<'\n';
  opening_layout(key,review.state);
  auto view=[&](int seat){return observe(review.state,seat);};
  std::array<OpeningMarks,kSeats> marks{OpeningMarks(view(0),true),OpeningMarks(view(1),true),
                                      OpeningMarks(view(2),true),OpeningMarks(view(3),true)};
  OpeningMarks common(view(kNone),true);
  std::array<bool,kPieces> involved{};
  for (int i=0;i<raw.entries;++i) {
    const auto row=raw.row(i);
    if (raw.byte(row)==245) { review.control(raw,row,i+1,rules); continue; }
    const auto prior=review.state;
    review.play(raw,row,i+1,rules);
    review.frames.str("");
    const auto e=review.last;
    if (e.outcome==Outcome::Move) quiet_camp(key,prior,e.move,involved,common);
    for (int seat=0;seat<kSeats;++seat) if (!prior.out[seat])
      advance_contact_ranks(marks[seat],observe(prior,seat),view(seat),e.move,e.outcome,rules);
    if (prior.ply<40)
      advance_contact_ranks(common,observe(prior,kNone),view(kNone),e.move,e.outcome,rules);
    involved[prior.id_at[e.move.from]]=true;
    if (e.outcome!=Outcome::Move) {
      combat_notes(key,prior,review.state,e,marks);
      involved[prior.id_at[e.move.to]]=true;
    }
  }
  std::cout<<"P "<<key<<' '<<review.state.ply<<'\n';
}

int main() {
  try {
    const auto config=read_text("config/ruleset.json"), visibility=read_text("config/observation.json");
    std::string key,tape;
    while (std::cin>>key>>tape) {
      std::optional<ArchiveRows> raw;
      try {
        raw.emplace(decode_archive_tape(tape));
      } catch (const std::runtime_error&) {
        std::cout<<"R "<<key<<'\n';
        continue;
      }
      const auto rules=strict_archive(*raw,config,visibility);
      if (rules) audit_game(key,*raw,*rules);
      else std::cout<<"R "<<key<<'\n';
    }
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
