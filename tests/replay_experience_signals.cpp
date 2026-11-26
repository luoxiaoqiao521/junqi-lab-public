#include "../sim/archive_review.h"
#include <array>
#include <iostream>

using namespace jq;

struct OpenCount {
  int turns=0;
  std::array<bool,2> untouched{true,true};
  std::array<int,2> opportunities{};
};

static bool ordinary_exit(const State& state,int seat,int node) {
  auto visible=observe(state,seat);
  // Evaluate a movable-rank hypothesis. Never use the unknown rank to define an opportunity.
  visible.ranks[node]=Platoon;
  std::vector<Move> moves;
  gen_moves(visible,seat,moves);
  return std::any_of(moves.begin(),moves.end(),[node](Move action){return action.from==node;});
}

static void opening_camp(const ArchiveRows& raw,const StepInfo& event,const State& before,
                         const std::string& key) {
  const auto& board=Board::get();
  const int actor=event.seat;
  const auto& landing=board.node(event.move.to);
  if (event.outcome!=Outcome::Move || landing.seat!=actor ||
      landing.kind!=NodeKind::Camp || landing.c==2) return;
  int marshal_column=kNone;
  for (int slot=0;slot<kPiecesPerSeat;++slot) if (raw.layouts[actor][slot]==Marshal)
    marshal_column=board.node(layout_slot_node(actor,slot)).c;
  expect_value(marshal_column>=0,"opening has no marshal");
  const int wing=marshal_column==2?2:(marshal_column<2)==(landing.c<2)?0:1;
  const auto& origin=board.node(event.move.from);
  if (origin.seat!=actor) return;
  std::cout<<"C "<<key<<' '<<actor<<' '<<int(landing.r)<<' '<<wing<<' '
           <<int(before.rank_at[event.move.from])<<' '<<int(origin.rail)<<' '<<int(origin.r)<<' '<<int(origin.c)<<'\n';
}

static void observe_corners(OpenCount& count,const State& before,const StepInfo& event,
                            const std::string& key) {
  const int seat=event.seat;
  for (int side:{0,1}) {
    const int node=Board::get().seat_node(seat,4,side*4);
    if (before.seat_at[node]!=seat) count.untouched[side]=false;
    if (!count.untouched[side]) continue;
    if (count.turns==5) {
      std::cout<<"N "<<key<<' '<<seat<<' '<<side<<' '<<count.opportunities[side]
               <<' '<<int(before.rank_at[node])<<'\n';
    } else {
      count.opportunities[side]+=ordinary_exit(before,seat,node);
      if (event.move.from==node) count.untouched[side]=false;
    }
  }
  ++count.turns;
}

static void early_signals(const ArchiveRows& raw,const Rules& rules,const std::string& key) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  std::array<OpenCount,kSeats> tally;
  for (int i=0;i<raw.entries && review.state.ply<40;++i) {
    const auto row=raw.row(i);
    if (raw.byte(row)==245) break;
    const auto before=review.state;
    review.play(raw,row,i+1,rules);
    review.frames.str("");
    const int seat=review.last.seat;
    auto& count=tally[seat];
    if (!count.turns) opening_camp(raw,review.last,before,key);
    if (count.turns<=5) observe_corners(count,before,review.last,key);
  }
}

int main(int argc,char** argv) {
  if (argc!=2) return 2;
  try {
    Rules rules;
    std::string error;
    expect_value(parse_rules(archive_review_rules(read_text(argv[1]),false),rules,error),"invalid study rules");
    std::string key,hex;
    int rotation=0;
    while (std::cin>>key>>rotation>>hex) {
      expect_value(rotation==0 || rotation==1,"invalid turn order");
      rules.ccw=bool(rotation);
      early_signals(ArchiveRows(decode_archive_tape(hex)),rules,key);
    }
    if (!std::cin.eof()) return 3;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 4;
  }
}
