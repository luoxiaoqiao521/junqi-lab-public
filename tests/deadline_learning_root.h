#pragma once

// This probe publishes the acting seat's information before the first changed
// move. Referee identities do not enter this function or the input fields.
static void print_learning_root(std::ostream& out,const jq::SeatDecision& decision,
    const jq::OpeningMarks& marks,int quiet,uint64_t seed,int learner,int context,uint64_t prefix) {
  using namespace jq;
  const auto& view=decision.observation;
  out<<"P "<<seed<<' '<<learner<<' '<<context<<' '<<prefix<<' '<<view.ply<<' '<<view.viewer<<' '<<quiet;
  for(int n=0;n<kNodes;++n)
    out<<' '<<view.seat_at(n)<<' '<<view.rank_at(n)<<' '
       <<(view.seat_at(n)>=0?compatible_rank_mask(view,marks,n):0)<<' '<<int(marks.moved[n]);
  for(const auto m:decision.actions) out<<' '<<(int(m.from)*kNodes+m.to);
  out<<'\n';
}
