#include "../sim/decision_branches.h"
#include "layout.h"
#include <iostream>

using namespace jq;

struct Metrics {
  int contact=0, contacts=0, first_flag=0, breach=0, no_moves=0;
  bool own_no_move=false, own_flag_taken=false;
};

struct RuleSplit {
  int ply=0, archive_mask=0, training_mask=0;
};

static int own_flag_front(const State& state,int seat) {
  const auto& board=Board::get();
  for(int col:{1,3})
    if(state.rank_at[board.seat_node(seat,5,col)]==Flag) return board.seat_node(seat,4,col);
  throw std::runtime_error("missing own flag");
}

static void update(Metrics& m,const State& state,const StepInfo& event,int defender,
                   int defender_rank,int seat,int front) {
  if(event.outcome!=Outcome::Move) {
    if(!m.contact) m.contact=state.ply;
    ++m.contacts;
  }
  const bool removed=event.outcome==Outcome::AttackerWins || event.outcome==Outcome::BothRemoved;
  const bool flag_removed=defender_rank==Flag && removed;
  if(flag_removed && !m.first_flag) m.first_flag=state.ply;
  if(flag_removed && defender==seat) m.own_flag_taken=true;
  if(!m.breach && event.move.to==front && event.move.from!=front &&
     team_of(event.seat)!=team_of(seat) && state.seat_at[front]==event.seat && !state.out[seat])
    m.breach=state.ply;
  for(int owner=0;owner<kSeats;++owner)
    if((event.eliminated_mask&(1u<<owner)) && !(flag_removed && owner==defender)) {
      ++m.no_moves;
      if(owner==seat) m.own_no_move=true;
    }
}

static void mirror_layout(int8_t (&layout)[kPiecesPerSeat]) {
  const auto& board=Board::get();
  for(int i=0;i<kPiecesPerSeat;++i) {
    const auto n=board.node(layout_slot_node(0,i));
    if(n.c>=2) continue;
    const int opposite=board.seat_node(0,n.r,4-n.c);
    for(int j=i+1;j<kPiecesPerSeat;++j)
      if(layout_slot_node(0,j)==opposite) { std::swap(layout[i],layout[j]); break; }
  }
}

static StepInfo audit_step(State& state,Move move,const Rules& rules,const Rules& archive,RuleSplit& split) {
  if(!rules.eliminate_exhausted_movers || split.ply) return step(state,move,rules);
  auto before=state;
  const auto legacy=step(before,move,archive);
  const auto training=step(state,move,rules);
  if(state_hash(before)!=state_hash(state))
    split={state.ply,legacy.eliminated_mask,training.eliminated_mask};
  return training;
}

static void play(State state,int block,int source,int side,int table,int seat,int first,uint64_t seed,
                 const ShortWeights& weights,const Rules& rules,bool pressure_test) {
  auto streams=review_streams(seed);
  const int front=own_flag_front(state,seat);
  const int targeted_enemy=next_seat(seat,rules);
  Metrics m;
  RuleSplit split;
  Rules archive=rules;
  archive.eliminate_exhausted_movers=false;
  archive.check_involved_no_move=true;
  const auto report=[&](int budget) {
    const int result=state.result==Result::Ongoing ? -1 : review_outcome(state.result,seat);
    std::cout<<block<<' '<<source<<' '<<side<<' '<<table<<' '<<seat<<' '<<first<<' '<<seed<<' '
             <<result<<' '<<state.ply<<' '<<m.contact<<' '<<m.contacts<<' '<<m.first_flag<<' '
             <<m.breach<<' '<<m.no_moves<<' '<<m.own_no_move<<' '<<m.own_flag_taken;
    if(pressure_test) std::cout<<' '<<budget;
    if(rules.eliminate_exhausted_movers)
      std::cout<<' '<<split.ply<<' '<<split.archive_mask<<' '<<split.training_mask;
    std::cout<<'\n';
  };
  while(state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto d=seat_decision(state);
    const bool heuristic=table && (!pressure_test || actor==targeted_enemy);
    const auto move=review_move(d,seat,heuristic ? 2 : 0,true,weights,streams[actor]);
    const int defender=state.seat_at[move.to], rank=state.rank_at[move.to];
    const auto event=audit_step(state,move,rules,archive,split);
    update(m,state,event,defender,rank,seat,front);
    if(pressure_test && state.ply==800 && state.result==Result::Ongoing) report(800);
  }
  if(!pressure_test || state.ply<=800) report(800);
  if(pressure_test) report(1600);
}

int main(int argc,char** argv) {
  const std::string mode=argc==2 ? argv[1] : "";
  const bool retirement=mode=="retirement";
  const bool pressure_test=retirement || mode=="pressure";
  try {
    int count;
    ShortWeights weights{};
    if(!(std::cin>>count)||count<1) return 2;
    for(size_t i=0;i<5;++i) if(!(std::cin>>weights[i])||!std::isfinite(weights[i])) return 2;
    Rules rules;
    rules.ccw=false;
    rules.marshal_reveals_flag=true;
    rules.check_involved_no_move=!retirement;
    rules.eliminate_exhausted_movers=retirement;
    rules.no_progress_limit=71;
    rules.truncation_cap=pressure_test ? 1600 : 800;
    for(int block=0;block<count;++block) {
      uint64_t seed;
      if(!(std::cin>>seed)) return 2;
      const int seat=block%4,first=(block/4)%4;
      int8_t layouts[kSeats][kPiecesPerSeat];
      Rng builder(mix_seed(seed,18));
      for(auto& layout:layouts) generate_layout(layout,builder);
      State reference{};
      init_state(reference,layouts,first,rules);
      for(int source=0;source<3;++source) {
        if(source) {
          for(auto& rank:layouts[seat]) {
            int value;
            if(!(std::cin>>value)||value<Flag||value>=RankCount) return 2;
            rank=static_cast<int8_t>(value);
          }
        }
        for(int side=0;side<2;++side) {
          if(side) mirror_layout(layouts[seat]);
          if(!valid_layout(layouts[seat])) return 3;
          State state{};
          init_state(state,layouts,first,rules);
          for(int observer=0;observer<kSeats;++observer)
            if(observer!=seat && !same_seat_view(reference,state,observer)) return 4;
          for(int table=0;table<2;++table)
            play(state,block,source,side,table,seat,first,seed,weights,rules,pressure_test);
        }
      }
    }
  } catch(const std::exception& err) {
    std::cerr<<err.what()<<'\n';
    return 5;
  }
}
