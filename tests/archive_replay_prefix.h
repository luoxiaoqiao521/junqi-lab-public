#pragma once
#include "../sim/draw_branch.h"
#include "layout.h"
#include <iostream>
#include <string>

namespace jq::archive_probe {
struct Row { int x,y,xx,yy,result; };

inline int point(int x,int y) {
  return x<0||x>16||y<0||y>16 ? kNone : Board::get().at_grid(y,x);
}

inline std::string coords(Move move) {
  const auto& a=Board::get().node(move.from);
  const auto& b=Board::get().node(move.to);
  return std::to_string(a.gr)+","+std::to_string(a.gc)+">"+
         std::to_string(b.gr)+","+std::to_string(b.gc);
}

inline bool read_layouts(int8_t (&layouts)[4][25]) {
  for(int i=0;i<100;++i) {
    int x,y,rank;
    if(!(std::cin>>x>>y>>rank) || rank<2 || rank>13) return false;
    const int node=point(x,y);
    if(node<0) return false;
    const int seat=Board::get().node(node).seat;
    if(seat<0) return false;
    for(int slot=0;slot<25;++slot) if(layout_slot_node(seat,slot)==node)
      layouts[seat][slot]=static_cast<int8_t>(rank<5?rank-2:16-rank);
  }
  return true;
}

// The archived prefix can include an already-recorded external retirement.
inline bool archive_notice(State& state,const Row& row,const Rules& rules) {
  if(row.xx!=3 && row.xx!=4) return false;
  const int seat=row.y;
  if(seat<0||seat>=4) return false;
  if(state.out[seat]) return true;
  if(row.xx!=3 || row.result!=4) return false;
  state.out[seat]=true;
  for(int node=0;node<kNodes;++node) if(state.seat_at[node]==seat) {
    state.alive[state.id_at[node]]=false;
    state.id_at[node]=static_cast<int8_t>(kNone);
    state.seat_at[node]=static_cast<int8_t>(kNone);
    state.rank_at[node]=static_cast<int8_t>(Unknown);
  }
  for(int j=0;j<4&&state.out[state.to_move];++j)
    state.to_move=static_cast<int8_t>(next_seat(state.to_move,rules));
  return !state.out[state.to_move];
}

inline bool archive_prefix(State& state,MovedSquares& moved,Move& focus, const Rules& rules,
                           int8_t (&layouts)[4][25],const std::vector<Row>& rows,bool focus_outcome=true,
                           const ReplayObserver& observed={}) {
  for(const auto& layout:layouts) if(!valid_layout(layout)) return false;
  bool started=false;
  for(const auto& row:rows) {
    if(row.x==-1) {
      if(!started||!archive_notice(state,row,rules)) return false;
      continue;
    }
    const int from=point(row.x,row.y),to=point(row.xx,row.yy);
    if(from<0||to<0) return false;
    if(!started) {
      init_state(state,layouts,Board::get().node(from).seat,rules);
      started=true;
    }
    const Move actual{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    if(!is_legal(state,actual)) return false;
    State next=state;
    const auto event=step(next,actual,rules);
    if(&row==&rows.back()) {
      focus=actual;
      return !focus_outcome || int(event.outcome)==row.result;
    }
    if(int(event.outcome)!=row.result) return false;
    if(observed) observed(state,event,next);
    state=next;
    observe_movement(moved,event,observe(state,state.to_move));
  }
  return false;
}
}
