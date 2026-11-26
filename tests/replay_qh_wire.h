#pragma once
#include <array>
#include <iostream>
#include <vector>
#include "../core/board.h"
#include "../core/layout.h"

namespace jq {

static int point(int x,int y) {
  return x<0 || y<0 || x>16 || y>16?kNone:Board::get().at_grid(y,x);
}

static bool read_qh(int8_t (&layouts)[4][25],std::vector<std::array<int,4>>& rows) {
  const auto& board=Board::get();
  int count;
  if(!(std::cin>>count) || count<1 || count>1002) return false;
  std::array<bool,kNodes> used{};
  for (int i=0;i<100;++i) {
    int x,y,rank;
    if(!(std::cin>>x>>y>>rank) || rank<Flag || rank>=RankCount) return false;
    const int n=point(x,y);
    if(n<0 || used[n] || board.node(n).seat<0 || board.node(n).kind==NodeKind::Camp) return false;
    used[n]=true;
    const int seat=board.node(n).seat;
    for(int slot=0;slot<25;++slot) if(layout_slot_node(seat,slot)==n) layouts[seat][slot]=static_cast<int8_t>(rank);
  }
  rows.resize(count);
  for(auto& a:rows) if(!(std::cin>>a[0]>>a[1]>>a[2]>>a[3])) return false;
  return true;
}
}
