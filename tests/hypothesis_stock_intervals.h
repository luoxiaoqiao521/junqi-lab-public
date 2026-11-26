#pragma once

static void check_stock_range(RankCountRange range,int low,int high) {
  single_case_check(range.minimum==low && range.maximum==high,"wrong stock range");
}

static void stock_interval_fixture() {
  auto state=single_case_state();
  place_single_case_piece(state,1,4,Engineer);
  place_single_case_piece(state,26,30,Bomb);
  place_single_case_piece(state,27,33,Engineer);
  const auto view=observe(state,0);
  OpeningMarks marks(view,false);
  constexpr unsigned lead=(1u<<Marshal)|(1u<<Corps), bomb=1u<<Bomb;
  marks.ranks[56]=1u<<Flag;
  marks.ranks[30]=lead|bomb;
  marks.ranks[33]=1u<<Engineer;
  marks.removed[1].add(lead);
  check_stock_range(rank_stock(view,marks,1,Marshal).collision_lost,0,1);
  check_stock_range(rank_stock(view,marks,1,Marshal).on_board,0,1);
  marks.removed[1].add(lead);
  check_stock_range(rank_stock(view,marks,1,Marshal).collision_lost,1,1);
  check_stock_range(rank_stock(view,marks,1,Corps).collision_lost,1,1);
  check_stock_range(rank_stock(view,marks,1,Marshal).on_board,0,0);
  check_stock_range(rank_stock(view,marks,1,Bomb).on_board,1,1);
  marks.removed[1].add(1u<<Engineer);
  check_stock_range(rank_stock(view,marks,1,Engineer).collision_lost,1,1);
  check_stock_range(rank_stock(view,marks,1,Engineer).on_board,1,1);
  check_stock_range(rank_stock(view,marks,3,Engineer).on_board,0,0);
  check_stock_range(rank_stock(view,marks,3,Engineer).collision_lost,0,0);
  // With two identical ranks, verify min/max against explicit assignments.
  const std::array<int,3> ranks{Regiment,Division,Corps};
  for (unsigned mask=1;mask<8;++mask) for (unsigned loss=1;loss<8;++loss) {
    marks.removed[1]=RemovedRanks{};
    marks.ranks[30]=0;
    unsigned dead=0;
    for (int i=0;i<3;++i) {
      if (mask&(1u<<i)) marks.ranks[30]|=1u<<ranks[i];
      if (loss&(1u<<i)) dead|=1u<<ranks[i];
    }
    if (mask==4 && loss==4) continue;
    marks.removed[1].add(dead);
    for (int target:ranks) {
      int live_min=1,live_max=0,loss_min=1,loss_max=0;
      for (int a:ranks) for (int b:ranks) {
        if (!(marks.ranks[30]&(1u<<a)) || !(dead&(1u<<b)) || (a==b && kRankCount[a]==1)) continue;
        live_min=std::min(live_min,int(a==target));
        live_max=std::max(live_max,int(a==target));
        loss_min=std::min(loss_min,int(b==target));
        loss_max=std::max(loss_max,int(b==target));
      }
      const auto stock=rank_stock(view,marks,1,target);
      check_stock_range(stock.on_board,live_min,live_max);
      check_stock_range(stock.collision_lost,loss_min,loss_max);
    }
  }
}
