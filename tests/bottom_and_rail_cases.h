#pragma once
#include "single_rules_cases.h"
#include <bitset>

namespace jq {
inline uint8_t movement_point(int seat,int row,int col) {
  return static_cast<uint8_t>(Board::get().seat_node(seat,row,col));
}

inline State movement_fixture(int seat,int flag_col=1) {
  State s{};
  std::fill_n(s.id_at,kNodes,int8_t(kNone));
  std::fill_n(s.seat_at,kNodes,int8_t(kNone));
  std::fill_n(s.rank_at,kNodes,int8_t(Unknown));
  std::fill_n(s.out,kSeats,true);
  const int enemy=(seat+1)%4;
  s.out[seat]=s.out[enemy]=false;
  s.to_move=static_cast<int8_t>(seat);
  place_single_case_piece(s,seat*25,movement_point(seat,5,flag_col),Flag);
  place_single_case_piece(s,enemy*25,movement_point(enemy,5,1),Flag);
  return s;
}

inline void movement_topology() {
  const auto& b=Board::get();
  const char cells[6][6]={"RRRRR","RCPCR","RPCPR","RCPCR","RRRRR","PHPHP"};
  for (int seat=0;seat<kSeats;++seat) for (int r=0;r<6;++r) for (int c=0;c<5;++c) {
    const char expected=cells[r][c];
    const auto& p=b.node(b.seat_node(seat,r,c));
    const char actual=p.rail ? 'R' : p.kind==NodeKind::Camp ? 'C' : p.kind==NodeKind::HQ ? 'H' : 'P';
    single_case_check(actual==expected,"local position type changed");
  }
  auto rotated=[&](int n) {
    const auto& p=b.node(n);
    return b.at_grid(16-p.gc,p.gr);
  };
  for (int n=0;n<kNodes;++n) {
    const int rot=rotated(n);
    single_case_check(rot>=0 && b.node(n).kind==b.node(rot).kind &&
                      b.node(n).rail==b.node(rot).rail,"quarter-turn topology mismatch");
    for (auto kind:{&Board::road,&Board::rail_adj}) {
      const auto& before=(b.*kind)(n);
      const auto& after=(b.*kind)(rot);
      single_case_check(before.size()==after.size(),"quarter-turn edge count mismatch");
      for (int to:before)
        single_case_check(std::find(after.begin(),after.end(),rotated(to))!=after.end(),
                          "quarter-turn edge mismatch");
    }
  }
}

inline void pictured_connections() {
  // Manual reading of the replay board: local streets and diagonals, plus blue rail links.
  const auto& b=Board::get();
  std::bitset<kNodes> road[kNodes],rail[kNodes];
  const auto link=[&](int a,int c,bool tracks) {
    single_case_check(a>=0 && c>=0,"pictured connection endpoint missing");
    road[a].set(c); road[c].set(a);
    if(tracks) { rail[a].set(c); rail[c].set(a); }
  };
  constexpr int diagonals[][5]={{6,0,2,10,12},{8,2,4,12,14},{12,6,8,16,18},
                                {16,10,12,20,22},{18,12,14,22,24}};
  for(int seat=0;seat<kSeats;++seat) {
    const auto local=[&](int r,int c) { return b.seat_node(seat,r,c); };
    for(int r=0;r<6;++r) for(int c=0;c<5;++c) {
      if(c<4) link(local(r,c),local(r,c+1),r==0||r==4);
      if(r<5) link(local(r,c),local(r+1,c),r<4&&(c==0||c==4));
    }
    for(const auto& camp:diagonals) for(int i=1;i<5;++i)
      link(local(camp[0]/5,camp[0]%5),local(camp[i]/5,camp[i]%5),false);
    link(local(0,4),b.seat_node((seat+1)%4,0,0),true);
  }
  for(int r:{6,8,10}) for(int c:{6,8,10}) {
    const int center=b.at_grid(r,c);
    if(r<10) link(center,b.at_grid(r+2,c),true);
    if(c<10) link(center,b.at_grid(r,c+2),true);
    if(r==6) link(center,b.at_grid(5,c),true);
    if(r==10) link(center,b.at_grid(11,c),true);
    if(c==6) link(center,b.at_grid(r,5),true);
    if(c==10) link(center,b.at_grid(r,11),true);
  }
  for(int n=0;n<kNodes;++n) {
    const auto verify=[&](const auto& actual,const auto& expected) {
      std::bitset<kNodes> drawn;
      for(int c:actual) drawn.set(c);
      return drawn==expected && drawn.count()==actual.size();
    };
    single_case_check(verify(b.road(n),road[n]),"pictured road missing or unexpected");
    single_case_check(verify(b.rail_adj(n),rail[n]),"pictured railway missing or unexpected");
  }
}

inline void bottom_movement_matrix(const Rules& rules) {
  for (int seat=0;seat<kSeats;++seat) for (int rank=Bomb;rank<RankCount;++rank) {
    const auto at=[&](int r,int c){ return movement_point(seat,r,c); };
    const int id=seat*25+1,enemy=(seat+1)%4;
    for (int col:{0,2,4}) {
      auto s=movement_fixture(seat);
      place_single_case_piece(s,id,at(4,col),rank);
      place_single_case_piece(s,enemy*25+1,movement_point(enemy,0,2),Brigade);
      const Move entry{at(4,col),at(5,col)},exit{at(5,col),at(4,col)};
      single_case_check(is_legal(s,entry),"ordinary sixth-row station rejected entry");
      const auto event=step(s,entry,rules);
      single_case_check(s.alive[id] && s.node_of[id]==entry.to && !s.out[seat] &&
                        !event.eliminated_mask && s.to_move==enemy,"ordinary station locked or retired a mover");
      single_case_check(!is_legal(s,exit),"station exit permitted on another seat's turn");
      s.to_move=static_cast<int8_t>(seat);
      single_case_check(is_legal(s,exit),"ordinary sixth-row station rejected exit");
      single_case_check(!is_legal(s,{entry.to,at(4,(col+2)%5)}) &&
                        !is_legal(s,{entry.to,at(5,(col+2)%5)}),
                        "ordinary station allowed a rail chain or skipped a sixth-row point");
    }
    for (int col:{1,3}) {
      auto s=movement_fixture(seat,4-col);
      place_single_case_piece(s,id,at(4,col),rank);
      place_single_case_piece(s,id+1,at(0,2),Platoon);
      place_single_case_piece(s,enemy*25+1,movement_point(enemy,0,2),Brigade);
      const Move entry{at(4,col),at(5,col)};
      single_case_check(is_legal(s,entry),"empty HQ rejected one-step entry");
      const auto locked=step(s,entry,rules);
      single_case_check(s.alive[id] && s.node_of[id]==entry.to && !locked.eliminated_mask &&
                        !can_leave_node(rank,entry.to),"HQ locking removed the piece or failed to lock it");
      s.to_move=static_cast<int8_t>(seat);
      single_case_check(!is_legal(s,{entry.to,entry.from}),"HQ let the entered piece leave");
      auto last=movement_fixture(seat,4-col);
      place_single_case_piece(last,id,entry.from,rank);
      place_single_case_piece(last,enemy*25+1,movement_point(enemy,0,2),Brigade);
      const auto exhausted=step(last,entry,rules);
      single_case_check(last.out[seat] && exhausted.exit_reason[seat]==ExitReason::MovablePiecesExhausted,
                        "entering HQ with the last movable piece failed to retire the seat");
    }
  }
}

inline void rail_and_camp_matrix() {
  const auto& b=Board::get();
  for (int seat=0;seat<kSeats;++seat) {
    const auto at=[&](int r,int c){ return movement_point(seat,r,c); };
    const int enemy=(seat+1)%4,partner=(seat+2)%4;
    auto s=movement_fixture(seat,3);
    const int id=seat*25+1;
    place_single_case_piece(s,id,at(4,0),Engineer);
    for (int dst=0;dst<kNodes;++dst) {
      const auto& p=b.node(dst);
      if (p.rail || p.seat!=seat) continue;
      const auto& adjacent=b.road(at(4,0));
      const bool road=std::find(adjacent.begin(),adjacent.end(),dst)!=adjacent.end();
      single_case_check(is_legal(s,{at(4,0),uint8_t(dst)})==road,
                        "engineer combined railway and road or omitted an adjacent road");
    }
    const Move far{at(4,0),at(4,4)};
    for (int owner:{seat,partner,enemy}) {
      auto detour=s;
      place_single_case_piece(detour,owner*25+8,at(4,2),Brigade);
      single_case_check(is_legal(detour,far),"open alternate railway path rejected");
      auto blocked=s;
      int id_offset=8;
      for (int n:b.rail_adj(far.from))
        place_single_case_piece(blocked,owner*25+id_offset++,n,Brigade);
      single_case_check(!is_legal(blocked,far),"engineer passed an occupied railway cut");
      const auto first=b.rail_adj(far.from).front();
      single_case_check(is_legal(blocked,{far.from,first})==(owner==enemy),"rail endpoint attack ownership");
    }
    for (int rank=Bomb;rank<RankCount;++rank) {
      auto camp=movement_fixture(seat);
      place_single_case_piece(camp,id,at(1,0),rank);
      const Move entry{at(1,0),at(1,1)};
      single_case_check(is_legal(camp,entry),"empty camp refused adjacent road entry");
      auto inside=movement_fixture(seat);
      place_single_case_piece(inside,id,entry.to,rank);
      single_case_check(is_legal(inside,{entry.to,entry.from}) &&
                        !is_legal(inside,{entry.to,at(4,0)}) && !is_legal(inside,{entry.to,at(1,3)}),
                        "camp exit must follow one road edge");
      place_single_case_piece(camp,enemy*25+1,entry.to,Engineer);
      single_case_check(!is_legal(camp,entry),"occupied enemy camp accepted attack");
    }
    auto arc=movement_fixture(seat);
    place_single_case_piece(arc,id,at(3,4),Marshal);
    single_case_check(is_legal(arc,{at(3,4),movement_point(enemy,3,0)}) &&
                      !is_legal(arc,{at(3,4),at(0,0)}),"ordinary arc and right-angle distinction");
  }
}

inline void blocked_bottom_engineer(const Rules& rules) {
  auto s=movement_fixture(0);
  const auto at=[](int r,int c){ return movement_point(1,r,c); };
  place_single_case_piece(s,26,at(5,2),Engineer);
  place_single_case_piece(s,27,at(5,3),Bomb);
  s.out[3]=false;
  place_single_case_piece(s,75,movement_point(3,5,1),Flag);
  place_single_case_piece(s,76,at(4,2),Platoon);
  place_single_case_piece(s,2,movement_point(0,0,0),Engineer);
  const Move prepare{movement_point(0,0,0),movement_point(0,0,1)},open{at(4,2),at(4,1)};
  single_case_check(is_legal(s,prepare),"blocked bottom fixture preparation");
  auto old=s;
  const auto retired=step(old,prepare,Rules{});
  single_case_check(old.out[1] && retired.exit_reason[1]==ExitReason::NoLegalMove,
                    "legacy no-legal-move fixture did not retire");
  const auto skipped=step(s,prepare,rules);
  single_case_check(skipped.skipped_mask==2 && !s.out[1] && s.alive[26] &&
                    s.no_progress==1 && s.ply==1 && s.to_move==3,"selfplay cleared a blocked bottom engineer");
  single_case_check(is_legal(s,open),"blocker could not vacate the bottom exit");
  step(s,open,rules);
  const Move wait{prepare.to,prepare.from};
  single_case_check(s.to_move==0 && is_legal(s,wait),"rotation did not return to the preceding seat");
  step(s,wait,rules);
  single_case_check(s.to_move==1 && is_legal(s,{at(5,2),at(4,2)}),"unblocked engineer did not resume legal play");
}

inline void retirement_opens_enemy_route(const Rules& rules) {
  auto s=movement_fixture(0);
  place_single_case_piece(s,1,51,Engineer);
  place_single_case_piece(s,2,15,Platoon);
  place_single_case_piece(s,26,22,Brigade);
  s.out[3]=false;
  place_single_case_piece(s,75,movement_point(3,5,1),Flag);
  place_single_case_piece(s,76,24,Brigade);
  auto before=s;
  before.to_move=3;
  single_case_check(!is_legal(before,{24,21}),"teammate blocker failed to close the route");
  const Move capture{51,56};
  single_case_check(is_legal(s,capture),"flag capture fixture invalid");
  auto retained=s;
  auto flag_only=rules;
  flag_only.flag_eliminates=false;
  const auto uncoupled=step(retained,capture,flag_only);
  single_case_check(!retained.alive[25] && retained.alive[26] && !retained.out[1] &&
                    retained.to_move==1 && !uncoupled.eliminated_mask,
                    "flag-only comparison must retain the defender's living mover and turn");
  retained.to_move=3;
  single_case_check(!is_legal(retained,{24,21}),"retained piece failed to block the next enemy's route");
  const auto captured=step(s,capture,rules);
  single_case_check(captured.exit_reason[1]==ExitReason::FlagRemoved && s.to_move==3 &&
                    !s.alive[25] && !s.alive[26] && s.alive[1] && s.node_of[1]==capture.to &&
                    s.result==Result::Ongoing && s.id_at[22]==kNone && is_legal(s,{24,21}),
                    "flag retirement failed to preserve attacker, clear victim or open route");
  step(s,{24,21},rules);
  single_case_check(s.to_move==0 && is_legal(s,{15,10}),"follow-on defender fixture invalid");
  step(s,{15,10},rules);
  single_case_check(s.to_move==3 && is_legal(s,{21,26}),"cleared line failed to produce a real flag threat");
}

inline void bottom_and_rail_contract() {
  Rules selfplay;
  selfplay.eliminate_exhausted_movers=true;
  movement_topology();
  pictured_connections();
  bottom_movement_matrix(selfplay);
  rail_and_camp_matrix();
  blocked_bottom_engineer(selfplay);
  retirement_opens_enemy_route(selfplay);
  std::cout<<"PASS pictured graph and bottom/rail gate: four seats, 10 movable ranks, 120 ordinary stations, "
              "80 HQ entries and exhaustion controls, camp/rail cuts, blocked engineer and post-retirement route\n";
}
}
