#pragma once
#include <iostream>
#include "opening_train.h"

namespace jq {

inline int archive_point(int x,int y) {
  return x<0 || x>16 || y<0 || y>16 ? kNone : Board::get().at_grid(y,x);
}

inline bool curriculum_layouts(int8_t (&layouts)[kSeats][kPiecesPerSeat]) {
  bool seen[kNodes]{};
  bool valid=true;
  for (int i=0;i<kPieces;++i) {
    int x,y,rank;
    expect_value(bool(std::cin>>x>>y>>rank),"incomplete curriculum layout");
    const int node=archive_point(x,y);
    if (node<0 || seen[node] || rank<2 || rank>13 || Board::get().node(node).seat<0) {
      valid=false;
      continue;
    }
    seen[node]=true;
    const int seat=Board::get().node(node).seat;
    bool slot=false;
    for (int j=0;j<kPiecesPerSeat;++j) if (layout_slot_node(seat,j)==node) {
      layouts[seat][j]=static_cast<int8_t>(rank<5 ? rank-2 : 16-rank);
      slot=true;
      break;
    }
    valid &= slot;
  }
  for (const auto& layout:layouts) valid &= valid_layout(layout);
  return valid;
}

inline const char* curriculum_prefix(State& state,const Rules& rules) {
  int count;
  expect_value(bool(std::cin>>count) && count>0 && count<=160,"invalid curriculum prefix bound");
  int8_t layouts[kSeats][kPiecesPerSeat]{};
  const char* error=curriculum_layouts(layouts) ? nullptr : "invalid_layout";
  for (int i=0;i<count;++i) {
    int x,y,xx,yy,result;
    expect_value(bool(std::cin>>x>>y>>xx>>yy>>result),"incomplete curriculum move");
    if (error) continue;
    const int from=archive_point(x,y),to=archive_point(xx,yy);
    if (from<0 || to<0 || result<0 || result>3) { error="invalid_record"; continue; }
    if (i==0) {
      const int seat=Board::get().node(from).seat;
      if (seat<0) { error="invalid_first_seat"; continue; }
      init_state(state,layouts,seat,rules);
    }
    const Move move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    if (!is_legal(state,move)) { error="illegal_move_or_turn"; continue; }
    const auto event=step(state,move,rules);
    if (int(event.outcome)!=result) error="collision_mismatch";
    else if (event.eliminated_mask || state.result!=Result::Ongoing) error="exit_or_terminal_in_prefix";
  }
  return error;
}

inline const char* curriculum_group(const SeatDecision& d) {
  bool quiet=false,contact=false,pressure=false;
  for (auto move:d.actions) {
    if (d.observation.seat_at(move.to)<0) quiet=true;
    else { contact=true; pressure |= hq_pressure_contact(d.observation,move); }
  }
  if (!quiet || !contact) return "unmatched";
  return pressure ? "pressure" : "ordinary";
}

inline void assess_curriculum(const std::vector<State>& roots,const Rules& rules,
                              ShortWeights weights,uint64_t seed) {
  for (size_t index=0;index<roots.size();++index) for (int context:{0,4}) {
    for (int rep=0;rep<6;++rep) {
      const auto& root=roots[index];
      const auto stream=mix_seed(seed+index,static_cast<uint64_t>(rep));
      const auto result=continue_opening(root,rules,stream,root.to_move,weights,false,false,context);
      std::cout << "{\"root\":" << index << ",\"seat\":" << int(root.to_move) << ",\"table\":" << context
                << ",\"rep\":" << rep << ",\"score\":";
      if (result.score) std::cout << *result.score;
      else std::cout << "null";
      std::cout << ",\"contact\":" << int(result.first_contact)
                << ",\"hq_pressure\":" << int(result.first_hq_pressure) << "}\n";
    }
  }
}

inline std::array<Move,2> pressure_practice_choices(const SeatDecision& d,const ShortWeights& weights) {
  expect_value(std::string(curriculum_group(d))=="pressure","practice requires a public HQ and quiet option");
  const auto masses=short_action_masses(d.observation,d.actions,weights);
  std::array<Move,2> choices{};
  std::array<double,2> best{-1,-1};
  for (size_t i=0;i<d.actions.size();++i) {
    const auto move=d.actions[i];
    const bool pressure=hq_pressure_contact(d.observation,move);
    if (!pressure && d.observation.seat_at(move.to)>=0) continue;
    const int index=pressure?0:1;
    if (masses[i]>best[index]) {
      choices[index]=move;
      best[index]=masses[i];
    }
  }
  return choices;
}

inline void print_practice_trial(State root,Move move,const Rules& rules,const ShortWeights& weights,
                                 uint64_t seed,int table) {
  const int learner=root.to_move;
  step(root,move,rules);
  const auto episode=continue_opening(root,rules,seed,learner,weights,false,false,table);
  std::cout << "{\"move\":[" << int(move.from) << ',' << int(move.to) << "],\"score\":";
  if (episode.score) std::cout << *episode.score;
  else std::cout << "null";
  std::cout << '}';
}

inline void practice_curriculum(const std::vector<State>& roots,const Rules& rules,
                                const ShortWeights& weights,uint64_t seed) {
  for (size_t i=0;i<roots.size();++i) {
    const auto& root=roots[i];
    const auto choices=pressure_practice_choices(seat_decision(root),weights);
    for (int table:{0,4}) for (int rep=0;rep<12;++rep) {
      const auto stream=mix_seed(seed+i,static_cast<uint64_t>(rep));
      std::cout << "{\"root\":" << i << ",\"seat\":" << int(root.to_move) << ",\"table\":" << table
                << ",\"rep\":" << rep << ",\"pressure\":";
      print_practice_trial(root,choices[0],rules,weights,stream,table);
      std::cout << ",\"quiet\":";
      print_practice_trial(root,choices[1],rules,weights,stream,table);
      std::cout << "}\n";
    }
  }
}

inline void run_archive_curriculum(const Rules& rules,const std::string& mode) {
  expect_value(rules.eliminate_exhausted_movers,"replay curriculum requires selfplay retirement rules");
  const bool scan=mode=="scan", train=mode=="train";
  expect_value(scan || train || mode=="assess" || mode=="practice","invalid archive curriculum mode");
  int count;
  ShortWeights weights{};
  uint64_t seed=0;
  expect_value(bool(std::cin>>count) && count>0 && count<=(scan?3500:96),"invalid curriculum case limit");
  if (!scan) {
    expect_value(bool(std::cin>>seed) && seed<=UINT64_MAX-512,"invalid curriculum seed");
    for (int i=0;i<5;++i) expect_value(bool(std::cin>>weights[i]) && std::isfinite(weights[i]),"invalid short-v1 weights");
  }
  std::vector<State> roots;
  for (int i=0;i<count;++i) {
    State root{};
    const auto error=curriculum_prefix(root,rules);
    if (scan) {
      const auto group=error ? error : curriculum_group(seat_decision(root));
      std::cout << "{\"group\":\"" << group << "\",\"seat\":" << (error?-1:int(root.to_move)) << "}\n";
    } else {
      if (error) throw std::runtime_error(error);
      roots.push_back(root);
    }
  }
  if (train) train_openings(rules,weights,seed,false,false,false,.015,32,true,true,0,&roots);
  else if (mode=="practice") practice_curriculum(roots,rules,weights,seed);
  else if (!scan) assess_curriculum(roots,rules,weights,seed);
}

}
