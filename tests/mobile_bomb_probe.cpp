#include "layout.h"
#include "state.h"
#include "../observe/filter.h"
#include "../strategy/heuristic.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace jq;

struct OutcomeSummary {
  int result=0, plies=0;
  bool bomb_moved=false;
};

static std::pair<int,int> swap_hq_bomb(int8_t (&layout)[25],int seat,Rng& rng) {
  const auto& board=Board::get();
  int hq=kNone;
  std::vector<int> alternatives;
  for(int slot=0;slot<25;++slot) {
    const int node=layout_slot_node(seat,slot);
    if(board.node(node).kind==NodeKind::HQ) {
      if(layout[slot]==Bomb) hq=slot;
    } else if(layout[slot]==Platoon && deployment_rank_allowed(node,Bomb))
      alternatives.push_back(slot);
  }
  if(hq==kNone||alternatives.empty()) return {kNone,kNone};
  const int choice=alternatives[rng.below(static_cast<uint32_t>(alternatives.size()))];
  std::swap(layout[hq],layout[choice]);
  if(!valid_layout(layout)) throw std::runtime_error("invalid opening swap");
  return {hq,choice};
}

static bool other_views_equal(const State& a,const State& b,int changed_seat) {
  for(int seat=0;seat<4;++seat) {
    if(seat==changed_seat) continue;
    const auto left=observe(a,seat), right=observe(b,seat);
    if(!std::equal(std::begin(left.ranks),std::end(left.ranks),std::begin(right.ranks))) return false;
  }
  return true;
}

static OutcomeSummary play(State s,const Rules& rules,int seat,int context,int tracked_slot,uint64_t seed) {
  std::array<Rng,4> streams{Rng(mix_seed(seed,0)),Rng(mix_seed(seed,1)),
                            Rng(mix_seed(seed,2)),Rng(mix_seed(seed,3))};
  OutcomeSummary summary;
  while(s.result==Result::Ongoing) {
    const int actor=s.to_move;
    auto obs=observe(s,actor);
    Move move;
    bool guided=context && (actor==seat || actor==(seat+3)%4);
    if(!(guided?heuristic_move(obs,streams[actor],move):random_move(obs,streams[actor],move)))
      throw std::runtime_error("opening probe legal-action mismatch");
    summary.bomb_moved |= actor==seat && s.id_at[move.from]==seat*25+tracked_slot;
    step(s,move,rules);
  }
  summary.result=s.result==Result::Truncated?-1:s.result==Result::Draw?1:
    int(s.result==(seat%2?Result::Team1Wins:Result::Team0Wins))*2;
  summary.plies=s.ply;
  return summary;
}

int main() {
  Rules rules;
  rules.ccw=false;
  rules.marshal_reveals_flag=true;
  rules.check_involved_no_move=true;
  rules.no_progress_limit=71;
  constexpr int count=256;
  int accepted=0;
  for(uint64_t seed=312000;seed<330000&&accepted<count;++seed) {
    const int seat=static_cast<int>(seed%4);
    Rng rng(seed);
    int8_t layouts[4][25];
    for(auto& layout:layouts) generate_layout(layout,rng);
    const int first=static_cast<int>(rng.below(4));
    State before,after;
    init_state(before,layouts,first,rules);
    const auto [old_slot,new_slot]=swap_hq_bomb(layouts[seat],seat,rng);
    if(old_slot==kNone) continue;
    init_state(after,layouts,first,rules);
    if(!other_views_equal(before,after,seat)) return 3;
    for(int context=0;context<2;++context) {
      auto a=play(before,rules,seat,context,old_slot,seed);
      auto b=play(after,rules,seat,context,new_slot,seed);
      std::cout<<seed<<' '<<seat<<' '<<context<<' '<<a.result<<' '<<b.result<<' '
               <<a.plies<<' '<<b.plies<<' '<<a.bomb_moved<<' '<<b.bomb_moved<<'\n';
    }
    ++accepted;
  }
  return accepted==count?0:2;
}
