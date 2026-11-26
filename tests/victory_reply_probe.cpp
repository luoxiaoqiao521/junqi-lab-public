#include "selfplay_elimination_cases.h"
#include <iostream>
#include <optional>

namespace {
using namespace jq;

Move path(int from,int to) {
  return {static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
}

bool won(const State& settled,int team) {
  return settled.result==(team==0?Result::Team0Wins:Result::Team1Wins);
}

// Referee-side proof for one supplied hypothetical board. A caller cannot
// treat these witnesses as known facts across different hidden worlds.
std::optional<Move> immediate_win(const State& state,const Rules& rules,int team) {
  if (state.result!=Result::Ongoing || team_of(state.to_move,state.match)!=team) return std::nullopt;
  std::vector<Move> moves;
  legal_moves(state,moves);
  for (auto action:moves) {
    auto next=state;
    step(next,action,rules);
    if (won(next,team)) return action;
  }
  return std::nullopt;
}

struct Replies {
  int total=0, attacker_wins=0, defender_wins=0, draws=0, unknown=0, beyond=0;
};

Replies examine_replies(const State& after,const Rules& rules,int attackers) {
  Replies result;
  if (after.result!=Result::Ongoing) return result;
  std::vector<Move> responses;
  legal_moves(after,responses);
  for (auto action:responses) {
    auto next=after;
    step(next,action,rules);
    ++result.total;
    if (won(next,1-attackers)) ++result.defender_wins;
    else if (won(next,attackers) || immediate_win(next,rules,attackers)) ++result.attacker_wins;
    else if (next.result==Result::Draw) ++result.draws;
    else if (next.result==Result::Truncated) ++result.unknown;
    else ++result.beyond;
  }
  return result;
}

State advantage_root() {
  auto s=selfplay_fixture();
  place_single_case_piece(s,1,51,Marshal);
  place_single_case_piece(s,2,111,Brigade);
  place_single_case_piece(s,3,22,Corps);
  place_single_case_piece(s,4,0,Engineer);
  place_single_case_piece(s,5,6,Bomb);
  place_single_case_piece(s,6,10,Regiment);
  place_single_case_piece(s,7,15,Division);
  place_single_case_piece(s,26,30,Platoon);
  return s;
}

void add_counter_wing(State& s) {
  s.out[3]=false;
  place_single_case_piece(s,75,116,Flag);
  place_single_case_piece(s,76,21,Engineer);
  place_single_case_piece(s,77,90,Platoon);
}

State report(const char* name,const State& root,Move first,const Rules& rules) {
  single_case_check(is_legal(root,first),"illegal first move in lesson");
  auto after=root;
  const auto event=step(after,first,rules);
  const int attackers=team_of(root.to_move,root.match);
  const auto r=examine_replies(after,rules,attackers);
  const auto counts=std::count(std::begin(root.seat_at),std::end(root.seat_at),root.to_move);
  const auto occupied=std::count_if(std::begin(root.seat_at),std::end(root.seat_at),[](int seat){return seat>=0;});
  std::cout<<name<<' '<<int(first.from)<<' '<<int(first.to)<<' '<<counts<<' '<<occupied-counts<<' '
           <<int(event.eliminated_mask)<<' '<<int(after.result)<<' '<<r.total<<' '<<r.attacker_wins<<' '
           <<r.defender_wins<<' '<<r.draws<<' '<<r.unknown<<' '<<r.beyond<<'\n';
  return after;
}

void flag_race(const Rules& rules) {
  const auto front=path(51,56),wing=path(111,116),counter=path(21,26);
  auto root=advantage_root();
  place_single_case_piece(root,27,21,Engineer);
  single_case_check(won(report("finish_now",root,front,rules),0),"final enemy flag did not end the game");

  root=advantage_root();
  add_counter_wing(root);
  const auto after_front=report("same_attack_live_counter",root,front,rules);
  single_case_check(after_front.result==Result::Ongoing && after_front.to_move==3 &&
                    is_legal(after_front,counter) && immediate_win(after_front,rules,1)==counter,
                    "other enemy did not retain the decisive reply");
  auto missed=after_front;
  const auto quiet=path(21,20);
  single_case_check(is_legal(missed,quiet),"alternative defender reply unavailable");
  step(missed,quiet,rules);
  single_case_check(immediate_win(missed,rules,0)==wing,"the missed defence did not allow the follow-up");

  const auto after_wing=report("counter_wing_first",root,wing,rules);
  single_case_check(after_wing.out[3] && after_wing.result==Result::Ongoing &&
                    after_wing.to_move==1 && !immediate_win(after_wing,rules,1),
                    "eliminating the counter wing did not remove its threat");
}

void decisive_exchange(const Rules& rules) {
  auto root=selfplay_fixture();
  place_single_case_piece(root,1,4,Bomb);
  place_single_case_piece(root,2,23,Engineer);
  place_single_case_piece(root,3,0,Marshal);
  place_single_case_piece(root,4,6,Brigade);
  place_single_case_piece(root,5,10,Division);
  place_single_case_piece(root,6,15,Company);
  place_single_case_piece(root,26,30,Platoon);
  const auto hit=path(4,30);
  single_case_check(won(report("exchange_last_mover",root,hit,rules),0),
                    "sacrificing the bomb for the last mover was not scored as a win");
  place_single_case_piece(root,27,21,Engineer);
  place_single_case_piece(root,28,35,Platoon);
  const auto after=report("same_exchange_with_reserve",root,hit,rules);
  single_case_check(after.result==Result::Ongoing && immediate_win(after,rules,1)==path(21,26),
                    "the surviving attackers did not refute the speculative trade");
}
}

int main(int argc,char** argv) {
  try {
    if (argc!=2) return 2;
    jq::Rules rules;
    std::string error;
    if (!jq::parse_rules(jq::read_text(argv[1]),rules,error) ||
        !rules.ccw || !rules.flag_eliminates || !rules.eliminate_exhausted_movers)
      throw std::runtime_error("this lesson requires the frozen selfplay rules");
    flag_race(rules);
    decisive_exchange(rules);
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
