#include "layout.h"
#include "state.h"
#include "public_flag_finish_cases.h"
#include "flag_objective_cases.h"
#include "selfplay_elimination_cases.h"
#include "last_mover_endgame_cases.h"
#include "bottom_and_rail_cases.h"
#include "relay_continuity_cases.h"
#include "public_relay_screen_cases.h"
#include "public_policy_value_cases.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void layouts() {
  for (uint64_t seed = 0; seed < 10000; ++seed) {
    jq::Rng a(seed), b(seed);
    int8_t x[jq::kPiecesPerSeat], y[jq::kPiecesPerSeat];
    jq::generate_layout(x, a); jq::generate_layout(y, b);
    require(jq::valid_layout(x), "generated invalid layout");
    require(std::memcmp(x, y, sizeof x) == 0, "seed not reproducible");
    x[0] = jq::Flag;
    require(!jq::valid_layout(x), "invalid deployment accepted");
  }
  require(jq::layout_slot_node(-1, 0) == jq::kNone, "seat bounds");
  require(jq::layout_slot_node(0, 25) == jq::kNone, "slot bounds");
}

void topology() {
  const auto& b = jq::Board::get();
  int camps = 0, hqs = 0, centers = 0;
  for (int n = 0; n < jq::kNodes; ++n) {
    const auto& p = b.node(n);
    require(b.at_grid(p.gr, p.gc) == n, "coordinate round trip");
    camps += p.kind == jq::NodeKind::Camp;
    hqs += p.kind == jq::NodeKind::HQ;
    centers += p.kind == jq::NodeKind::Center;
    for (int m : b.road(n)) {
      require(m >= 0 && m < jq::kNodes && m != n, "invalid edge");
      const auto& back = b.road(m);
      require(std::find(back.begin(), back.end(), n) != back.end(), "asymmetric road");
    }
  }
  require(camps == 20 && hqs == 8 && centers == 9, "point counts");
}

void collision_survivor() {
  jq::State s{};
  std::fill(std::begin(s.id_at), std::end(s.id_at), int8_t(jq::kNone));
  std::fill(std::begin(s.seat_at), std::end(s.seat_at), int8_t(jq::kNone));
  std::fill(std::begin(s.rank_at), std::end(s.rank_at), int8_t(jq::Unknown));
  auto put = [&](int id, int node, int rank) {
    s.alive[id] = true; s.node_of[id] = uint8_t(node);
    s.rank_of[id] = int8_t(rank); s.id_at[node] = int8_t(id);
    s.rank_at[node] = int8_t(rank); s.seat_at[node] = int8_t(id / 25);
  };
  const auto& b = jq::Board::get();
  int from = b.seat_node(0, 0, 0), to = b.seat_node(0, 0, 1);
  put(0, from, jq::Marshal); put(25, to, jq::Engineer);
  jq::Move m{uint8_t(from), uint8_t(to)};
  require(jq::is_legal(s, m), "fixture move illegal");
  auto info = jq::step(s, m, jq::Rules{});
  require(info.outcome == jq::Outcome::AttackerWins, "combat result");
  require(s.alive[0] && !s.alive[25], "survivor alive flag");
  require(s.id_at[to] == 0 && s.node_of[0] == to, "survivor placement");
  require(s.id_at[from] == jq::kNone, "source not cleared");
}

void archive_draw_boundary() {
  using namespace jq;
  Rng rng(17);
  int8_t board[kSeats][kPiecesPerSeat];
  for(auto& layout:board) generate_layout(layout,rng);
  Rules archive;
  archive.check_involved_no_move=true;
  archive.no_progress_limit=71;
  State state;
  init_state(state,board,0,archive);
  state.no_progress=69;
  for(int remaining : {70,71}) {
    std::vector<Move> moves;
    legal_moves(state,moves);
    auto quiet=std::find_if(moves.begin(),moves.end(),[&](Move move){return state.seat_at[move.to]==kNone;});
    require(quiet!=moves.end(),"archive fixture needs a quiet move");
    step(state,*quiet,archive);
    require(state.no_progress==remaining,"archive quiet counter");
    require(state.result==(remaining==70?Result::Ongoing:Result::Draw),"archive draw boundary");
  }
  Rules old;
  const auto before=rules_hash(old);
  old.check_involved_no_move=true;
  require(rules_hash(old)!=before,"archive timing must affect rules hash");
}

int main(int argc, char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--relay-continuity-fixtures") {
      jq::relay_continuity_cases();
      return 0;
    }
    if (argc==2 && std::string(argv[1])=="--public-relay-screen-fixtures") {
      jq::public_relay_screen_cases();
      return 0;
    }
    if (argc==3 && std::string(argv[1])=="--last-mover-endgame-fixtures") {
      jq::Rules rules;
      std::string error;
      if (!jq::parse_rules(jq::read_text(argv[2]),rules,error)) throw std::runtime_error(error);
      jq::last_mover_endgame_cases(rules);
      return 0;
    }
    if (argc==4 && std::string(argv[1])=="--public-policy-value-fixtures") {
      jq::Rules rules;
      std::string error;
      if (!jq::parse_rules(jq::read_text(argv[2]),rules,error)) throw std::runtime_error(error);
      jq::public_policy_value_cases(rules,argv[3]);
      return 0;
    }
    layouts(); topology(); collision_survivor(); archive_draw_boundary();
    jq::public_flag_finish_cases();
    jq::flag_objective_cases();
    jq::selfplay_elimination_cases();
    jq::last_mover_endgame_cases(jq::relay_test_rules());
    jq::bottom_and_rail_contract();
    jq::relay_continuity_cases();
    jq::public_relay_screen_cases();
    std::cout << "PASS layouts: 10000 seeds; topology invariants; collision survivor\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n'; return 1;
  }
}
