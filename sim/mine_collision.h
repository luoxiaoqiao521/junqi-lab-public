#pragma once
#include "mine_evidence.h"

namespace jq {
struct CollisionMineWitness {
  std::array<bool,kNodes> excluded{};
  int new_exclusions=0;
  void advance(const Observation& before,const Observation& after,Move move,Outcome outcome,
               const OpeningMarks& marks,const Rules& rules) {
    excluded[move.from]=false;
    if (outcome!=Outcome::DefenderWins) excluded[move.to]=false;
    constexpr unsigned mine=1u << Mine;
    if (outcome==Outcome::DefenderWins && before.viewer==before.turn &&
        after.seat_at(move.to)==before.seat_at(move.to) &&
        before.rank_at(move.to)==Unknown && (marks.ranks[move.to] & mine) &&
        !(public_defender_ranks(before.rank_at(move.from),outcome,rules) & mine)) {
      if (!excluded[move.to]) ++new_exclusions;
      excluded[move.to]=true;
    }
    for (int node=0; node<kNodes; ++node)
      if (after.seat_at(node)<0) excluded[node]=false;
  }
};

inline void mine_collision_fixture(const Rules& rules) {
  Observation before{};
  std::fill(std::begin(before.seats),std::end(before.seats),int8_t(-1));
  std::fill(std::begin(before.ranks),std::end(before.ranks),int8_t(Unknown));
  const auto& board=Board::get();
  const Move attack{uint8_t(board.seat_node(1,4,1)),uint8_t(board.seat_node(1,4,0))};
  before.seats[attack.from]=0;
  before.seats[attack.to]=1;
  before.ranks[attack.from]=Engineer;
  std::vector<Move> actions;
  gen_piece_moves(before,0,attack.from,actions);
  expect_value(std::find(actions.begin(),actions.end(),attack)!=actions.end(),"illegal engineer probe fixture");
  auto after=before;
  after.seats[attack.from]=-1;
  after.ranks[attack.from]=Unknown;
  OpeningMarks base(before,true);
  CollisionMineWitness marks, blinded;
  const auto lost=resolve(Engineer,Company,rules);
  expect_value(lost==Outcome::DefenderWins && resolve(Engineer,Mine,rules)==Outcome::AttackerWins,
               "mine loss fixture does not apply to current rules");
  marks.advance(before,after,attack,lost,base,rules);
  before.ranks[attack.from]=Unknown;
  blinded.advance(before,after,attack,lost,base,rules);
  expect_value(marks.excluded[attack.to] && !blinded.excluded[attack.to],
               "defender evidence changed without a known attacker");
  marks.advance(before,after,attack,Outcome::AttackerWins,base,rules);
  expect_value(!marks.excluded[attack.to],"new defender inherited another piece's evidence");
}

inline void mine_collision_evidence(const Rules& rules,uint64_t seed,int learner,int variant,
                                    const ShortWeights& weights) {
  mine_collision_fixture(rules);
  Rng layout_rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State state;
  init_state(state,layouts,first,rules);
  auto streams=review_streams(seed);
  OpeningMarks base(observe(state,learner),true);
  CollisionMineWitness additional;
  int decisions=0, attack_turns=0, changed_turns=0, opportunities=0;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    if (state.to_move==learner) {
      bool hit=false, changes=false;
      ++decisions;
      for (const auto a : d.actions) {
        if (!additional.excluded[a.to] || d.observation.rank_at(a.to)!=Unknown) continue;
        const auto prior=base.ranks[a.to];
        constexpr unsigned mine=1u << Mine;
        if (!(prior & mine)) continue;
        expect_value(state.rank_at[a.to]!=Mine,"public collision excluded the actual mine");
        hit=true;
        ++opportunities;
        const int rank=d.observation.rank_at(a.from);
        changes |= collision_outcome_mask(rank,prior,rules)!=collision_outcome_mask(rank,prior & ~mine,rules);
      }
      attack_turns+=hit;
      changed_turns+=changes;
    }
    const auto a=review_move(d,learner,variant,true,weights,streams[state.to_move]);
    const auto before=observe(state,learner);
    const auto e=step(state,a,rules);
    const auto after=observe(state,learner);
    additional.advance(before,after,a,e.outcome,base,rules);
    advance_public_ranks(base,before,after,a,e.outcome);
  }
  std::cout << "{\"seed\":" << seed << ",\"seat\":" << learner << ",\"variant\":" << variant
            << ",\"decisions\":" << decisions << ",\"exclusions\":" << additional.new_exclusions
            << ",\"opportunities\":" << opportunities << ",\"attack_turns\":" << attack_turns
            << ",\"changed_turns\":" << changed_turns << "}\n";
}
}
