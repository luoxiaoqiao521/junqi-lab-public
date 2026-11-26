#pragma once

// Bounded decision evidence over shared public-compatible samples. A fixed
// follow-up must work across all sampled ongoing continuations, including
// observationally indistinguishable worlds.
struct PublicReplyProof {
  bool finishes=false, refuted=false;
};

static std::vector<jq::Move> terminal_moves(const jq::State& state,const jq::Rules& rules,int team) {
  using namespace jq;
  if (state.result!=Result::Ongoing || team_of(state.to_move,state.match)!=team) return {};
  const auto win=team==0?Result::Team0Wins:Result::Team1Wins;
  std::vector<Move> legal, ends;
  legal_moves(state,legal);
  for (const auto action:legal) {
    auto next=state;
    step(next,action,rules);
    if (next.result==win) ends.push_back(action);
  }
  return ends;
}

static void require_common(std::optional<std::vector<jq::Move>>& shared,std::vector<jq::Move> current) {
  if (!shared) {
    shared=std::move(current);
    return;
  }
  auto& moves=*shared;
  moves.erase(std::remove_if(moves.begin(),moves.end(),[&](const auto move) {
    return std::find(current.begin(),current.end(),move)==current.end();
  }),moves.end());
}

static PublicReplyProof public_reply_proof(const std::vector<jq::State>& worlds,jq::Move first,
                                         const jq::Rules& rules,int team) {
  using namespace jq;
  std::optional<std::vector<Move>> winning_tail, enemy_counter;
  const auto win=team==0?Result::Team0Wins:Result::Team1Wins;
  for (auto world:worlds) {
    single_case_check(is_legal(world,first),"sampled board changed the candidate's legality");
    step(world,first,rules);
    require_common(enemy_counter,terminal_moves(world,rules,1-team));
    if (world.result==win) continue;
    if (world.result!=Result::Ongoing || team_of(world.to_move,world.match)==team) {
      require_common(winning_tail,terminal_moves(world,rules,team));
      continue;
    }
    std::vector<Move> replies;
    legal_moves(world,replies);
    for (const auto reply:replies) {
      auto answered=world;
      step(answered,reply,rules);
      if (answered.result!=win) require_common(winning_tail,terminal_moves(answered,rules,team));
    }
  }
  return {!worlds.empty() && (!winning_tail || !winning_tail->empty()),
          enemy_counter && !enemy_counter->empty()};
}

struct PublicReplyCheck {
  std::array<PublicReplyProof,2> proof;
  int selected;
  unsigned flag_positions;
};

static PublicReplyCheck compare_public_replies(const jq::SeatDecision& d,const jq::OpeningMarks& marks,
    const jq::Rules& rules,const jq::ShortWeights& weights,const std::array<jq::Move,2>& pair) {
  using namespace jq;
  const auto& view=d.observation;
  for (auto candidate:pair)
    single_case_check(std::find(d.actions.begin(),d.actions.end(),candidate)!=d.actions.end(),
                      "candidate disappeared before scoring");
  single_case_check(short_features(view,pair[0])==short_features(view,pair[1]),"old encoding did not tie");
  const auto masses=short_action_masses(view,d.actions,weights);
  const auto index=[&](Move move) {
    return static_cast<size_t>(std::find(d.actions.begin(),d.actions.end(),move)-d.actions.begin());
  };
  single_case_check(masses[index(pair[0])]==masses[index(pair[1])],"frozen model did not tie");
  std::vector<State> worlds;
  unsigned flags=0;
  const auto seed=public_rollout_seed(view,marks,0);
  for (uint64_t i=0;i<16 && worlds.size()<8;++i) {
    const auto sample=public_sample(view,marks,mix_seed(seed,i));
    if (!sample) continue;
    if (sample->rank_at[116]==Flag) flags|=1;
    if (sample->rank_at[118]==Flag) flags|=2;
    worlds.push_back(*sample);
  }
  single_case_check(worlds.size()==8,"incomplete public sample budget");
  const auto left=public_reply_proof(worlds,pair[0],rules,view.team(view.viewer));
  const auto right=public_reply_proof(worlds,pair[1],rules,view.team(view.viewer));
  const int selected=left.finishes && right.refuted?0:right.finishes && left.refuted?1:-1;
  return {{left,right},selected,flags};
}

static jq::State reply_flag_root(bool reverse_counter,bool spare_hq=false) {
  using namespace jq;
  auto root=selfplay_fixture();
  root.out[3]=false;
  place_single_case_piece(root,1,51,Division);
  place_single_case_piece(root,2,111,Division);
  place_single_case_piece(root,3,0,Engineer);
  place_single_case_piece(root,4,22,Marshal);
  place_single_case_piece(root,5,5,Bomb);
  place_single_case_piece(root,6,10,Brigade);
  place_single_case_piece(root,7,15,Corps);
  place_single_case_piece(root,26,30,Marshal);
  place_single_case_piece(root,75,116,Flag);
  place_single_case_piece(root,76,90,Marshal);
  place_single_case_piece(root,reverse_counter?27:77,21,Engineer);
  if (spare_hq) place_single_case_piece(root,78,118,Mine);
  return root;
}

static PublicReplyCheck check_reply_case(const char* name,const jq::State& root,const jq::Rules& rules,
    const jq::ShortWeights& weights,const std::array<jq::Move,2>& pair,int expected) {
  using namespace jq;
  const auto d=seat_decision(root);
  auto marks=imagined_marks(d.observation);
  const auto& board=Board::get();
  for (int n=0;n<kNodes;++n) {
    const int owner=d.observation.seat_at(n);
    if (owner>=0 && (board.node(n).seat!=owner || board.node(n).r<4))
      marks.ranks[n]&=~(1u<<Mine);
  }
  single_case_check(!root.flag_revealed[1] && !root.flag_revealed[3],"scenario must keep both flags dark");
  const bool left_hq=board.node(pair[0].to).kind==NodeKind::HQ;
  if (left_hq) single_case_check(d.observation.rank_at(pair[0].to)==Unknown &&
       d.observation.rank_at(pair[1].to)==Unknown,"the sampled flag decision saw dark identities");
  const auto evidence=compare_public_replies(d,marks,rules,weights,pair);
  single_case_check(evidence.selected==expected,"public reply evidence produced the wrong ordering");
  if (expected>=0) {
    const int team=d.observation.team(d.observation.viewer);
    single_case_check(public_reply_proof({root},pair[expected],rules,team).finishes &&
                      public_reply_proof({root},pair[1-expected],rules,team).refuted,
                      "independent fixture board did not reproduce the public recommendation");
  }
  std::cout<<name<<' '<<evidence.selected<<' '<<evidence.flag_positions;
  for (const auto proof:evidence.proof) std::cout<<' '<<proof.finishes<<' '<<proof.refuted;
  std::cout<<'\n';
  return evidence;
}

static void public_reply_fixtures(const jq::Rules& rules,const jq::ShortWeights& weights) {
  using namespace jq;
  const std::array<Move,2> flags{{{51,56},{111,116}}};
  check_reply_case("dark_flag_counter_on_far_wing",reply_flag_root(false),rules,weights,flags,1);
  check_reply_case("dark_flag_counter_on_near_wing",reply_flag_root(true),rules,weights,flags,0);

  auto uncertain=reply_flag_root(false,true);
  const auto before=seat_decision(uncertain);
  const auto a=check_reply_case("decoy_same_observation",uncertain,rules,weights,flags,-1);
  auto captured=uncertain;
  const auto event=step(captured,flags[1],rules);
  single_case_check(uncertain.alive[76] && !uncertain.flag_revealed[3] &&
                    event.exit_reason[3]==ExitReason::FlagRemoved && captured.out[3] && !captured.alive[76],
                    "a dark true flag must retire its seat even with the marshal still alive");
  uncertain.rank_at[116]=uncertain.rank_of[75]=Mine;
  uncertain.rank_at[118]=uncertain.rank_of[78]=Flag;
  const auto after=seat_decision(uncertain);
  const auto b=check_reply_case("decoy_hidden_swap",uncertain,rules,weights,flags,-1);
  auto missed=uncertain;
  single_case_check(step(missed,flags[1],rules).exit_reason[3]==ExitReason::None && !missed.out[3],
                    "a nonflag headquarters capture was counted as flag removal");
  single_case_check(before.actions==after.actions &&
                    std::equal(std::begin(before.observation.ranks),std::end(before.observation.ranks),
                               std::begin(after.observation.ranks)) &&
                    a.flag_positions==3 && b.flag_positions==3,
                    "the sampler did not preserve both indistinguishable flag locations");

  auto clash=selfplay_fixture();
  place_single_case_piece(clash,1,20,Marshal);
  place_single_case_piece(clash,2,51,Engineer);
  place_single_case_piece(clash,3,5,Division);
  place_single_case_piece(clash,4,10,Brigade);
  place_single_case_piece(clash,26,21,Bomb);
  place_single_case_piece(clash,27,15,Platoon);
  check_reply_case("probe_and_keep_finisher",clash,rules,weights,{{{20,15},{20,21}}},1);
}
