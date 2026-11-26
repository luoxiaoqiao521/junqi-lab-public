#pragma once

struct BombWork {
  int contacts=0, low=0, large=0, mine=0, flag=0;
  int offered=0, considered=0, kept_critical=0, changed=0;
  int first=0;
  uint64_t prefix=0;
};

constexpr unsigned large_ranks=(1u<<jq::Division)|(1u<<jq::Corps)|(1u<<jq::Marshal);

static bool mobile_mask(const jq::Observation& view,const jq::OpeningMarks& marks,int node) {
  return jq::can_leave_node(jq::Engineer,node) && (compatible_rank_mask(view,marks,node)&~immobile_ranks);
}

static bool preserve_exchange(const jq::Observation& view,const jq::OpeningMarks& marks,jq::Move hit) {
  using namespace jq;
  const auto rank=compatible_rank_mask(view,marks,hit.to);
  if (rank && (rank&~(large_ranks|(1u<<Flag)))==0) return true;
  const int enemy=view.seat_at(hit.to);
  const int team=view.team(view.viewer);
  bool another_mobile=false, friendly_reserve=false;
  for (int n=0;n<kNodes;++n) {
    const int owner=view.seat_at(n);
    if (owner<0) continue;
    if (owner==enemy && n!=hit.to && mobile_mask(view,marks,n)) another_mobile=true;
    if (view.team(owner)!=team) continue;
    if (owner!=view.viewer || (n!=hit.from && mobile_mask(view,marks,n))) friendly_reserve=true;
  }
  if (!another_mobile && friendly_reserve && mobile_mask(view,marks,hit.to)) return true;
  if (!mobile_mask(view,marks,hit.to)) return false;
  for (const int node:Board::get().road(hit.to)) {
    const int owner=view.seat_at(node);
    if (owner>=0 && view.team(owner)==team && Board::get().node(node).kind==NodeKind::HQ &&
        (compatible_rank_mask(view,marks,node)&(1u<<Flag))) return true;
  }
  return false;
}

static jq::Move bomb_balance(const jq::SeatDecision& d,const jq::OpeningMarks& marks,
    const jq::PublicReplyHistory& prefix,jq::Move hit,
    jq::Move last,const jq::Rules& rules,const jq::ShortWeights& peer,int clock,BombWork& work) {
  using namespace jq;
  if (preserve_exchange(d.observation,marks,hit)) {++work.kept_critical; return hit;}
  if (work.considered>=7) return hit;
  SeatDecision guarded{d.observation,{}};
  for (const auto candidate:d.actions)
    if (d.observation.rank_at(candidate.from)!=Bomb || d.observation.seat_at(candidate.to)<0)
      guarded.actions.push_back(candidate);
  if (guarded.actions.empty()) return hit;
  ++work.considered;
  const auto alternate=plan(guarded,marks,guarded.actions.front(),last,rules);
  const auto seed=public_rollout_seed(d.observation,marks,clock);
  const auto evidence=prefix.for_actor(d.observation,marks);
  int count=0;
  ResponseValue original,held;
  for (uint64_t i=0;i<12 && count<4;++i) {
    const auto sample=public_sample(d.observation,evidence,mix_seed(seed,i));
    if (!sample) continue;
    const auto rng=mix_seed(seed,i+107);
    const int context=count%2;
    response_trial(*sample,hit,evidence,prefix,rules,peer,context,rng,original,false);
    response_trial(*sample,alternate,evidence,prefix,rules,peer,context,rng,held,false);
    ++count;
  }
  if (count!=4 || held.wins<original.wins || held.losses>original.losses || held.value<original.value+.24) return hit;
  return alternate;
}

static void bomb_fixtures(const jq::Rules& rules) {
  using namespace jq;
  auto root=selfplay_fixture();
  place_single_case_piece(root,1,4,Bomb);
  place_single_case_piece(root,2,23,Engineer);
  place_single_case_piece(root,26,30,Platoon);
  const Move hit{4,30},quiet{4,3};
  auto view=observe(root,0);
  OpeningMarks marks(view,false);
  marks.ranks[30]=1u<<Platoon;
  auto captured=root;
  single_case_check(preserve_exchange(view,marks,hit) && step(captured,hit,rules).eliminated_mask==2 &&
                    captured.result==Result::Team0Wins,"low last mover must preserve the winning bomb exchange");
  place_single_case_piece(root,27,52,Engineer);
  view=observe(root,0);
  marks.ranks[52]=1u<<Engineer;
  single_case_check(!preserve_exchange(view,marks,hit),"low target with surviving enemy reserve was treated as certain value");
  marks.ranks[30]=(1u<<Marshal)|(1u<<Platoon);
  single_case_check(!preserve_exchange(view,marks,hit),"possible large piece was promoted to proven large piece");
  marks.ranks[30]=(1u<<Marshal)|(1u<<Corps);
  single_case_check(preserve_exchange(view,marks,hit),"proven large target was not retained");
  single_case_check(is_legal(root,quiet) && step(root,quiet,rules).outcome==Outcome::Move && root.alive[1] &&
                    root.no_progress==1,"a bomb move to an empty square must stay a quiet move");
}
