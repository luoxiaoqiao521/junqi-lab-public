#pragma once

struct GatePlan {
  jq::Move prepare;
  jq::Move receive;
  int partner;
  int flag;
};

static bool public_gate_reach(const jq::Observation& view,const jq::OpeningMarks& marks,
                              int partner,jq::Move move) {
  using namespace jq;
  if (view.seat_at(move.to)>=0) return false;
  const auto ranks=compatible_rank_mask(view,marks,move.from);
  if (!ranks || (ranks&immobile_ranks)) return false;
  for (const int rank : {Platoon,Engineer}) {
    const unsigned family=rank==Engineer?1u<<Engineer:~(1u<<Engineer);
    if (!(ranks&family)) continue;
    auto probe=view;
    probe.ranks[move.from]=static_cast<int8_t>(rank);
    std::vector<Move> actions;
    gen_piece_moves(probe,partner,move.from,actions);
    if (std::find(actions.begin(),actions.end(),move)==actions.end()) return false;
  }
  return true;
}

static bool threatened_friendly_flag(const jq::Observation& view,const jq::OpeningMarks& marks) {
  using namespace jq;
  for (int n=0;n<kNodes;++n) {
    const int owner=view.seat_at(n);
    if (owner>=0 && view.team(owner)==view.team(view.viewer) && Board::get().node(n).kind==NodeKind::HQ &&
        (compatible_rank_mask(view,marks,n)&(1u<<Flag)) && flag_exposure(view,marks,n)) return true;
  }
  return false;
}

static std::optional<GatePlan> gate_plan(const jq::SeatDecision& d,const jq::OpeningMarks& marks,
                                         jq::Move baseline) {
  using namespace jq;
  const auto& view=d.observation;
  if (view.ply>=140 || view.seat_at(baseline.to)>=0 || threatened_friendly_flag(view,marks)) return std::nullopt;
  const auto targets=flag_targets(view,marks);
  const auto base_after=project_quiet(view,baseline);
  const auto& board=Board::get();
  const int partner=(view.viewer+2)%kSeats;
  std::vector<GatePlan> new_routes;
  for (const int target : targets) for (const auto gate : board.road(target)) {
    if (!board.node(gate).rail || view.seat_at(gate)==partner) continue;
    for (int origin=0;origin<kNodes;++origin) {
      if (view.seat_at(origin)!=partner || !board.node(origin).rail) continue;
      const Move receive{static_cast<uint8_t>(origin),gate};
      if (public_gate_reach(view,marks,partner,receive) || public_gate_reach(base_after,marks,partner,receive)) continue;
      new_routes.push_back({{},receive,partner,target});
    }
  }
  if (new_routes.empty()) return std::nullopt;
  std::optional<GatePlan> best;
  int best_distance=100;
  for (const auto action : d.actions) {
    if (action==baseline || view.seat_at(action.to)>=0 || !board.node(action.from).rail ||
        board.node(action.to).kind==NodeKind::HQ) continue;
    const auto after=project_quiet(view,action);
    for (const auto& route : new_routes) {
      const int distance_to_flag=distance(action.to,route.flag);
      if (distance_to_flag>=best_distance || !public_gate_reach(after,marks,partner,route.receive)) continue;
      best={action,route.receive,partner,route.flag};
      best_distance=distance_to_flag;
    }
  }
  return best;
}

struct GateTrial {
  std::optional<GatePlan> offered;
  uint64_t prefix;
  int offered_at=0;
  // 0=no opportunity, 1=partner didn't act, 2=closed, 3=available, 4=used.
  int answer=0;
  bool observed=false;

  void turn(const jq::SeatDecision& d,jq::Move actual) {
    using namespace jq;
    if (!offered || observed || d.observation.viewer!=offered->partner) return;
    observed=true;
    const auto expected=offered->receive;
    const auto& view=d.observation;
    if (view.seat_at(expected.to)>=0 || view.seat_at(offered->flag)<0 ||
        std::find(d.actions.begin(),d.actions.end(),expected)==d.actions.end()) answer=2;
    else answer=actual==expected?4:3;
  }
};
