#pragma once

struct SeatFinish {
  int flag=jq::kNone, last=jq::kNone, movable=0;
};

using RouteReadiness=std::array<std::array<int,jq::kNodes>,jq::kSeats>;

static int paired_readiness(const RouteReadiness& ready,int first,int second) {
  const auto& a=ready[first];
  int best=5 * *std::max_element(a.begin(),a.end());
  if (second<0) return best;
  const auto& b=ready[second];
  best=std::max(best,5 * *std::max_element(b.begin(),b.end()));
  for (int i=0;i<jq::kNodes;++i) if (a[i])
    for (int j=0;j<jq::kNodes;++j) if (i!=j && b[j])
      best=std::max(best,5*std::max(a[i],b[j])+4*std::min(a[i],b[j]));
  return best;
}

static RouteReadiness legal_finish_routes(const jq::State& scenario,const jq::Rules& rules) {
  using namespace jq;
  SeatFinish goals[kSeats]{};
  int reserves[2]{}, active[2]{};
  int gates[kNodes]{};
  for (int seat=0;seat<kSeats;++seat) active[team_of(seat)]+=!scenario.out[seat];
  for (int n=0;n<kNodes;++n) if (scenario.seat_at[n]>=0) {
    const int seat=scenario.seat_at[n], rank=scenario.rank_at[n];
    if (rank==Flag) {
      goals[seat].flag=n;
      for (const int gate:Board::get().road(n)) gates[gate]|=1<<seat;
    }
    if (can_leave_node(rank,n)) {
      goals[seat].last=n;
      ++goals[seat].movable;
      ++reserves[team_of(seat)];
    }
  }
  RouteReadiness ready{};
  std::vector<Move> moves;
  const StateView view{scenario};
  for (int n=0;n<kNodes;++n) {
    const int actor=scenario.seat_at[n];
    if (actor<0 || !can_leave_node(scenario.rank_at[n],n)) continue;
    const int team=team_of(actor);
    moves.clear();
    gen_piece_moves(view,actor,n,moves);
    for (const auto move:moves) {
      const int owner=scenario.seat_at[move.to];
      const auto effect=owner<0?Outcome::Move:resolve(scenario.rank_at[n],scenario.rank_at[move.to],rules);
      const bool survives=effect==Outcome::Move || effect==Outcome::AttackerWins;
      if (effect==Outcome::DefenderWins) continue;
      if (owner>=0) {
        const auto& goal=goals[owner];
        const bool flag=rules.flag_eliminates && move.to==goal.flag;
        const bool final_mover=rules.eliminate_exhausted_movers && goal.movable==1 && move.to==goal.last;
        if ((flag || final_mover) && (survives || reserves[team]>1 || (flag && active[1-team]==1)))
          ready[owner][n]=std::max(ready[owner][n],4+int(scenario.to_move==actor));
      }
      if (survives && gates[move.to]) for (int foe=1-team;foe<kSeats;foe+=2)
        if (gates[move.to]&(1<<foe)) ready[foe][n]=std::max(ready[foe][n],2);
    }
  }
  return ready;
}

static int team_readiness(const jq::State& scenario,const RouteReadiness& ready,int team,bool two=true) {
  const int a=1-team,b=a+2;
  if (scenario.out[a]) return scenario.out[b]?0:paired_readiness(ready,b,jq::kNone);
  if (!two && !scenario.out[b])
    return std::max(paired_readiness(ready,a,jq::kNone),paired_readiness(ready,b,jq::kNone));
  return paired_readiness(ready,a,scenario.out[b]?jq::kNone:b);
}

enum class RoutePriority { Old, Joint, FirstThenContinue };

static double readiness_term(const jq::State& sample,const jq::Rules& rules,int team,
                             bool first_exit,RoutePriority priority) {
  const auto routes=legal_finish_routes(sample,rules);
  const int all=team_readiness(sample,routes,team)-team_readiness(sample,routes,1-team);
  if (priority!=RoutePriority::FirstThenContinue || !first_exit) return .045*all;
  const int first=team_readiness(sample,routes,team,false)-team_readiness(sample,routes,1-team,false);
  return .045*first+.010*(all-first);
}

static void two_target_fixtures(const jq::Rules& rules) {
  using namespace jq;
  RouteReadiness values{};
  values[1][4]=4;
  values[3][4]=4;
  single_case_check(paired_readiness(values,1,3)==20,"single resource counted on two wings");
  values[3][30]=2;
  single_case_check(paired_readiness(values,1,3)==28,"distinct support did not improve the second route");
  auto root=selfplay_fixture();
  root.out[3]=false;
  single_case_check(team_readiness(root,values,0,false)==20 && team_readiness(root,values,0)==28,
                    "prioritizing the first target changed its legal resources");
  root.out[3]=true;
  place_single_case_piece(root,1,4,Bomb);
  place_single_case_piece(root,2,23,Engineer);
  place_single_case_piece(root,26,30,Platoon);
  const auto routes=legal_finish_routes(root,rules);
  single_case_check(routes[1][4]==5,"real last-mover attack failed to count as an alternative win route");
  const auto event=step(root,{4,30},rules);
  single_case_check(event.eliminated_mask==2 && root.result==Result::Team0Wins &&
                    team_readiness(root,legal_finish_routes(root,rules),0)==0,"retired target was not cleared from route scoring");
}
