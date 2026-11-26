#include "../sim/replay_deadline.h"
#include "archive_replay_prefix.h"

using namespace jq;
using namespace jq::archive_probe;

struct Trial {
  State at;
  Move actual, defense, ally, follow;
  std::vector<Move> ending;
  Rules rules;
  ShortWeights weights;
  int target;
};

static Move leg(Row row) {
  return {static_cast<uint8_t>(point(row.x,row.y)),static_cast<uint8_t>(point(row.xx,row.yy))};
}

static Move frozen_other(const State& state,Move exclude,const ShortWeights& weights) {
  const auto d=seat_decision(state);
  const auto scores=review_learned(d,weights);
  double best=-1;
  Move selected=exclude;
  for(size_t i=0;i<d.actions.size();++i)
    if(!(d.actions[i]==exclude) && scores[i]>best) {
      best=scores[i]; selected=d.actions[i];
    }
  expect_value(!(selected==exclude),"no competing action in replay");
  return selected;
}

static std::optional<Move> team_win_offer(const State& state,int attacker,const Rules& rules) {
  if(state.result!=Result::Ongoing || state.to_move!=attacker) return std::nullopt;
  const auto player=seat_decision(state);
  if(player.actions.empty()) return std::nullopt;
  return public_flag_finish(player,player.actions.front(),rules);
}

static void defender_paths(const Trial& game,int world,int root,Move first) {
  State at=game.at;
  const int front_id=at.id_at[game.target];
  const int front_seat=front_id/kPiecesPerSeat;
  step(at,first,game.rules);
  expect_value(at.result==Result::Ongoing && team_of(at.to_move)!=team_of(game.at.to_move),
               "no defending turn after this root");
  std::vector<Move> replies;
  legal_moves(at,replies);
  int front_removed=0, ignored=0, attacks_flag=0, bridge=0, side_legal=0, side_survives=0;
  int while_blocked_legal=0, while_blocked_survives=0;
  for(auto reaction:replies) {
    State after=at;
    const bool counter=observe(at,at.to_move).rank_at(reaction.to)==Flag;
    step(after,reaction,game.rules);
    const bool neutralized=!after.alive[front_id];
    front_removed+=neutralized;
    ignored+=bool(team_win_offer(after,front_seat,game.rules));
    attacks_flag+=counter;
    if(after.to_move!=front_seat || !is_legal(after,game.ally)) continue;
    ++bridge;
    step(after,game.ally,game.rules);
    if(after.to_move!=game.at.to_move || !is_legal(after,game.follow)) continue;
    ++side_legal;
    const auto event=step(after,game.follow,game.rules);
    const bool survived=event.outcome==Outcome::AttackerWins || event.outcome==Outcome::Move;
    side_survives+=survived;
    while_blocked_legal+=neutralized;
    while_blocked_survives+=neutralized && survived;
  }
  std::cout<<"R "<<world<<' '<<root<<' '<<replies.size()<<' '<<front_removed<<' '<<ignored<<' '
           <<attacks_flag<<' '<<bridge<<' '<<side_legal<<' '<<side_survives<<' '
           <<while_blocked_legal<<' '<<while_blocked_survives<<'\n';
}

static bool same_input(const State& a,const State& b) {
  const auto x=seat_decision(a), y=seat_decision(b);
  return x.observation.viewer==y.observation.viewer && x.actions==y.actions &&
         std::equal(std::begin(x.observation.ranks),std::end(x.observation.ranks),y.observation.ranks) &&
         std::equal(std::begin(x.observation.seats),std::end(x.observation.seats),y.observation.seats);
}

static bool untouched(const std::vector<Row>& prefix,int node) {
  return std::none_of(prefix.begin(),prefix.end(),[=](const Row& row) {
    return row.x>=0 && (point(row.x,row.y)==node || point(row.xx,row.yy)==node);
  });
}

static State alternate_world(const Trial& game,int8_t (&layout)[4][25],const std::vector<Row>& prefix,
                             const MovedSquares& moves) {
  const int target=game.follow.to;
  const int enemy=game.at.seat_at[target];
  expect_value(enemy>=0 && game.at.rank_at[target]==Mine && untouched(prefix,target),"corner mine fixture");
  const auto slot=static_cast<size_t>(game.at.id_at[target]%kPiecesPerSeat);
  for(int i=0;i<kPiecesPerSeat;++i) {
    const int node=layout_slot_node(enemy,i);
    const int rank=layout[enemy][i];
    if(rank<Platoon || game.at.rank_at[node]!=rank || !untouched(prefix,node) ||
       !deployment_rank_allowed(node,Mine) || node==game.defense.from || node==game.defense.to ||
       node==game.ally.from || node==game.ally.to) continue;
    std::swap(layout[enemy][i],layout[enemy][slot]);
    State shadow{};
    MovedSquares trail{};
    Move first;
    const bool ok=archive_prefix(shadow,trail,first,game.rules,layout,prefix);
    std::swap(layout[enemy][i],layout[enemy][slot]);
    if(ok && first==game.actual && moves==trail && same_input(game.at,shadow)) return shadow;
  }
  throw std::runtime_error("no compatible corner-mine variant");
}

static State after_source_defense(const Trial& game,State state) {
  for(auto action:{game.actual,game.defense,game.ally}) {
    if(!is_legal(state,action))
      throw std::runtime_error("archive bridge mismatch at ply "+std::to_string(state.ply+1)+
                               " for "+coords(action)+", turn "+std::to_string(state.to_move)+
                               ", origin seat "+std::to_string(state.seat_at[action.from]));
    step(state,action,game.rules);
  }
  expect_value(state.to_move==game.at.to_move && is_legal(state,game.follow),"no observed next move");
  return state;
}

static void paired(const State& start,const Trial& game,int world,const char* phase,Move original,Move other) {
  expect_value(is_legal(start,original) && is_legal(start,other),"paired move not legal");
  for(int response:{0,1,2}) for(int seed=0;seed<32;++seed) {
    const auto common=mix_seed(408300+start.ply,seed);
    const auto a=review_rollout(start,original,game.rules,response,true,game.weights,common);
    const auto b=review_rollout(start,other,game.rules,response,true,game.weights,common);
    std::cout<<"P "<<phase<<' '<<world<<' '<<response<<' '<<seed<<' '
             <<review_outcome(a,start.to_move)<<' '<<review_outcome(b,start.to_move)<<'\n';
  }
}

static void examine(const Trial& game,const State& changed) {
  const auto first_other=frozen_other(game.at,game.actual,game.weights);
  const auto source_after=after_source_defense(game,game.at);
  const auto changed_after=after_source_defense(game,changed);
  expect_value(same_input(source_after,changed_after),"compatible world leaked before the follow-up");
  const auto later_other=frozen_other(source_after,game.follow,game.weights);
  auto end=source_after;
  for(auto move:game.ending) {
    expect_value(is_legal(end,move),"recorded finish is not legal");
    step(end,move,game.rules);
  }
  expect_value(team_score(end.result,game.at.to_move)==1.,"recorded 119th move did not win");
  for(int world:{0,1}) {
    const auto& decision=world==0?game.at:changed;
    const auto& later=world==0?source_after:changed_after;
    auto actual=later;
    const auto defender=actual.rank_at[game.follow.to];
    const auto step_info=step(actual,game.follow,game.rules);
    std::cout<<"C "<<world<<' '<<int(step_info.outcome)<<' '<<int(defender)<<' '<<coords(game.actual)<<' '
             <<coords(first_other)<<' '<<coords(game.follow)<<' '<<coords(later_other)<<'\n';
    auto case_game=game;
    case_game.at=decision;
    defender_paths(case_game,world,0,game.actual);
    defender_paths(case_game,world,1,first_other);
    paired(decision,game,world,"start",game.actual,first_other);
    paired(later,game,world,"continue",game.follow,later_other);
  }
}

int main() {
  try {
    int entries;
    Trial game{};
    if(!(std::cin>>entries)||entries<116||entries>400) return 2;
    for(int i=0;i<5;++i) if(!(std::cin>>game.weights[i])||!std::isfinite(game.weights[i])) return 2;
    int8_t layouts[4][25]{};
    if(!read_layouts(layouts)) return 2;
    std::vector<Row> prefix;
    int ply=0;
    for(int j=0;j<entries;++j) {
      Row row;
      if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return 2;
      if(row.x>=0) ++ply;
      if(ply<=112 || (ply==113 && row.x>=0)) prefix.push_back(row);
      if(row.x<0) continue;
      if(ply==112) game.target=leg(row).to;
      if(ply==114) game.defense=leg(row);
      if(ply==115) game.ally=leg(row);
      if(ply==116) game.follow=leg(row);
      if(ply>=116) game.ending.push_back(leg(row));
    }
    game.rules.ccw=false;
    game.rules.no_progress_limit=71;
    game.rules.check_involved_no_move=true;
    game.rules.marshal_reveals_flag=true;
    MovedSquares marks{};
    expect_value(ply>=119 && archive_prefix(game.at,marks,game.actual,game.rules,layouts,prefix),
                 "000433 prefix failed native verification");
    const auto shadow=alternate_world(game,layouts,prefix,marks);
    examine(game,shadow);
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 3;
  }
}
