#include "../sim/replay_deadline.h"
#include "archive_replay_prefix.h"

using namespace jq;
using namespace jq::archive_probe;

static int terminal_or_warning(State state,Move action,const Rules& rules) {
  const int learner=state.to_move;
  step(state,action,rules);
  const auto score=team_score(state.result,learner);
  if(score==1.) return 2;
  if(score==0.) return -2;
  if(state.result!=Result::Ongoing||team_of(learner)==team_of(state.to_move)) return 0;
  std::vector<Move> replies;
  legal_moves(state,replies);
  for(auto reply:replies)
    if(state.rank_at[reply.to]==Flag) return -1;
  return 0;
}

static void verify_dark_hq(const State& original,const MovedSquares& moved,Move focus,const Rules& rules,
                           int8_t (&layouts)[4][25],const std::vector<Row>& rows) {
  const auto& board=Board::get();
  const auto& goal=board.node(focus.to);
  const auto twin=board.seat_node(goal.seat,5,4-goal.c);
  for(size_t i=0;i+1<rows.size();++i)
    expect_value(rows[i].x<0 || (point(rows[i].xx,rows[i].yy)!=focus.to &&
                                   point(rows[i].xx,rows[i].yy)!=twin),"dark HQ was touched");
  std::swap(layouts[goal.seat][20+goal.c],layouts[goal.seat][24-goal.c]);
  State altered{};
  MovedSquares alt_moved{};
  Move alt_focus;
  expect_value(archive_prefix(altered,alt_moved,alt_focus,rules,layouts,rows,false),"invalid dark HQ prefix");
  const auto a=seat_decision(original),b=seat_decision(altered);
  expect_value(std::equal(std::begin(a.observation.ranks),std::end(a.observation.ranks),b.observation.ranks) &&
               std::equal(std::begin(a.observation.seats),std::end(a.observation.seats),b.observation.seats) &&
               a.observation.turn==b.observation.turn && a.actions==b.actions && moved==alt_moved &&
               focus==alt_focus,"dark HQ changed the decision input");
  const auto plan=replay_flag_deadline(b.observation,alt_moved,alt_focus,rules);
  expect_value(plan.enemy_turn==1 && plan.unchanged_seat_turn==2,"dark HQ changed the forecast");
  const auto quiet_turn=[&](State state) {
    step(state,focus,rules);
    return static_cast<int>(state.to_move);
  };
  expect_value(quiet_turn(original)!=quiet_turn(altered),"HQ did not change the actual next seat");
}

static size_t best_public_guard(const std::vector<FlagDeadline>& deadlines,
                                const std::vector<double>& preference,size_t frozen,size_t human) {
  if(deadlines[frozen].enemy_turn==kSeats && deadlines[human].enemy_turn==kSeats) return frozen;
  std::optional<size_t> best;
  for(size_t i=0;i<deadlines.size();++i) {
    const auto& v=deadlines[i];
    if(v.enemy_turn!=kSeats || v.team_win || v.uncertain_finish) continue;
    if(!best || preference[i]>preference[*best]) best=i;
  }
  return best.value_or(frozen);
}

static void compare_actions(const State& state,const Rules& rules,const ShortWeights& weights,
                            const std::string& key,int repeats,const char* kind,Move a,Move b) {
  if(a==b) return;
  for(int table:{0,1,2}) for(int pair=0;pair<repeats;++pair) {
    const auto seed=mix_seed(58601+state.ply,pair);
    std::cout<<"P "<<key<<' '<<kind<<' '<<table<<' '<<pair<<' '
             <<review_outcome(review_rollout(state,a,rules,table,true,weights,seed),state.to_move)
             <<' '<<review_outcome(review_rollout(state,b,rules,table,true,weights,seed),state.to_move)<<'\n';
  }
}

static void print_decisions(const State& state,const MovedSquares& moved,Move actual,const Rules& rules,
                            const ShortWeights& weights,const std::string& key,int repeats) {
  const auto d=seat_decision(state);
  const auto parent=review_learned(d,weights);
  const auto static_policy=review_reference(d);
  const size_t frozen=std::max_element(parent.begin(),parent.end())-parent.begin();
  const size_t human=std::find(d.actions.begin(),d.actions.end(),actual)-d.actions.begin();
  expect_value(human<d.actions.size(),"recorded action missing from projection");
  std::vector<FlagDeadline> deadlines;
  for(const auto action:d.actions)
    deadlines.push_back(replay_flag_deadline(d.observation,moved,action,rules));
  const auto choice=replay_deadline_choice(deadlines,parent);
  const auto top_heur=std::max_element(static_policy.begin(),static_policy.end())-static_policy.begin();
  const auto guard=best_public_guard(deadlines,parent,frozen,human);
  const std::array<size_t,5> options{human,frozen,choice,static_cast<size_t>(top_heur),guard};
  int guards=0,win_options=0;
  double immediate_mass=0;
  for(size_t i=0;i<deadlines.size();++i) {
    guards+=deadlines[i].enemy_turn==kSeats && !deadlines[i].team_win;
    win_options+=deadlines[i].team_win;
    if(deadlines[i].enemy_turn==1) immediate_mass+=parent[i];
  }
  std::cout<<"D "<<key<<' '<<state.ply+1<<' '<<int(state.to_move)<<' '<<d.actions.size()
           <<' '<<guards<<' '<<win_options<<' '<<immediate_mass<<' '<<(choice!=frozen);
  for(auto index:options) {
    const auto m=d.actions[index];
    const auto& v=deadlines[index];
    const int higher=static_cast<int>(std::count_if(parent.begin(),parent.end(),
                                                   [&](double x){return x>parent[index];}));
    const int attacking_seat=v.threat==kNone ? kNone : d.observation.seat_at(v.threat);
    const int flag_owner=v.flag==kNone ? kNone : d.observation.seat_at(v.flag);
    std::cout<<' '<<coords(m)<<' '<<v.enemy_turn<<' '<<v.unchanged_seat_turn<<' '<<v.team_win<<' '<<v.uncertain_finish
             <<' '<<terminal_or_warning(state,m,rules)<<' '<<higher<<' '<<parent[index]
             <<' '<<attacking_seat<<' '<<flag_owner;
  }
  std::cout<<'\n';
  compare_actions(state,rules,weights,key,repeats,"deadline",d.actions[frozen],d.actions[choice]);
  if(guard!=choice) compare_actions(state,rules,weights,key,repeats,"guard",d.actions[frozen],d.actions[guard]);
  if(repeats==64 && guard!=frozen && human!=frozen)
    compare_actions(state,rules,weights,key,repeats,"human_guard",actual,d.actions[guard]);
}

static bool read_one(const ShortWeights& weights) {
  std::string key;
  int count,ccw,repeats;
  if(!(std::cin>>key>>count>>ccw>>repeats)) return false;
  int8_t layouts[4][25]{};
  if(!read_layouts(layouts)) return false;
  std::vector<Row> rows(static_cast<size_t>(count));
  for(auto& row:rows)
    if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return false;
  Rules rules;
  rules.ccw=ccw!=0;
  rules.marshal_reveals_flag=true;
  rules.no_progress_limit=71;
  rules.check_involved_no_move=true;
  State state{};
  MovedSquares moved{};
  Move actual;
  if(!archive_prefix(state,moved,actual,rules,layouts,rows)) {
    std::cout<<"X "<<key<<'\n';
    return true;
  }
  if(key=="002596") verify_dark_hq(state,moved,actual,rules,layouts,rows);
  print_decisions(state,moved,actual,rules,weights,key,repeats);
  return true;
}

int main() {
  int count;
  ShortWeights weights{};
  if(!(std::cin>>count)||count<1) return 2;
  for(size_t i=0;i<5;++i) if(!(std::cin>>weights[i])||!std::isfinite(weights[i])) return 2;
  std::cout<<std::setprecision(12);
  for(int i=0;i<count;++i) if(!read_one(weights)) return 3;
}
