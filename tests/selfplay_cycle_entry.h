#pragma once
// Included after the frozen selfplay helpers in selfplay_evaluation_signal.cpp.
#include <bitset>

using Access = std::bitset<kNodes>;

struct PublicPly {
  int seat;
  Move action;
  bool operator==(const PublicPly& other) const {
    return seat==other.seat && action==other.action;
  }
};

struct Visit {
  Observation view;
  int ply;
  size_t offset;
  Move departure;
};

struct Recurrence {
  const Visit* previous=nullptr;
  const Visit* start=nullptr;
};

struct QuietVisits {
  std::vector<PublicPly> trace;
  std::vector<Visit> visits;

  Recurrence repeat(const Observation& view) const {
    Recurrence found;
    for(auto it=visits.rbegin();it!=visits.rend();++it) {
      if(it->view.viewer!=view.viewer ||
         std::memcmp(it->view.seats,view.seats,sizeof view.seats) ||
         std::memcmp(it->view.ranks,view.ranks,sizeof view.ranks)) continue;
      if(!found.previous) { found.previous=&*it; continue; }
      const auto a=it->offset,b=found.previous->offset,end=trace.size();
      if(b-a==end-b && std::equal(trace.begin()+a,trace.begin()+b,trace.begin()+b))
        found.start=&*it;
      break;
    }
    return found;
  }

  void advance(const Observation& view,Move action,const StepInfo& event) {
    if(event.outcome!=Outcome::Move || event.eliminated_mask) {
      visits.clear();
      trace.clear();
      return;
    }
    visits.push_back(Visit{view,view.ply+1,trace.size(),action});
    trace.push_back(PublicPly{view.viewer,action});
  }
};

static void check_public_repeat() {
  QuietVisits book;
  Observation a{};
  std::fill(std::begin(a.seats),std::end(a.seats),int8_t(kNone));
  std::fill(std::begin(a.ranks),std::end(a.ranks),int8_t(Unknown));
  a.seats[1]=0;
  a.ranks[1]=Engineer;
  const Move forward{1,2},back{2,1};
  auto b=a;
  b.seats[1]=kNone;
  b.ranks[1]=Unknown;
  b.seats[2]=0;
  b.ranks[2]=Engineer;
  for(int round=0;round<2;++round) {
    a.ply=2*round;
    b.ply=a.ply+1;
    book.advance(a,forward,StepInfo{forward,0,Outcome::Move,0,0});
    book.advance(b,back,StepInfo{back,0,Outcome::Move,0,0});
  }
  a.ply=4;
  const auto found=book.repeat(a);
  expect_value(found.previous && found.previous->ply==3 && found.start && found.start->ply==1,
               "cycle must require two identical excursions despite changed ply");
  a.viewer=1;
  expect_value(!book.repeat(a).previous,"different seat matched another player's view");
  a.viewer=0;
  a.ranks[1]=Marshal;
  expect_value(!book.repeat(a).previous,"different visible rank matched the cycle");
  book.advance(a,forward,StepInfo{forward,0,Outcome::Move,1,0});
  expect_value(book.visits.empty() && book.trace.empty(),"seat retirement did not reset the cycle");
}

static Access route_goals(const Observation& o,const Rules& rules) {
  Access goals;
  const auto& board=Board::get();
  for(int seat=0;seat<kSeats;++seat) for(int col:{1,3}) {
    const int hq=board.seat_node(seat,5,col);
    if(seat==o.viewer && o.rank_at(hq)==Flag)
      for(int point:board.road(hq)) if(board.node(point).kind!=NodeKind::HQ) goals.set(point);
    if(o.team(seat)==o.team(o.viewer) || o.seat_at(hq)!=seat) continue;
    const auto known=sole_enemy_hq(o,seat,rules);
    if(known==kNone || known==hq) goals.set(board.seat_node(seat,4,col));
  }
  return goals;
}

static Access access_after(const Observation& o,const Access& goals) {
  std::vector<Move> moves;
  gen_moves(o,o.viewer,moves);
  Access mask;
  for(auto move:moves) if(goals.test(move.to)) mask.set(move.to);
  return mask;
}

static bool quiet_option(const Observation& o,Move a) {
  return o.seat_at(a.to)==kNone && Board::get().node(a.to).kind!=NodeKind::HQ;
}

static Access project_quiet(Observation o,Move action,const Access& goals) {
  o.seats[action.to]=o.seats[action.from];
  o.ranks[action.to]=o.ranks[action.from];
  o.seats[action.from]=kNone;
  o.ranks[action.from]=Unknown;
  return access_after(o,goals);
}

struct RoutePair {
  Move route, control;
  Access new_access;
};

static std::optional<RoutePair> route_pair(const SeatDecision& d,Move actual,const Rules& rules,
                                          std::optional<Move>& neutral) {
  const auto& o=d.observation;
  if(!quiet_option(o,actual)) return {};
  const auto goals=route_goals(o,rules);
  const auto baseline=project_quiet(o,actual,goals);
  std::vector<Move> controls;
  std::optional<RoutePair> best;
  int best_score=0;
  for(auto a:d.actions) {
    if(a==actual || !quiet_option(o,a)) continue;
    const auto next=project_quiet(o,a,goals),added=next&~baseline;
    if(next==baseline) controls.push_back(a);
    if(!added.any() || (baseline&~next).any()) continue;
    const int score=score_move(o,a).total();
    if(!best || added.count()>best->new_access.count() ||
       (added.count()==best->new_access.count() && score>best_score)) {
      best=RoutePair{a,actual,added};
      best_score=score;
    }
  }
  if(!controls.empty()) {
    const int original_score=score_move(o,actual).total();
    const auto like_original=[&](Move m) {
      return std::pair{m.from==actual.from,-std::abs(score_move(o,m).total()-original_score)};
    };
    neutral=*std::max_element(controls.begin(),controls.end(),
        [&](Move a,Move b){return like_original(a)<like_original(b);});
  }
  if(!best || controls.empty()) return {};
  const auto merit=[&](Move m) {
    return std::pair{m.from==best->route.from,
                     -std::abs(score_move(o,m).total()-best_score)};
  };
  best->control=*std::max_element(controls.begin(),controls.end(),
      [&](Move a,Move b){return merit(a)<merit(b);});
  return best;
}

struct RouteOutcome {
  int score=-1, enemy=0, ally=0, persists=0, used=0,back=0;
};

static RouteOutcome route_rollout(State state,Move first,Move edge,int learner,int table,const Access& goal,
                                  const Rules& rules,const ShortWeights& weights,uint64_t seed) {
  const int owner=state.to_move;
  step(state,first,rules);
  auto rng=review_streams(seed);
  RouteOutcome result;
  bool visited=false;
  while(state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto d=seat_decision(state);
    const auto action=play_turn(d,learner,table,rules,weights,rng[actor]);
    if(actor==owner && !visited) {
      visited=true;
      result.persists=(access_after(d.observation,goal)).any();
      result.used=goal.test(action.to);
      result.back=(action==edge || (action.from==edge.to && action.to==edge.from));
    } else if(!visited) {
      if(team_of(actor)==team_of(owner)) ++result.ally;
      else ++result.enemy;
    }
    step(state,action,rules);
  }
  result.score=review_outcome(state.result,owner);
  return result;
}

static void compare_routes(const State& state,Move actual,const RoutePair& pair,int block,int table,
                           int learner,const Rules& rules,const ShortWeights& weights,uint64_t seed,
                           bool actor_mode) {
  std::cout<<"O "<<block<<' '<<table<<' '<<state.ply+1<<' '<<state.no_progress<<' '
           <<pair.new_access.count()<<' '<<int(pair.route.from==pair.control.from)<<' '
           <<int(actual.from)<<' '<<int(actual.to)<<' '<<int(pair.route.from)<<' '<<int(pair.route.to)
           <<' '<<int(pair.control.from)<<' '<<int(pair.control.to)<<'\n';
  const std::array<Move,3> moves{actual,pair.route,pair.control};
  for(Move a:moves) expect_value(is_legal(state,a),"matched route contains illegal move");
  for(int rep=0;rep<12;++rep) {
    std::cout<<"P "<<block<<' '<<table<<' '<<rep;
    std::array<int,3> back{};
    for(size_t i=0;i<moves.size();++i) {
      const auto r=route_rollout(state,moves[i],actual,learner,table,pair.new_access,rules,weights,
                                 mix_seed(seed,931+rep));
      back[i]=r.back;
      std::cout<<' '<<r.score<<' '<<r.enemy<<' '<<r.ally<<' '<<r.persists<<' '<<r.used;
    }
    std::cout<<'\n';
    if(actor_mode) std::cout<<"E "<<block<<' '<<table<<' '<<rep<<' '<<back[0]<<' '<<back[1]<<' '<<back[2]<<'\n';
  }
}

static void compare_neutral(const State& state,Move actual,Move change,int block,int table,int learner,
                            const Rules& rules,const ShortWeights& weights,uint64_t seed) {
  expect_value(is_legal(state,actual) && is_legal(state,change),"neutral branch contains illegal move");
  for(int rep=0;rep<12;++rep) {
    std::cout<<"U "<<block<<' '<<table<<' '<<rep;
    for(auto action:{actual,change}) {
      const auto r=route_rollout(state,action,actual,learner,table,Access{},rules,weights,
                                 mix_seed(seed,931+rep));
      std::cout<<' '<<r.score<<' '<<r.enemy<<' '<<r.back;
    }
    std::cout<<'\n';
  }
}

struct EdgeRun {
  Move last{};
  int first=0,length=0;

  void mark(Move move,int ply,bool quiet) {
    if(!quiet) { length=0; return; }
    if(length && last.from==move.to && last.to==move.from) ++length;
    else { first=ply; length=1; }
    last=move;
  }
};

struct LocalCycle {
  std::array<EdgeRun,kSeats> rows{};
  int first=0, established=0, quiet_first=0, quiet_at=0, quiet_clock=0;
  int quiet_seat=-1, node_a=-1, node_b=-1;
  int warning=0, warning_seat=-1, check_at=0, prediction=-1;

  bool record(const Observation& view,Move action,int clock) {
    const int actor=view.viewer,ply=view.ply+1;
    auto& edge=rows[actor];
    edge.mark(action,ply,quiet_option(view,action));
    const bool uninterrupted=edge.length>=3 && clock>=ply-edge.first;
    if(warning && !check_at && actor==warning_seat) {
      check_at=ply;
      prediction=uninterrupted && edge.length==4;
    }
    if(!warning && uninterrupted && edge.length==3) {
      warning=ply;
      warning_seat=actor;
    }
    if(!established && edge.length==4) {
      first=edge.first;
      established=ply;
    }
    if(!quiet_at && uninterrupted && edge.length==4) {
      quiet_first=edge.first;
      quiet_at=ply;
      quiet_clock=clock;
      quiet_seat=actor;
      node_a=action.from;
      node_b=action.to;
    }
    return uninterrupted && edge.length==4;
  }

  void show(int block,int table) const {
    std::cout<<"Q "<<block<<' '<<table<<' '<<first<<' '<<established<<' '<<quiet_first<<' '<<quiet_at
             <<' '<<quiet_clock<<' '<<quiet_seat<<' '<<node_a<<' '<<node_b<<' '<<warning<<' '
             <<check_at<<' '<<prediction<<'\n';
  }
};

static void check_local_cycle() {
  EdgeRun run;
  for(int k=0;k<4;++k) run.mark(k%2 ? Move{3,2} : Move{2,3},1+k*4,true);
  expect_value(run.first==1 && run.length==4,"four moves must alternate on one edge");
  run.mark(Move{2,4},17,true);
  expect_value(run.first==17 && run.length==1,"different action did not break the edge cycle");
  run.mark(Move{4,2},21,false);
  expect_value(run.length==0,"contact counted as a quiet reversal");
}

static void survey_cycles(State& state,int block,int learner,int table,const Rules& rules,
                         const ShortWeights& weights,uint64_t seed,bool local,bool actor_mode) {
  QuietVisits history;
  LocalCycle edges;
  auto streams=review_streams(seed);
  int first=0,returned=0,start=0,confirmed=0,period=0,alert=0,quiet_at_alert=0;
  while(state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto d=seat_decision(state);
    const auto action=play_turn(d,learner,table,rules,weights,streams[actor]);
    const auto match=history.repeat(d.observation);
    const int at=state.ply+1;
    if(!returned && match.previous) {
      first=match.previous->ply;
      returned=at;
    }
    if(!confirmed && match.start) {
      start=match.start->ply;
      confirmed=at;
      period=at-match.previous->ply;
    }
    const bool local_signal=local && edges.record(d.observation,action,state.no_progress);
    const bool trigger=local ? local_signal : match.previous && action==match.previous->departure;
    if(!alert && (actor==learner || actor_mode) && trigger) {
      alert=at;
      quiet_at_alert=state.no_progress;
      std::optional<Move> neutral;
      const auto pair=route_pair(d,action,rules,neutral);
      if(actor_mode)
        std::cout<<"A "<<block<<' '<<table<<' '<<actor<<' '<<(pair ? 1 : neutral ? 2 : 0)<<' '
                 <<(neutral ? int(neutral->from) : -1)<<' '<<(neutral ? int(neutral->to) : -1)<<'\n';
      if(pair) compare_routes(state,action,*pair,block,table,learner,rules,weights,seed,actor_mode);
      else if(actor_mode && neutral)
        compare_neutral(state,action,*neutral,block,table,learner,rules,weights,seed);
    }
    const auto event=step(state,action,rules);
    history.advance(d.observation,action,event);
    if(local && event.eliminated_mask) edges.rows={};
  }
  std::cout<<"L "<<block<<' '<<table<<' '<<first<<' '<<returned<<' '<<start<<' '<<confirmed<<' '
           <<period<<' '<<alert<<' '<<quiet_at_alert<<'\n';
  if(local) edges.show(block,table);
}
