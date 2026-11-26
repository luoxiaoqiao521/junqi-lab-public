#pragma once

static std::optional<Move> planned_gate(const Observation& view,const OpeningMarks& marks,Move first) {
  if (view.seat_at(first.to)>=0) return std::nullopt;
  auto projection=project_quiet(view,first);
  std::vector<Move> exits;
  gen_piece_moves(projection,view.viewer,first.to,exits);
  const auto targets=flag_targets(view,marks);
  if (targets.empty()) return std::nullopt;
  std::optional<Move> plan;
  int best=flag_distance(first.to,targets);
  for (const auto move:exits) {
    if (move.to==first.from || view.seat_at(move.to)>=0) continue;
    const auto dist=flag_distance(move.to,targets);
    if (dist<best) { best=dist; plan=move; }
  }
  return plan;
}

static bool gate_usable(const State& state,int viewer,Move plan,const Rules& rules) {
  if (state.seat_at[plan.from]!=viewer || state.out[viewer]) return false;
  std::vector<Move> legal;
  gen_piece_moves(StateView{state},viewer,plan.from,legal);
  if (std::find(legal.begin(),legal.end(),plan)==legal.end()) return false;
  return state.seat_at[plan.to]<0 ||
    resolve(state.rank_at[plan.from],state.rank_at[plan.to],rules)==Outcome::AttackerWins;
}

static Move gate_peer_move(const EarlyPrefix& game,const SeatDecision& d,int viewer,int table,
                            const Rules& rules,OrdinaryDarkPeers peers,Rng& rng) {
  const int actor=d.observation.viewer;
  auto move=ordinary_dark_move(game,d,rules,peers.policy(actor,viewer,table,rules),rng);
  if (peers.model_seat(actor,viewer,table,rules)) move=ordinary_dark_student_move(game,d,move,peers.historical,rules);
  return move;
}

constexpr int gate_window=4, gate_blocked=-1, gate_unknown=-2, gate_settled=-3, gate_retired=-4;

static std::optional<int> gate_path_status(const State& state,int viewer,Move plan,const Rules& rules,int origin) {
  if (state.result!=Result::Ongoing) return gate_settled;
  if (state.out[viewer]) return gate_retired;
  const int elapsed=state.ply-origin;
  if (elapsed>gate_window) return gate_unknown;
  if (state.to_move==viewer) return gate_usable(state,viewer,plan,rules)?elapsed:gate_blocked;
  if (elapsed==gate_window) return gate_unknown;
  return std::nullopt;
}

static void gate_boundary_fixture(const Rules& rules) {
  const Move plan{4,30};
  auto state=single_case_state();
  place_single_case_piece(state,1,4,Brigade);
  place_single_case_piece(state,26,30,Engineer);
  state.ply=gate_window;
  state.to_move=0;
  single_case_check(gate_path_status(state,0,plan,rules,0)==gate_window,"last permitted gate step was cut off");
  state.to_move=1;
  single_case_check(gate_path_status(state,0,plan,rules,0)==gate_unknown,"extra opponent turn fits beyond gate budget");
  state.out[0]=true;
  single_case_check(gate_path_status(state,0,plan,rules,0)==gate_retired,"retired seat counted as blocked");
  state.result=Result::Draw;
  single_case_check(gate_path_status(state,0,plan,rules,0)==gate_settled,"settled game counted as blocked");
  state.result=Result::Ongoing;
  state.out[0]=false;
  state.to_move=0;
  state.rank_at[30]=Marshal;
  single_case_check(gate_path_status(state,0,plan,rules,0)==gate_blocked,"strong enemy still allows passage");
}

static int gate_resumes(EarlyPrefix branch,std::array<Rng,kSeats> rng,int viewer,int table,Move plan,
                         const Rules& rules,OrdinaryDarkPeers peers,int origin,int& work) {
  while (true) {
    if (const auto status=gate_path_status(branch.state,viewer,plan,rules,origin)) return *status;
    const auto d=seat_decision(branch.state);
    const auto move=gate_peer_move(branch,d,viewer,table,rules,peers,rng[d.observation.viewer]);
    branch.play(move,viewer,rules);
    ++work;
  }
}

struct GateTiming {
  int natural=-1,arrival=-1,provisional=-1,persistent=-1,restored=0,occupy_only=0,probes=0,steps=0,
      unknown=0,settled=0,retired=0;
};

static void gate_replies(const EarlyPrefix& game,const SeatDecision& d,const std::array<Rng,kSeats>& rng,
    Move natural,int viewer,int table,Move plan,const Rules& rules,OrdinaryDarkPeers peers,int origin,GateTiming& stats) {
  if (!gate_usable(game.state,viewer,plan,rules)) return;
  auto replies=budget_candidates(d,natural,rules,8);
  for (const auto move:d.actions) {
    if (replies.size()>=12) break;
    if ((move.to==plan.to || move.to==plan.from) && std::find(replies.begin(),replies.end(),move)==replies.end())
      replies.push_back(move);
  }
  for (const auto reply:replies) {
    auto branch=game;
    branch.play(reply,viewer,rules);
    ++stats.probes;
    ++stats.steps;
    if (branch.state.result!=Result::Ongoing) { ++stats.settled; continue; }
    if (branch.state.out[viewer]) { ++stats.retired; continue; }
    if (reply.to==plan.to && stats.arrival<0) stats.arrival=branch.state.ply-origin;
    if (gate_usable(branch.state,viewer,plan,rules)) {
      stats.occupy_only+=reply.to==plan.to;
      continue;
    }
    if (stats.provisional<0) stats.provisional=branch.state.ply-origin;
    const int resumed=gate_resumes(branch,rng,viewer,table,plan,rules,peers,origin,stats.steps);
    if (resumed>0) ++stats.restored;
    else if (resumed==gate_unknown) ++stats.unknown;
    else if (resumed==gate_settled) ++stats.settled;
    else if (resumed==gate_retired) ++stats.retired;
    else if (stats.persistent<0) stats.persistent=branch.state.ply-origin;
  }
}

static GateTiming measure_gate(EarlyPrefix game,Move first,Move plan,int viewer,int table,const Rules& rules,
                                OrdinaryDarkPeers peers,uint64_t seed) {
  GateTiming stats;
  const int origin=game.state.ply;
  auto rng=review_streams(seed);
  game.play(first,viewer,rules);
  ++stats.steps;
  while (true) {
    if (const auto status=gate_path_status(game.state,viewer,plan,rules,origin)) {
      stats.natural=*status;
      break;
    }
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    const auto move=gate_peer_move(game,d,viewer,table,rules,peers,rng[actor]);
    if (team_of(actor)!=team_of(viewer))
      gate_replies(game,d,rng,move,viewer,table,plan,rules,peers,origin,stats);
    game.play(move,viewer,rules);
    ++stats.steps;
  }
  return stats;
}

static int emit_gate_world(const EarlyPrefix& game,Move first,Move plan,int a,int w,int viewer,int table,
                           const Rules& rules,OrdinaryDarkPeers peers,uint64_t salt) {
  const auto g=measure_gate(game,first,plan,viewer,table,rules,peers,mix_seed(salt,4410+w));
  std::cout<<"T "<<a<<' '<<w<<' '<<g.natural<<' '<<g.arrival<<' '<<g.provisional<<' '<<g.persistent
           <<' '<<g.restored<<' '<<g.occupy_only<<' '<<g.probes<<' '<<g.steps<<' '<<g.unknown<<' '
           <<g.settled<<' '<<g.retired<<'\n';
  return (g.natural>0?1:0)+(g.natural>0 && g.persistent<0 && !g.unknown && !g.settled && !g.retired?2:0);
}

static void audit_gate_routes(uint64_t seed,int viewer,int table,int cut,const Rules& rules,
                               const StudentNetwork& model,OrdinaryDarkPeers peers) {
  const auto root=restore_replan_root(seed,viewer,table,cut,-1,rules,model,peers);
  const auto& game=root.ordinary;
  std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<game.state.ply;
  if (!root.anchor || game.state.result!=Result::Ongoing || game.state.to_move!=viewer || game.state.out[viewer]) {
    std::cout<<" 0\n"; return;
  }
  const auto d=seat_decision(game.state);
  const auto& view=d.observation;
  const auto marks=game.reply_history.for_actor(view,game.marks[viewer]);
  const auto salt=public_rollout_seed(view,marks,view.quiet_clock);
  auto rng=root.streams[viewer];
  const auto fallback=ordinary_dark_move(game,d,rules,OrdinaryDarkPolicy::Tactical,rng);
  const auto portfolio=replan_portfolio(d,marks,fallback,rules,model,salt);
  const auto base_plan=planned_gate(view,marks,portfolio.base);
  if (!base_plan) { std::cout<<" 1\n"; return; }
  auto alt=portfolio.base;
  std::optional<Move> alternate;
  double best=-1;
  for (const auto move:portfolio.moves) {
    if (move==portfolio.base) continue;
    const auto plan=planned_gate(view,marks,move);
    if (!plan || plan->to==base_plan->to) continue;
    const auto value=model.score(student_features(d,marks,move,&rules),false);
    if (value>best) { best=value; alt=move; alternate=plan; }
  }
  if (!alternate) { std::cout<<" 1\n"; return; }
  const auto worlds=assemble_replan_worlds(root.path,d,rules,salt);
  if (worlds.games.size()!=4) { std::cout<<" 2\n"; return; }
  const bool control=seed%8==0;
  if (control) {
    const auto& witness=worlds.games.back();
    const auto other=seat_decision(witness.state);
    const auto other_marks=witness.reply_history.for_actor(other.observation,witness.marks[viewer]);
    const auto ref=replan_portfolio(other,other_marks,fallback,rules,model,salt);
    single_case_check(ref.moves==portfolio.moves && planned_gate(other.observation,other_marks,portfolio.base)==base_plan &&
         planned_gate(other.observation,other_marks,alt)==alternate,"gate plans used hidden ranks");
  }
  std::cout<<" 3 "<<early_action_key(portfolio.base)<<' '<<early_action_key(alt)<<' '
           <<int(base_plan->to)<<' '<<int(alternate->to)<<' '<<int(control)<<'\n';
  const std::array<Move,2> actions{portfolio.base,alt},plans{*base_plan,*alternate};
  int rank[2]{};
  for (int a:{0,1}) for (int w:{0,1})
    rank[a]+=emit_gate_world(worlds.games[w],actions[a],plans[a],a,w,viewer,table,rules,peers,salt);
  const int selected=rank[1]>rank[0]?1:0;
  std::cout<<"C "<<selected<<'\n';
  for (int a:{0,1}) for (int w:{-1,2,3}) for (int r:{0,1,2,3}) {
    const int desk=w<0?table:5;
    const auto out=finish_replan(w<0?game:worlds.games[w],actions[a],viewer,desk,rules,model,peers,
                                 review_streams(mix_seed(salt,51000+r)));
    std::cout<<"R "<<a<<' '<<w<<' '<<r<<' '<<out.result<<' '<<out.steps<<'\n';
  }
}

static void run_dark_gate_timing(const char* config) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(config),rules,error),"invalid gate timing rules");
  gate_boundary_fixture(rules);
  StudentNetwork model,history;
  single_case_check(model.load(std::cin,true) && history.load(std::cin),"invalid gate timing models");
  const OrdinaryDarkPeers peers{&history,true};
  uint64_t seed;
  int viewer,table,cut;
  while (std::cin>>seed>>viewer>>table>>cut) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==4) && cut>=0,"bad gate timing assignment");
    audit_gate_routes(seed,viewer,table,cut,rules,model,peers);
  }
  single_case_check(std::cin.eof(),"incomplete gate timing assignments");
}
