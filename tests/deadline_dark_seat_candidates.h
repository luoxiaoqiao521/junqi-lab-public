#pragma once

struct ArrivalWindow {
  int defence=0,advance=0,steps=0;
};

static std::pair<bool,int> earliest_flag_route(const State& possible,const SeatDecision& d) {
  bool can_take=false;
  int room=0;
  const int actor=d.observation.viewer;
  std::vector<int> flags;
  for (int n=0;n<kNodes;++n) {
    const int owner=possible.seat_at[n];
    if (possible.rank_at[n]==Flag && owner>=0 && team_of(owner)!=team_of(actor)) flags.push_back(n);
  }
  for (const auto move:d.actions) {
    can_take|=possible.rank_at[move.to]==Flag;
    for (const int n:flags) {
      const int left=distance(move.to,n);
      if (left<distance(move.from,n)) room=std::max(room,std::max(0,12-left));
    }
  }
  return {can_take,room};
}

static ArrivalWindow arrival_window(EarlyPrefix game,Move first,int viewer,int table,const Rules& rules,
                                    OrdinaryDarkPeers peers,uint64_t seed) {
  auto streams=review_streams(seed);
  ArrivalWindow result;
  game.play(first,viewer,rules);
  auto settle=[&] {
    const int value=review_outcome(game.state.result,viewer);
    if (value==0) { result.defence=-100; result.advance=-100; }
    else if (value==2) { result.defence=100; result.advance=100; }
  };
  settle();
  for (int elapsed=1;elapsed<=4 && game.state.result==Result::Ongoing;++elapsed) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    const auto [takes,room]=earliest_flag_route(game.state,d);
    if (team_of(actor)==team_of(viewer)) {
      result.advance=std::max(result.advance,takes?36-elapsed:room);
    } else if (takes) {
      result.defence=std::min(result.defence,elapsed-10);
    }
    if (actor==viewer || elapsed==4) break;
    auto move=ordinary_dark_move(game,d,rules,peers.policy(actor,viewer,table,rules),streams[actor]);
    if (peers.model_seat(actor,viewer,table,rules))
      move=ordinary_dark_student_move(game,d,move,peers.historical,rules);
    game.play(move,viewer,rules);
    settle();
    ++result.steps;
  }
  return result;
}

static std::vector<Move> seat_nominations(const SeatDecision& d,const ReplanPortfolio& old,
    const std::vector<EarlyPrefix>& worlds,const Rules& rules,OrdinaryDarkPeers peers,uint64_t salt,int& work) {
  if (old.moves.size()==static_cast<size_t>(old.shortlist)) return old.moves;
  std::vector<Move> list(old.moves.begin(),old.moves.begin()+old.shortlist);
  std::vector<ArrivalWindow> scores(d.actions.size());
  const int viewer=d.observation.viewer;
  for (size_t a=0;a<d.actions.size();++a) for (int w:{0,1}) for (int t:{0,2}) {
    const auto window=arrival_window(worlds[w],d.actions[a],viewer,t,rules,peers,mix_seed(salt,41300+w));
    scores[a].defence+=window.defence;
    scores[a].advance+=window.advance;
    work+=window.steps+1;
  }
  const auto base=scores[std::find(d.actions.begin(),d.actions.end(),old.base)-d.actions.begin()];
  for (int arm:{0,1}) {
    if (list.size()==old.moves.size()) break;
    size_t index=d.actions.size();
    int best=arm?base.advance:base.defence;
    for (size_t i=0;i<d.actions.size();++i) {
      const int value=arm?scores[i].advance:scores[i].defence;
      if (value<=best || std::find(list.begin(),list.end(),d.actions[i])!=list.end()) continue;
      best=value;
      index=i;
    }
    if (index<d.actions.size()) list.push_back(d.actions[index]);
  }
  for (const auto move:old.moves) {
    if (list.size()==old.moves.size()) break;
    if (std::find(list.begin(),list.end(),move)==list.end()) list.push_back(move);
  }
  return list;
}

static void seat_candidate_values(const DistributionRoot& root,const std::vector<Move>& new_moves) {
  const auto& old=root.portfolio;
  auto all=old.moves;
  for (const auto move:new_moves) if (std::find(all.begin(),all.end(),move)==all.end()) all.push_back(move);
  const ReplanPortfolio ext{old.base,old.wide,all,old.shortlist};
  const DistributionRoot panel{root.root,ext,root.worlds,root.rules,root.model,root.peers,root.salt,root.viewer,root.table};
  const auto d=seat_decision(root.root.ordinary.state);
  const auto& game=root.root.ordinary;
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[root.viewer]);
  for (size_t a=0;a<all.size();++a) {
    const auto move=all[a];
    const bool in_old=a<old.moves.size();
    const bool in_new=std::find(new_moves.begin(),new_moves.end(),move)!=new_moves.end();
    std::cout<<"S "<<a<<' '<<early_action_key(move)<<' '<<int(in_old)+2*int(in_new)<<'\n';
    std::cout<<"Q "<<a;
    for (double x:student_features(d,evidence,move,&root.rules)) std::cout<<' '<<std::setprecision(17)<<x;
    std::cout<<'\n';
    for (int w:{0,1}) for (int desk:{0,2}) for (int i:{0,1}) emit_distribution_cell(panel,'L',a,w,desk,i);
    for (int w:{2,3}) for (int desk:{0,5}) for (int i:{60,61,62}) emit_distribution_cell(panel,'R',a,w,desk,i);
    for (int i:{67,68,69,70}) emit_distribution_cell(panel,'G',a,-1,root.table,i);
  }
}

static void audit_seat_candidates(uint64_t seed,int viewer,int table,int cut,const Rules& rules,
                                  const StudentNetwork& model,OrdinaryDarkPeers peers) {
  const auto root=restore_replan_root(seed,viewer,table,cut,-1,rules,model,peers);
  const auto& game=root.ordinary;
  std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<root.anchor_ply<<' '<<game.state.ply;
  if (!root.anchor || game.state.result!=Result::Ongoing || game.state.to_move!=viewer || game.state.out[viewer]) {
    std::cout<<" 0 0 -1 0 0 0 0 0\n";
    return;
  }
  const auto d=seat_decision(game.state);
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[viewer]);
  const auto salt=public_rollout_seed(d.observation,evidence,d.observation.quiet_clock);
  auto rng=root.streams[viewer];
  const auto fallback=ordinary_dark_move(game,d,rules,OrdinaryDarkPolicy::Tactical,rng);
  const auto p=replan_portfolio(d,evidence,fallback,rules,model,salt);
  const auto worlds=assemble_replan_worlds(root.path,d,rules,salt);
  const bool control=worlds.games.size()==4 && seed%8==0;
  std::cout<<' '<<d.actions.size()<<' '<<p.moves.size()<<' '<<early_action_key(p.base)<<' '<<worlds.games.size()
           <<' '<<worlds.rejected<<' '<<worlds.nodes<<' '<<worlds.reasons[0]<<' '<<int(control)<<'\n';
  if (worlds.games.size()!=4) return;
  int work=0;
  const auto selected=seat_nominations(d,p,worlds.games,rules,peers,salt,work);
  if (control) {
    const auto alt=seat_decision(worlds.games.back().state);
    const auto marks=worlds.games.back().reply_history.for_actor(alt.observation,worlds.games.back().marks[viewer]);
    const auto shadow=replan_portfolio(alt,marks,fallback,rules,model,salt);
    int steps=0;
    single_case_check(shadow.moves==p.moves && seat_nominations(alt,shadow,worlds.games,rules,peers,salt,steps)==selected &&
                     steps==work,"seat nomination leaked the original hidden board");
  }
  std::cout<<"N "<<work<<'\n';
  seat_candidate_values({root,p,worlds.games,rules,model,peers,salt,viewer,table},selected);
}

static void run_dark_seat_candidates(const char* path) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(path),rules,error),"invalid seat candidate rules");
  StudentNetwork model,history;
  single_case_check(model.load(std::cin,true) && history.load(std::cin),"invalid seat candidate models");
  const OrdinaryDarkPeers peers{&history,true};
  uint64_t seed;
  int viewer,table,cut;
  while (std::cin>>seed>>viewer>>table>>cut) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==4) && cut>=0,"invalid seat candidate root");
    audit_seat_candidates(seed,viewer,table,cut,rules,model,peers);
  }
  single_case_check(std::cin.eof(),"incomplete seat candidate assignments");
}
