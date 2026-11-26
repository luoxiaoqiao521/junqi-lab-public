#pragma once

static Move ordinary_pair_alternative(const SeatDecision& d,const OpeningMarks& marks,const ReplanPortfolio& candidates,
                                      const StudentNetwork& model,const Rules& rules) {
  Move alternative=candidates.base;
  double best=-1.;
  for (const auto move:candidates.moves) {
    if (move==candidates.base) continue;
    const auto value=model.score(student_features(d,marks,move,&rules),false);
    if (value>best) { best=value; alternative=move; }
  }
  return alternative;
}

static int audit_broad_pair(uint64_t seed,int viewer,int table,int cut,int replicas,const Rules& rules,
                            const StudentNetwork& model,OrdinaryDarkPeers peers,int budget) {
  const bool control=seed%16==0;
  const auto root=restore_replan_root(seed,viewer,table,cut,-1,rules,model,peers,control,std::min(720,budget-2));
  const auto& game=root.ordinary;
  int spent=game.state.ply;
  std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<game.state.ply;
  if (!root.anchor || game.state.result!=Result::Ongoing || game.state.to_move!=viewer || game.state.out[viewer]) {
    std::cout<<" -1 -1 0\n"; return spent;
  }
  const auto d=seat_decision(game.state);
  const auto marks=game.reply_history.for_actor(d.observation,game.marks[viewer]);
  const auto salt=public_rollout_seed(d.observation,marks,d.observation.quiet_clock);
  auto rng=root.streams[viewer];
  const auto fallback=ordinary_dark_move(game,d,rules,OrdinaryDarkPolicy::Tactical,rng);
  const auto list=replan_portfolio(d,marks,fallback,rules,model,salt);
  const auto rival=ordinary_pair_alternative(d,marks,list,model,rules);
  if (rival==list.base) { std::cout<<" -1 -1 0\n"; return spent; }
  auto left=student_features(d,marks,list.base,&rules);
  auto right=student_features(d,marks,rival,&rules);
  int verified=0;
  if (control) {
    const auto worlds=assemble_replan_worlds(root.path,d,rules,salt,2);
    if (!worlds.games.empty()) {
      const auto& other=worlds.games.back();
      const auto visible=seat_decision(other.state);
      const auto other_marks=other.reply_history.for_actor(visible.observation,other.marks[viewer]);
      const auto counterpart=replan_portfolio(visible,other_marks,fallback,rules,model,salt);
      single_case_check(list.moves==counterpart.moves && list.base==counterpart.base &&
        rival==ordinary_pair_alternative(visible,other_marks,counterpart,model,rules) &&
        left==student_features(visible,other_marks,list.base,&rules) &&
        right==student_features(visible,other_marks,rival,&rules),"ordinary pair teacher used a true hidden rank");
      verified=1;
    }
  }
  std::cout<<' '<<early_action_key(list.base)<<' '<<early_action_key(rival)<<' '<<verified<<'\n';
  for (const auto& x:{left,right}) {
    std::cout<<"Q";
    for (double value:x) std::cout<<' '<<std::setprecision(17)<<value;
    std::cout<<'\n';
  }
  for (int i=0;i<replicas && spent+2<=budget;++i) {
    const auto streams=review_streams(mix_seed(salt,99200+i));
    const int cap=std::min(720,game.state.ply+(budget-spent)/2);
    const auto a=finish_replan(game,list.base,viewer,table,rules,model,peers,streams,cap);
    const auto b=finish_replan(game,rival,viewer,table,rules,model,peers,streams,cap);
    spent+=a.steps+b.steps;
    std::cout<<"P "<<i<<' '<<a.result<<' '<<b.result<<' '<<a.steps<<' '<<b.steps<<'\n';
  }
  return spent;
}

static void run_dark_broad_pairs(const char* config) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(config),rules,error),"invalid broad pair rules");
  StudentNetwork model,history;
  single_case_check(model.load(std::cin,true) && history.load(std::cin),"invalid broad pair models");
  const OrdinaryDarkPeers peers{&history,true};
  int budget=0;
  single_case_check(bool(std::cin>>budget) && budget>=3 && budget<=200000,"invalid teacher step budget");
  uint64_t seed;
  int viewer,table,cut,replicas;
  while (std::cin>>seed>>viewer>>table>>cut>>replicas) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==4) && cut>=0 && replicas>=1 && replicas<=8,
                      "invalid ordinary pair assignment");
    if (budget>=3) budget-=audit_broad_pair(seed,viewer,table,cut,replicas,rules,model,peers,budget);
  }
  single_case_check(std::cin.eof(),"incomplete ordinary pair assignments");
  std::cout<<"B "<<budget<<'\n';
}
