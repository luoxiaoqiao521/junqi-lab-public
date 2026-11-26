#pragma once

struct DistributionRoot {
  const ReplanRoot& root;
  const ReplanPortfolio& portfolio;
  const std::vector<EarlyPrefix>& worlds;
  const Rules& rules;
  const StudentNetwork& model;
  OrdinaryDarkPeers peers;
  uint64_t salt;
  int viewer,table;

  ReplanPayoff run(size_t action,int world,int desk,int stream) const {
    const auto& start=world<0?root.ordinary:worlds[world];
    return finish_replan(start,portfolio.moves[action],viewer,desk,rules,model,peers,
                         review_streams(mix_seed(salt,930000+stream)));
  }
};

static int emit_distribution_cell(const DistributionRoot& root,char tag,size_t move,int world,int desk,int stream) {
  const auto out=root.run(move,world,desk,stream);
  std::cout<<tag<<' '<<move<<' '<<world<<' '<<desk<<' '<<stream<<' '<<out.result<<' '<<out.steps<<'\n';
  return std::max(0,out.result);
}

static int emit_distribution_selection(const DistributionRoot& root,bool audit) {
  const auto& moves=root.portfolio.moves;
  std::vector<std::pair<int,size_t>> screen;
  for (size_t a=0;a<moves.size();++a) {
    int sum=0;
    if (audit) std::cout<<"S "<<a<<' '<<early_action_key(moves[a])<<'\n';
    for (const int w:{0,1}) for (const int desk:{0,2}) {
      sum+=audit?emit_distribution_cell(root,'L',a,w,desk,0):std::max(0,root.run(a,w,desk,0).result);
      if (audit) emit_distribution_cell(root,'L',a,w,desk,1);
    }
    if (audit) {
      for (const int w:{2,3}) for (const int desk:{0,2}) emit_distribution_cell(root,'L',a,w,desk,0);
      for (const int w:{0,1}) for (const int desk:{3,4}) emit_distribution_cell(root,'L',a,w,desk,0);
    }
    screen.emplace_back(sum,a);
  }
  const auto base=static_cast<size_t>(std::find(moves.begin(),moves.end(),root.portfolio.base)-moves.begin());
  std::stable_sort(screen.begin(),screen.end(),[&](const auto& x,const auto& y){
    if (x.second==base || y.second==base) return x.second==base && y.second!=base;
    return x.first>y.first;
  });
  int checksum=0;
  const size_t keep=std::max<size_t>(1,moves.size()/2);
  for (size_t i=0;i<keep;++i) {
    const size_t a=screen[i].second;
    checksum=checksum*7+int(a)+1;
    if (!audit) continue;
    const size_t extra=4*moves.size()/keep+(i<4*moves.size()%keep);
    for (size_t n=0;n<extra;++n) {
      const int w=int(n%2),desk=std::array<int,4>{0,2,3,4}[(n/2+w)%4];
      emit_distribution_cell(root,'D',a,w,desk,10+int(n));
    }
  }
  return checksum;
}

static void audit_distribution(uint64_t seed,int viewer,int table,int cut,const Rules& rules,
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
  const auto worlds=assemble_replan_worlds(root.path,d,rules,salt,8);
  const DistributionRoot panel{root,p,worlds.games,rules,model,peers,salt,viewer,table};
  std::cout<<' '<<d.actions.size()<<' '<<p.moves.size()<<' '<<early_action_key(p.base)<<' '<<worlds.games.size()
           <<' '<<worlds.rejected<<' '<<worlds.nodes<<' '<<worlds.reasons[0]<<' '<<int(worlds.games.size()==8 && seed%8==0)<<'\n';
  if (worlds.games.size()!=8) return;
  const int screen=emit_distribution_selection(panel,true);
  for (size_t a=0;a<p.moves.size();++a) {
    for (int w=4;w<8;++w) for (const int desk:{0,2,5}) for (int rep:{60,61})
      emit_distribution_cell(panel,'R',a,w,desk,rep);
    for (int rep:{62,63,64,65}) emit_distribution_cell(panel,'G',a,-1,table,rep);
  }
  if (seed%8==0) {
    auto alternate=root.path;
    alternate.game=*worlds.last_strict;
    const auto view=seat_decision(worlds.games.back().state);
    const auto marks=worlds.games.back().reply_history.for_actor(view.observation,worlds.games.back().marks[viewer]);
    const auto ref=replan_portfolio(view,marks,fallback,rules,model,salt);
    const auto sample=assemble_replan_worlds(alternate,view,rules,salt,8);
    const DistributionRoot shadow{root,ref,sample.games,rules,model,peers,salt,viewer,table};
    single_case_check(ref.moves==p.moves && sample.games.size()==worlds.games.size() &&
                     emit_distribution_selection(shadow,false)==screen,"distribution hidden substitution changed selection");
    for (size_t i=0;i<8;++i)
      single_case_check(state_hash(sample.games[i].state)==state_hash(worlds.games[i].state),"hidden substitution changed worlds");
  }
}

static void run_dark_response_distribution(const char* path) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(path),rules,error),"invalid response distribution rules");
  StudentNetwork model,history;
  single_case_check(model.load(std::cin,true) && history.load(std::cin),"invalid response distribution models");
  const OrdinaryDarkPeers peers{&history,true};
  uint64_t seed;
  int viewer,table,cut;
  while (std::cin>>seed>>viewer>>table>>cut) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==4) && cut>=0,"invalid response root");
    audit_distribution(seed,viewer,table,cut,rules,model,peers);
  }
  single_case_check(std::cin.eof(),"incomplete response distribution assignments");
}
