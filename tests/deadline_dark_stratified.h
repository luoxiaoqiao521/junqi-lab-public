#pragma once

struct StratifiedPivot {
  Move move{};
  unsigned loss=0,other=0;
};

static StratifiedPivot stratified_pivot(const SeatDecision& d,const OpeningMarks& evidence,const Rules& rules,
                                       const StudentNetwork& model) {
  StratifiedPivot pivot;
  double best=-1.;
  for (const auto move:d.actions) {
    const auto& view=d.observation;
    if (view.seat_at(move.to)<0 || view.rank_at(move.to)!=Unknown) continue;
    const auto domain=compatible_rank_mask(view,evidence,move.to);
    unsigned loss=0;
    for (int rank=0;rank<RankCount;++rank) {
      if ((domain&(1u<<rank)) && resolve(view.rank_at(move.from),rank,rules)==Outcome::DefenderWins)
        loss|=1u<<rank;
    }
    if (!loss || !(domain&~loss)) continue;
    const auto score=model.score(student_features(d,evidence,move,&rules),false);
    if (score>best) { best=score; pivot={move,loss,domain&~loss}; }
  }
  return pivot;
}

static std::vector<EarlyPrefix> stratified_pack(const DarkOpeningPath& path,const SeatDecision& d,
    const Rules& rules,const StratifiedPivot& p,uint64_t salt,int& rejects,int& nodes) {
  std::vector<EarlyPrefix> pack;
  for (unsigned domain:{p.loss,p.other}) {
    auto set=assemble_replan_worlds(path,d,rules,salt+domain,2,p.move.to,domain);
    rejects+=set.rejected;
    nodes+=set.nodes;
    if (set.games.size()!=2) return {};
    for (auto& g:set.games) {
      single_case_check((domain&(1u<<g.state.rank_at[p.move.to]))!=0,"stratum identity changed on strict replay");
      pack.push_back(std::move(g));
    }
  }
  return pack;
}

static void stratified_emit(const DistributionRoot& panel,const std::vector<EarlyPrefix>& set,
                            const StratifiedPivot& p,char tag,int max_steps) {
  for (size_t w=0;w<set.size();++w) {
    const auto& game=set[w];
    const int group=(p.loss&(1u<<game.state.rank_at[p.move.to]))?0:1;
    const int cap=std::min(720,game.state.ply+max_steps);
    for (size_t a=0;a<panel.portfolio.moves.size();++a) for (int desk:{0,tag=='R'?5:2}) {
      const auto payoff=finish_replan(game,panel.portfolio.moves[a],panel.viewer,desk,panel.rules,panel.model,panel.peers,
                                     review_streams(mix_seed(panel.salt,77000+(tag=='R'?900:0)+w)),cap);
      std::cout<<tag<<' '<<a<<' '<<w<<' '<<group<<' '<<desk<<' '<<payoff.result<<' '<<payoff.steps<<'\n';
    }
  }
}

static bool sample_overlaps(const std::vector<EarlyPrefix>& a,const std::vector<EarlyPrefix>& b) {
  for (const auto& x:a) for (const auto& y:b) if (state_hash(x.state)==state_hash(y.state)) return true;
  return false;
}

static void audit_strata(uint64_t seed,int viewer,int table,int cut,const Rules& rules,const StudentNetwork& model,
                         OrdinaryDarkPeers peers) {
  const auto root=restore_replan_root(seed,viewer,table,cut,-1,rules,model,peers);
  const auto& game=root.ordinary;
  std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<game.state.ply;
  if (!root.anchor || game.state.result!=Result::Ongoing || game.state.to_move!=viewer || game.state.out[viewer]) {
    std::cout<<" 0 0 0 0 0 0\n"; return;
  }
  const auto d=seat_decision(game.state);
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[viewer]);
  const auto p=stratified_pivot(d,evidence,rules,model);
  if (!p.loss) { std::cout<<" 1 0 0 0 0 0\n"; return; }
  auto rng=root.streams[viewer];
  const auto fallback=ordinary_dark_move(game,d,rules,OrdinaryDarkPolicy::Tactical,rng);
  const auto salt=public_rollout_seed(d.observation,evidence,d.observation.quiet_clock);
  auto portfolio=replan_portfolio(d,evidence,fallback,rules,model,salt);
  std::vector<Move> competing{portfolio.base};
  if (!(portfolio.base==p.move)) competing.push_back(p.move);
  for (auto move:portfolio.moves) {
    if (std::find(competing.begin(),competing.end(),move)==competing.end()) competing.push_back(move);
    if (competing.size()==3) break;
  }
  portfolio.moves=std::move(competing);
  auto ordinary=assemble_replan_worlds(root.path,d,rules,salt,4);
  int rejected=ordinary.rejected,nodes=ordinary.nodes;
  const auto stratified=stratified_pack(root.path,d,rules,p,mix_seed(salt,793),rejected,nodes);
  const auto reference=stratified_pack(root.path,d,rules,p,mix_seed(salt,794),rejected,nodes);
  const bool available=ordinary.games.size()==4 && stratified.size()==4 && reference.size()==4 &&
      !sample_overlaps(ordinary.games,stratified) && !sample_overlaps(reference,ordinary.games) &&
      !sample_overlaps(reference,stratified);
  const bool isolation=available && seed%8==0;
  std::cout<<' '<<(available?3:2)<<' '<<portfolio.moves.size()<<' '<<early_action_key(p.move)<<' '<<rejected<<' '
           <<nodes<<' '<<int(isolation)<<'\n';
  if (!available) return;
  if (isolation) {
    const auto alt=seat_decision(reference.back().state);
    const auto marks=reference.back().reply_history.for_actor(alt.observation,reference.back().marks[viewer]);
    const auto blind=stratified_pivot(alt,marks,rules,model);
    single_case_check(p.move==blind.move && p.loss==blind.loss && p.other==blind.other,"pivot exposed hidden identity");
  }
  const DistributionRoot panel{root,portfolio,ordinary.games,rules,model,peers,salt,viewer,table};
  stratified_emit(panel,ordinary.games,p,'U',180);
  stratified_emit(panel,stratified,p,'T',180);
  stratified_emit(panel,reference,p,'R',720);
  for (size_t a=0;a<portfolio.moves.size();++a) {
    std::cout<<"S "<<a<<' '<<early_action_key(portfolio.moves[a])<<'\n';
    for (int r:{62,63,64,65}) emit_distribution_cell(panel,'G',a,-1,table,r);
  }
}

static void run_dark_stratified(const char* config) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(config),rules,error),"invalid stratification rules");
  StudentNetwork model,history;
  single_case_check(model.load(std::cin,true) && history.load(std::cin),"invalid stratification models");
  const OrdinaryDarkPeers peers{&history,true};
  uint64_t seed;
  int viewer,table,cut;
  while (std::cin>>seed>>viewer>>table>>cut) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==4) && cut>=0,"bad stratification assignment");
    audit_strata(seed,viewer,table,cut,rules,model,peers);
  }
  single_case_check(std::cin.eof(),"incomplete stratification assignments");
}
