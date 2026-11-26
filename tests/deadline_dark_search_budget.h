#pragma once

struct DarkBudgetCell {
  double value=0.,lower=0.,upper=0.;
  int samples=0,steps=0;
  void add(const EarlyPrefix& world,Move action,const Rules& rules,int table,uint64_t seed,int depth) {
    double estimate=0.;
    const auto result=dark_rollout(world,action,rules,table,seed,depth,steps,&estimate);
    value+=estimate;
    lower+=result<0?0.:estimate;
    upper+=result<0?1.:estimate;
    ++samples;
  }
};

using DarkBudgetWorlds=std::array<std::vector<EarlyPrefix>,2>;
using DarkBudgetValues=std::vector<std::array<DarkBudgetCell,2>>;
// First index is exact/swap, second index is non-bomb/bomb.
using FormationWorlds=std::array<std::array<CoverageTrial,2>,2>;

static int preferred_dark_side(const DarkBudgetValues& data,double probability,bool& pivotal) {
  std::array<size_t,2> ranks{};
  double spans[2]{};
  for (int side=0;side<2;++side) {
    double low=1.,high=-1.;
    for (size_t action=0;action<data.size();++action) {
      const auto& cell=data[action][side];
      const double value=cell.value/cell.samples;
      low=std::min(low,value);
      if (value>high) { high=value; ranks[side]=action; }
    }
    spans[side]=high-low;
  }
  pivotal=ranks[0]!=ranks[1];
  return probability*spans[1]>(1-probability)*spans[0]?1:0;
}

static DarkBudgetValues allocate_dark_budget(const DarkBudgetWorlds& worlds,const std::vector<Move>& moves,
                                              const Rules& rules,int arm,uint64_t seed,double p,bool& pivotal) {
  DarkBudgetValues data(moves.size());
  const auto trial=[&](int side,int i,int world,int table,int depth){
    for (size_t a=0;a<moves.size();++a)
      data[a][side].add(worlds[side][world],moves[a],rules,table,mix_seed(seed,501+i),depth);
  };
  if (arm!=4) {
    for (int side=0;side<2;++side) {
      const int count=arm==0?2:arm==3?1:4;
      const int depth=96/count;
      for (int i=0;i<count;++i) trial(side,i,arm==2?i%2:i,arm==2?i:3*(i%2),depth);
    }
    return data;
  }
  for (int side=0;side<2;++side) for (int i=0;i<2;++i) trial(side,i,i,3*i,24);
  const int extra=preferred_dark_side(data,p,pivotal);
  for (int side=0;side<2;++side) {
    const int count=pivotal?(side==extra?3:1):2;
    for (int i=0;i<count;++i) trial(side,2+i,(2+i)%int(worlds[side].size()),i%4,24);
  }
  return data;
}

static void write_dark_budget_arm(const std::string& key,const std::vector<Move>& moves,int arm,
                                  bool pivotal,const DarkBudgetValues& data) {
  std::cout<<"K "<<key<<' '<<arm<<' '<<int(pivotal)<<'\n';
  for (size_t a=0;a<moves.size();++a) for (int s=0;s<2;++s) {
    const auto& c=data[a][s];
    std::cout<<"B "<<key<<' '<<arm<<' '<<early_action_key(moves[a])<<' '<<s<<' '<<c.samples<<' '<<c.steps
             <<' '<<std::setprecision(17)<<c.value<<' '<<c.lower<<' '<<c.upper<<'\n';
  }
}

static void write_dark_budget(const std::string& key,const DarkBudgetWorlds& worlds,const Rules& rules,
                              const std::vector<Move>& moves,uint64_t seed,double p,int arms) {
  for (int arm=0;arm<arms;++arm) {
    if (arm==1 && (worlds[0].size()<4 || worlds[1].size()<4)) continue;
    bool pivotal=false;
    const auto data=allocate_dark_budget(worlds,moves,rules,arm,seed,p,pivotal);
    write_dark_budget_arm(key,moves,arm,pivotal,data);
  }
}

static bool add_distinct_dark_world(std::vector<EarlyPrefix>& target,const EarlyPrefix& world) {
  for (const auto& other:target)
    if (std::equal(world.state.rank_of,world.state.rank_of+kPieces,other.state.rank_of)) return false;
  target.push_back(world);
  return true;
}

static void write_formation_source_search(const std::string& key,const CoverageRoot& root,
    const DarkBudgetWorlds& generic,const FormationWorlds& formations,const std::vector<Move>& moves,
    const StudentNetwork& student,uint64_t salt,double p) {
  for (int arm=0;arm<2;++arm) {
    DarkBudgetWorlds sampled;
    for (int side=0;side<2;++side) {
      auto& worlds=sampled[side];
      int used[2]{};
      const auto exact=dark_world_prefixes(root,formations[0][side],student);
      const auto swaps=arm?dark_world_prefixes(root,formations[1][side],student):std::vector<EarlyPrefix>{};
      // Preserve one slot for the swap sample when both sources are present.
      if (arm && !exact.empty() && !swaps.empty()) used[0]+=add_distinct_dark_world(worlds,exact.front());
      for (const auto& world:swaps) if (worlds.size()<2) used[1]+=add_distinct_dark_world(worlds,world);
      for (const auto& world:exact) if (worlds.size()<2) used[0]+=add_distinct_dark_world(worlds,world);
      for (const auto& world:generic[side]) if (worlds.size()<2) add_distinct_dark_world(worlds,world);
      single_case_check(worlds.size()==2,"formation search did not fill a conditional branch");
      std::cout<<"J "<<key<<' '<<arm<<' '<<side<<' '<<used[0]<<' '<<used[1]<<'\n';
    }
    bool pivotal=false;
    const auto result=allocate_dark_budget(sampled,moves,root.rules,0,salt,p,pivotal);
    write_dark_budget_arm(key,moves,5+arm,pivotal,result);
  }
}

static void dark_budget_root(const ArchiveRows& raw,const Rules& rules,const std::string& key,int viewer,
                             int cut,uint64_t seed,const StudentNetwork& student,const FormationWorlds* formations=nullptr) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto d=seat_decision(prefix.game.state);
  const auto history=prefix.game.reply_history.for_actor(d.observation,prefix.game.marks[viewer]);
  const auto focus=directed_contact(d,history,rules,0);
  if (focus.kind!=0) { std::cout<<"M "<<key<<" no-bomb-sensitive-contact\n"; return; }
  const CoverageRoot root{raw,rules,prefix,d,history};
  const auto salt=mix_seed(seed,public_rollout_seed(d.observation,history,d.observation.quiet_clock));
  auto rng=Rng(salt);
  Move fallback;
  single_case_check(DecisionScorer(d.observation,rules,DecisionKind::FlagRoute).choose(rng,fallback),"budget has no fallback");
  auto list=budget_candidates(d,fallback,rules,4);
  const auto original=student_pick(d,history,fallback,list,student,rules,false);
  if (std::find(list.begin(),list.end(),focus.move)==list.end()) list.push_back(focus.move);
  const auto no=bomb_side(root,focus.move.to,false,salt,4),yes=bomb_side(root,focus.move.to,true,salt,4);
  const DarkBudgetWorlds worlds{dark_world_prefixes(root,no,student),dark_world_prefixes(root,yes,student)};
  const unsigned mask=compatible_rank_mask(d.observation,history,focus.move.to);
  int slots=0;
  for (int rank=Flag;rank<RankCount;++rank) if (mask&(1u<<rank)) slots+=kRankCount[rank];
  std::cout<<"D "<<key<<' '<<early_action_key(original)<<' '<<early_action_key(focus.move)<<' '<<slots
           <<' '<<worlds[0].size()<<' '<<worlds[1].size()<<' '<<no.nodes+yes.nodes<<' '<<cut;
  for (auto action:list) std::cout<<' '<<early_action_key(action);
  std::cout<<'\n';
  if (worlds[0].size()<2 || worlds[1].size()<2) return;
  write_dark_budget(key,worlds,rules,list,salt,2./slots,formations?1:5);
  if (formations) write_formation_source_search(key,root,worlds,*formations,list,student,salt,2./slots);
  for (auto action:list) for (int table=0;table<4;++table) for (int i=0;i<4;++i) {
    int plies=0;
    const auto result=dark_rollout(prefix.game,action,rules,table,mix_seed(seed,94609+i),400,plies);
    std::cout<<"Y "<<key<<' '<<early_action_key(action)<<' '<<table<<' '<<i<<' '<<result<<' '<<plies<<'\n';
  }
}

static void run_dark_search_budget(const char* rules_path) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid dark budget student");
  std::string key,hex,error;
  int viewer,cut,direction;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>direction>>seed>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && (direction==0 || direction==1) &&
                      parse_rules(archive_review_rules(read_text(rules_path),bool(direction)),rules,error),
                      "invalid dark budget archive input");
    dark_budget_root(ArchiveRows(decode_archive_tape(hex)),rules,key,viewer,cut,seed,student);
  }
  single_case_check(std::cin.eof(),"incomplete dark budget request");
}
