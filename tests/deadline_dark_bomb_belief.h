#pragma once

static CoverageTrial bomb_side(const CoverageRoot& root,int square,bool bomb,uint64_t seed,int count=2) {
  CoverageTrial trial;
  std::array<unsigned,kNodes> pins{};
  pins[square]=bomb?(1u<<Bomb):all_evidence_ranks&~(1u<<Bomb);
  while (trial.proposals<3*count && trial.nodes<1536u*unsigned(count) &&
         int(trial.worlds.size())<count && !trial.contradicted) {
    Rng rng(mix_seed(seed,trial.proposals++));
    coverage_attempt(root,pins,&rng,std::min(512u,1536u*unsigned(count)-trial.nodes),trial);
  }
  return trial;
}

static void emit_bomb_values(const std::string& key,const CoverageRoot& root,
                             const std::array<std::vector<EarlyPrefix>,2>& worlds,
                             const std::vector<Move>& list,uint64_t seed) {
  for (auto action:list) for (int side=0;side<2;++side) {
    int known=0,points=0,work=0;
    int samples=int(worlds[side].size())*2;
    for (const auto& world:worlds[side]) for (int table:{0,1}) {
      const auto output=dark_rollout(world,action,root.rules,table,mix_seed(seed,311+table),180,work);
      known+=output>=0;
      if (output>=0) points+=output;
    }
    std::cout<<"Q "<<key<<' '<<early_action_key(action)<<' '<<side<<' '<<samples<<' '<<known
             <<' '<<points<<' '<<points+2*(samples-known)<<' '<<work<<'\n';
  }
}

static void dark_bomb_root(const ArchiveRows& raw,const Rules& rules,const std::string& key,int viewer,int cut,
                           uint64_t seed,const StudentNetwork& student) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto d=seat_decision(prefix.game.state);
  const auto& view=d.observation;
  const auto evidence=prefix.game.reply_history.for_actor(view,prefix.game.marks[viewer]);
  const auto focus=directed_contact(d,evidence,rules,0);
  if (focus.kind!=0) {
    std::cout<<"M "<<key<<" no-bomb-sensitive-contact\n";
    return;
  }
  const CoverageRoot root{raw,rules,prefix,d,evidence};
  const auto square=focus.move.to;
  const int enemy=view.seat_at(square);
  const auto domain=compatible_rank_mask(view,evidence,square);
  int slots=0;
  for (int rank=Flag;rank<RankCount;++rank)
    if (domain&(1u<<rank)) slots+=kRankCount[rank];
  const auto stock=rank_stock(view,evidence,enemy,Bomb).on_board;
  const auto& tile=Board::get().node(square);
  const auto salt=mix_seed(seed,public_rollout_seed(view,evidence,view.quiet_clock));
  auto rng=Rng(salt);
  Move fallback;
  single_case_check(DecisionScorer(view,rules,DecisionKind::FlagRoute).choose(rng,fallback),"bomb root has no fallback");
  auto list=budget_candidates(d,fallback,rules,4);
  const auto history=student_pick(d,evidence,fallback,list,student,rules,false);
  if (std::find(list.begin(),list.end(),focus.move)==list.end()) list.push_back(focus.move);
  const auto no=bomb_side(root,square,false,salt),yes=bomb_side(root,square,true,salt);
  const std::array<std::vector<EarlyPrefix>,2> worlds{dark_world_prefixes(root,no,student),
                                                   dark_world_prefixes(root,yes,student)};
  std::cout<<"P "<<key<<' '<<early_action_key(history)<<' '<<early_action_key(focus.move)<<' '<<slots<<' '
           <<stock.minimum<<' '<<stock.maximum<<' '<<int(evidence.moved[square])<<' '<<int(tile.r)<<' '<<cut<<' '
           <<worlds[0].size()<<' '<<worlds[1].size()<<' '<<no.contradicted<<' '<<yes.contradicted<<' '
           <<no.nodes+yes.nodes<<' '<<list.size()<<'\n';
  emit_bomb_values(key,root,worlds,list,salt);
  // A separate referee-only label and outcomes. The belief and candidate inputs above use legal evidence.
  std::cout<<"T "<<key<<' '<<(prefix.game.state.rank_at[square]==Bomb)<<'\n';
  if (worlds[0].empty() || worlds[1].empty()) return;
  for (auto action:list) for (int table=0;table<4;++table) for (int i=0;i<4;++i) {
    int plies=0;
    const auto result=dark_rollout(prefix.game,action,rules,table,mix_seed(seed,90209+i),400,plies);
    std::cout<<"Y "<<key<<' '<<early_action_key(action)<<' '<<table<<' '<<i<<' '<<result<<' '<<plies<<'\n';
  }
}

static void run_dark_bomb_belief(const char* rules_path) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid bomb belief comparison student");
  std::string key,hex,error;
  int viewer,cut,direction;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>direction>>seed>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && (direction==0 || direction==1) &&
                      parse_rules(archive_review_rules(read_text(rules_path),bool(direction)),rules,error),
                      "invalid bomb archive input");
    dark_bomb_root(ArchiveRows(decode_archive_tape(hex)),rules,key,viewer,cut,seed,student);
  }
  single_case_check(std::cin.eof(),"incomplete bomb archive request");
}
