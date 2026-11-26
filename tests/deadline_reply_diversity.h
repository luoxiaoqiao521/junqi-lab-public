#pragma once

static void emit_reply_diversity(const State& root,Move fallback,const Rules& rules,const ShortWeights& weights,
                                 uint64_t seed,const std::string& key) {
  const auto d=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto shortlist=budget_candidates(d,fallback,rules,4);
  std::cout<<"R "<<key<<' '<<root.ply+1<<' '<<state_hash(root)<<' '<<shortlist.size()<<'\n';
  for (Move action:shortlist) {
    for (int arm:{0,1,2,3}) {
      const auto start=std::chrono::steady_clock::now();
      const auto score=audit_reply_value(root,action,rules,weights,seed,arm>=2?1:2,arm%2);
      const auto micros=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-start).count();
      std::cout<<"S "<<key<<' '<<arm<<' '<<early_action_key(action)<<' '<<std::setprecision(17)
               <<score.value<<' '<<score.moves<<' '<<score.reassigned<<' '<<score.inspected<<' '
               <<score.forced_replies<<' '<<score.distinct_replies<<' '<<micros<<'\n';
    }
    for (int table:{0,1,2,3}) for (int i=0;i<8;++i)
      review_visited_choice(root,action,rules,root.to_move,table,mix_seed(seed,39523+i),key,i);
  }
}

static void run_reply_diversity_panel(const Rules& rules) {
  visit_reached_roots(rules,[&](const State& root,Move move,const FrozenReplyPlayers& parent,
                                const StudentNetwork&,uint64_t seed,const std::string& key,int) {
    emit_reply_diversity(root,move,rules,parent.peer,seed,key);
  });
}
