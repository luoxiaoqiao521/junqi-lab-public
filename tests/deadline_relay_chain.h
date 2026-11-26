#pragma once

struct RelayChainGame {
  uint64_t handoff=0;
  int ply=0, decision=0, before=-1, after=-1;
};

static RelayChainGame relay_chain(State state,const Rules& rules,const ShortWeights& peer,Move first,
                                   int table,int stream,bool replan,uint64_t seed) {
  const int actor=state.to_move, ally=(actor+2)%4;
  auto rng=review_streams(mix_seed(seed,stream));
  RelayChainGame trace;
  step(state,first,rules);
  while (state.result==Result::Ongoing) {
    const int turn=state.to_move;
    Move move;
    if (table==1 && turn==ally) {
      const auto view=observe(state,turn,IdentityVisibility::AllIdentities);
      single_case_check(heuristic_move(view,rng[turn],move),"frozen partner failed to play");
    } else move=route_baseline(state,rules,rng[turn]);
    if (turn==ally && !trace.ply) {
      trace.handoff=state_hash(state);
      trace.ply=state.ply+1;
      trace.before=early_action_key(move);
      if (replan) {
        const auto result=budget_choice(state,rules,peer,move,BudgetArm::Fixed,mix_seed(seed,181+stream));
        move=result.action;
        trace.decision=result.work.plies;
      }
      trace.after=early_action_key(move);
    }
    step(state,move,rules);
  }
  const auto score=team_score(state.result,actor);
  std::cout<<' '<<(score?int(*score*2):-1)<<' '<<state.ply;
  return trace;
}

struct RelayChainScan {
  const ShortWeights& peer;
  std::string key;
  uint64_t seed;
  int cut, evaluated=0;
  void inspect(const State& root,Move human,const Rules& rules) {
    if (root.ply+1!=cut) return;
    Rng rng(42);
    const auto ordinary=route_baseline(root,rules,rng);
    single_case_check(!(ordinary==human),"locked human approach must differ from the old route");
    ++evaluated;
    for (int table:{0,1}) for (int stream:{0,1,2,3}) for (int arm:{0,1,2,3}) {
      const bool open=arm==1 || arm==2, replan=arm>=2;
      const Move first=open?human:ordinary;
      std::cout<<"L "<<key<<' '<<cut<<' '<<table<<' '<<stream<<' '<<arm<<' '<<early_action_key(first);
      const auto out=relay_chain(root,rules,peer,first,table,stream,replan,seed);
      std::cout<<' '<<out.handoff<<' '<<out.ply<<' '<<out.decision<<' '<<out.before<<' '<<out.after<<'\n';
    }
  }
};

static void run_relay_chain(const char* rules_path,const Rules& rules,const ShortWeights& peer) {
  std::string tag,key,tape,error;
  const auto config=read_text(rules_path);
  int ccw,cap;
  uint64_t seed;
  while (std::cin>>tag>>key>>ccw>>cap>>seed) {
    single_case_check((ccw==0 || ccw==1) && cap>=24 && cap<=224,"invalid relay chain source");
    RelayChainScan scan{peer,key,seed,cap};
    auto local=rules;
    local.ccw=ccw!=0;
    RuleComparison result;
    if (tag=="H") result=scan_jqh_pressure(scan,local,cap);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid relay chain archive");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      result=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cap);
    }
    single_case_check(scan.evaluated==1 && result.agrees==cap,"historical relay cut failed to restore");
    std::cout<<"A "<<key<<' '<<cap<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete chain payload");
}
