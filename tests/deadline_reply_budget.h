#pragma once

#include <chrono>

struct BudgetChoice {
  Move action;
  ResponseAudit work;
  int considered=0, order=0;
  int64_t micros=0;
};

enum class BudgetArm { Fixed, Wide, Extend, Leaf, MatchedShort };

static std::vector<Move> budget_candidates(const SeatDecision& decision,Move fallback,
                                           const Rules& rules,size_t width) {
  DecisionScorer baseline(decision.observation,rules,DecisionKind::FlagRoute);
  std::vector<ScoredMove> ranked;
  ranked.reserve(decision.actions.size());
  for (const auto move:decision.actions) ranked.push_back({move,baseline.score(move).total()});
  std::stable_sort(ranked.begin(),ranked.end(),[](const ScoredMove& a,const ScoredMove& b) {
    return a.score>b.score;
  });
  std::vector<Move> candidates{fallback};
  for (const auto& candidate:ranked) {
    if (candidates.size()==width) break;
    if (!(candidate.move==fallback)) candidates.push_back(candidate.move);
  }
  return candidates;
}

static BudgetChoice budget_choice(const State& root,const Rules& rules,const ShortWeights& peer,
                                  Move fallback,BudgetArm arm,uint64_t seed) {
  const auto start=std::chrono::steady_clock::now();
  const auto decision=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto& view=decision.observation;
  const PublicReplyHistory history(root,false);
  const OpeningMarks marks(view,false);
  const bool wide=arm==BudgetArm::Wide, extend=arm==BudgetArm::Extend;
  const auto candidates=budget_candidates(decision,fallback,rules,wide?8:4);
  ResponseAudit meter{extend?8:6,0,0,extend,true};
  meter.extension_at=6;
  const int samples=wide?2:extend || arm==BudgetArm::MatchedShort?3:4;
  double best=-1e9;
  Move chosen=fallback;
  for (const auto action:candidates) {
    ResponseValue score;
    for (int i=0;i<samples;++i)
      response_trial(root,action,marks,history,rules,peer,i%2,mix_seed(seed,i),score,false,
                     arm==BudgetArm::Leaf?RoutePriority::Joint:RoutePriority::Old,true,&meter);
    if (score.value>best+1e-9) {
      best=score.value;
      chosen=action;
    }
  }
  single_case_check(meter.plies<=96,"equal search budget exceeded");
  const auto dt=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start);
  const int order=int(std::find(candidates.begin(),candidates.end(),chosen)-candidates.begin());
  return {chosen,meter,int(candidates.size()),order,dt.count()};
}

static void print_budget_choice(const std::string& key,int cut,BudgetArm arm,const BudgetChoice& item) {
  const auto& work=item.work;
  std::cout<<"B "<<key<<' '<<cut<<' '<<int(arm)<<' '<<early_action_key(item.action)<<' '<<item.considered<<' '<<item.order<<' '
           <<work.plies<<' '<<item.micros<<' '<<work.gate_visits<<' '<<work.extended<<' '
           <<work.extension_steps<<' '<<work.early_settled<<' '<<work.early_unresolved;
  for (const int count:work.gate_reasons) std::cout<<' '<<count;
  std::cout<<'\n';
}

static void budget_root(const std::string& key,const State& root,const Rules& rules,
                        const ShortWeights& peer,uint64_t seed) {
  const int cut=root.ply+1;
  const int viewer=root.to_move;
  Rng rng(42);
  const auto baseline=route_baseline(root,rules,rng);
  std::vector<Move> selected{baseline};
  std::vector<Move> legals;
  legal_moves(root,legals);
  std::cout<<"R "<<key<<' '<<cut<<' '<<viewer<<' '<<legals.size()<<' '<<early_action_key(baseline)<<'\n';
  for (const auto arm:{BudgetArm::Fixed,BudgetArm::Wide,BudgetArm::Extend,BudgetArm::Leaf,BudgetArm::MatchedShort}) {
    const auto item=budget_choice(root,rules,peer,baseline,arm,seed);
    print_budget_choice(key,cut,arm,item);
    if (std::find(selected.begin(),selected.end(),item.action)==selected.end()) selected.push_back(item.action);
  }
  for (size_t index=0;index<selected.size();++index)
    for (int table:{0,1}) for (int stream:{0,1,2,3})
      route_continuation(root,selected[index],rules,viewer,table,mix_seed(seed,917+stream),key,cut,int(index),stream);
}

static void budget_gate_controls(const Rules& rules,const ShortWeights& peer) {
  for (bool counter:{false,true}) {
    auto root=selfplay_fixture();
    place_single_case_piece(root,1,4,Brigade);
    place_single_case_piece(root,26,30,Engineer);
    place_single_case_piece(root,27,counter?21:61,Engineer);
    const Move bait{4,30}, guard{4,3}, reply{21,26};
    if (counter) root.no_progress=rules.no_progress_limit-1;
    const auto exact=full_route_proof(root,bait,rules,2);
    if (counter) {
      auto post=root;
      step(post,bait,rules);
      single_case_check(post.to_move==1 && is_legal(post,reply) && exact.upper==0 &&
                        full_route_proof(root,guard,rules,2).lower==1,
                        "enemy counterattack or safe control invalid");
    }
    ResponseAudit work{3,0,0,true,true};
    work.extension_at=1;
    ResponseValue value;
    const auto view=observe(root,root.to_move,IdentityVisibility::AllIdentities);
    response_trial(root,counter?bait:guard,OpeningMarks(view,false),PublicReplyHistory(root,false),
                   rules,peer,1,27,value,false,RoutePriority::Old,true,&work);
    single_case_check(work.gate_visits==1 && (work.extended>0)==counter &&
                      (work.extension_steps>0)==counter && (!counter || (work.cutoff_reasons&2u)),
                      "live cutoff path failed positive or quiet control");
    std::cout<<"C "<<counter<<' '<<exact.lower<<' '<<exact.upper<<' '<<work.gate_visits<<' '
             <<work.extended<<' '<<work.extension_steps<<' '<<work.cutoff_reasons<<'\n';
  }
}

struct BudgetReplayScan {
  const ShortWeights& peer;
  std::string key;
  uint64_t seed;
  int cut, evaluated=0;

  void inspect(const State& root,Move,const Rules& rules) {
    if (root.ply+1!=cut) return;
    budget_root(key,root,rules,peer,seed);
    ++evaluated;
  }
};

static void run_reply_budget(const char* rules_path,const Rules& rules,const ShortWeights& peer) {
  budget_gate_controls(rules,peer);
  std::string tag,key,tape,error;
  const auto config=read_text(rules_path);
  int ccw,cap;
  uint64_t seed;
  while (std::cin>>tag>>key>>ccw>>cap>>seed) {
    if (tag=="S") {
      auto root=opening(seed,rules);
      auto rng=review_streams(seed);
      while (root.result==Result::Ongoing && root.ply<cap-1)
        step(root,route_baseline(root,rules,rng[root.to_move]),rules);
      const bool reached=root.result==Result::Ongoing;
      if (reached) budget_root(key,root,rules,peer,seed+123);
      std::cout<<"A "<<key<<' '<<root.ply<<' '<<0<<' '<<(reached?"matched":"terminal")<<' '<<reached<<'\n';
      continue;
    }
    single_case_check((ccw==0 || ccw==1) && cap>=24 && cap<=224,"invalid matched budget source");
    BudgetReplayScan scan{peer,key,seed,cap};
    auto local=rules;
    local.ccw=ccw!=0;
    RuleComparison result;
    if (tag=="H") result=scan_jqh_pressure(scan,local,cap);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid budget archive");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      result=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cap);
    }
    std::cout<<"A "<<key<<' '<<result.agrees<<' '<<result.divergence<<' '<<result.reason<<' '<<scan.evaluated<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete budget payload");
}
