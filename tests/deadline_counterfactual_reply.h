#pragma once

struct Counterreply {
  std::optional<Move> move;
  int actor=kNone, legal=0, team_losses=0, flag_losses=0, mover_losses=0, steps=0;
};

static Counterreply choose_counterreply(const State& root,Move first,const Rules& rules,
                                        const ShortWeights& peer,uint64_t seed) {
  auto next=root;
  step(next,first,rules);
  Counterreply result;
  result.actor=next.result==Result::Ongoing?next.to_move:kNone;
  if (result.actor==kNone || team_of(result.actor)==team_of(root.to_move)) return result;
  std::vector<Move> actions;
  legal_moves(next,actions);
  result.legal=int(actions.size());
  const auto view=observe(next,next.to_move,IdentityVisibility::AllIdentities);
  const PublicReplyHistory history(next,false);
  const OpeningMarks marks(view,false);
  const int mover=root.id_at[first.from];
  bool selected_win=false;
  double best=-1e9;
  for (const auto reply:actions) {
    auto post=next;
    const auto event=step(post,reply,rules);
    const bool wins=team_score(post.result,root.to_move)==0.0;
    ++result.steps;
    result.team_losses+=wins;
    result.flag_losses+=event.exit_reason[root.to_move]==ExitReason::FlagRemoved ||
                       event.exit_reason[(root.to_move+2)%4]==ExitReason::FlagRemoved;
    result.mover_losses+=next.alive[mover] && !post.alive[mover];
    if (selected_win) continue;
    if (wins) {
      result.move=reply;
      selected_win=true;
      continue;
    }
    ResponseValue values;
    ResponseAudit work{5,0,0,false,true};
    for (int stream:{0,1})
      response_trial(next,reply,marks,history,rules,peer,stream,mix_seed(seed,stream),
                     values,false,RoutePriority::Old,true,&work);
    result.steps+=work.plies;
    if (values.value>best) {
      best=values.value;
      result.move=reply;
    }
  }
  return result;
}

static void counterreply_controls(const Rules& rules,const ShortWeights& peer) {
  auto danger=selfplay_fixture();
  place_single_case_piece(danger,1,4,Brigade);
  place_single_case_piece(danger,26,30,Engineer);
  place_single_case_piece(danger,27,21,Engineer);
  danger.no_progress=rules.no_progress_limit-1;
  const auto reply=choose_counterreply(danger,{4,30},rules,peer,142);
  const auto quiet=choose_counterreply(danger,{4,3},rules,peer,142);
  single_case_check(reply.actor==1 && reply.move==Move{21,26} && reply.team_losses>0 &&
                    !quiet.move && quiet.actor==kNone && quiet.legal==0,"counterreply controls failed");
  auto win=selfplay_fixture();
  place_single_case_piece(win,1,51,Engineer);
  place_single_case_piece(win,26,21,Platoon);
  const auto ends=choose_counterreply(win,{51,56},rules,peer,142);
  single_case_check(ends.actor==kNone && !ends.move && ends.steps==0,"terminal win permitted a fictitious counterreply");
  std::cout<<"C "<<reply.legal<<' '<<reply.team_losses<<' '<<quiet.legal<<' '<<ends.legal<<'\n';
}

static Move archived_action(int key) {
  single_case_check(key>=0 && key<kNodes*kNodes,"invalid locked action");
  return {static_cast<uint8_t>(key/kNodes),static_cast<uint8_t>(key%kNodes)};
}

struct CounterreplyScan {
  const ShortWeights& peer;
  std::string key;
  uint64_t seed;
  int cut, original, alternative, examined=0;
  bool ranked=false;

  void inspect(const State& root,Move,const Rules& rules) {
    if (root.ply+1!=cut) return;
    Rng rng(42);
    const Move base=route_baseline(root,rules,rng);
    if (original>=0) single_case_check(base==archived_action(original),"locked old action drifted");
    const auto other=ranked?budget_choice(root,rules,peer,base,BudgetArm::Wide,seed):BudgetChoice{archived_action(alternative),{6}};
    const bool trial=ranked?other.order>=4:!(base==other.action);
    std::cout<<"R "<<key<<' '<<cut<<' '<<root.to_move+0<<' '<<early_action_key(base)<<' '
             <<early_action_key(other.action)<<' '<<other.order<<' '<<trial<<'\n';
    ++examined;
    if (!trial) return;
    single_case_check(!(base==other.action) && is_legal(root,other.action),"invalid comparative action");
    const std::array<Move,2> choices{base,other.action};
    for (int arm:{0,1}) {
      const auto action=choices[arm];
      const auto counter=choose_counterreply(root,action,rules,peer,mix_seed(seed,1921));
      const int answer=counter.move?early_action_key(*counter.move):kNone;
      const int rank=root.rank_at[action.from];
      std::cout<<"P "<<key<<' '<<arm<<' '<<early_action_key(action)<<' '<<rank<<' '<<root.rank_at[action.to]+0<<' '
               <<counter.actor<<' '<<counter.legal<<' '<<counter.team_losses<<' '<<counter.flag_losses<<' '
               <<counter.mover_losses<<' '<<counter.steps<<' '<<answer<<'\n';
      for (int table:{0,1}) for (int arm_reply:{0,1}) for (int stream:{0,1,2,3}) {
        const auto forced=arm_reply?counter.move:std::nullopt;
        route_continuation(root,action,rules,root.to_move,table,mix_seed(seed,6203+stream),key,cut,
                           arm*2+arm_reply,stream,forced);
      }
    }
  }
};

static void run_counterreply_audit(const char* path,const Rules& rules,const ShortWeights& peer) {
  counterreply_controls(rules,peer);
  std::string tag,key,tape,error;
  const auto config=read_text(path);
  int ccw,cut,original,alternative;
  uint64_t seed;
  while (std::cin>>tag>>key>>ccw>>cut>>seed>>original>>alternative) {
    single_case_check((ccw==0 || ccw==1) && cut>=24 && cut<=224 &&
                      ((original<0)==(alternative<0)),"invalid counterreply manifest");
    CounterreplyScan scan{peer,key,seed,cut,original,alternative,0,original<0};
    auto local=rules;
    local.ccw=ccw!=0;
    RuleComparison check;
    if (tag=="H") check=scan_jqh_pressure(scan,local,cut);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid counterreply archive");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      check=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cut);
    }
    std::cout<<"A "<<key<<' '<<check.agrees<<' '<<check.divergence<<' '<<check.reason<<' '<<scan.examined<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete counterreply sample");
}
