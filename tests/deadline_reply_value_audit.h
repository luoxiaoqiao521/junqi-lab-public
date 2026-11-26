#pragma once

static int response_priority(const ResponseBranch& branch,const Rules& rules,int& inspected) {
  const auto& state=branch.sample;
  if (state.result!=Result::Ongoing) return -1;
  std::vector<Move> replies;
  gen_moves(StateView{state},state.to_move,replies);
  inspected+=int(replies.size());
  int priority=int(branch.collision);
  for (Move move:replies) {
    const int target=state.rank_at[move.to];
    if (state.seat_at[move.to]==kNone) continue;
    if (target==Flag) return 4;
    const auto outcome=resolve(state.rank_at[move.from],target,rules);
    if ((outcome==Outcome::AttackerWins && target>=Division) ||
        (outcome==Outcome::BothRemoved && (target>=Division || state.rank_at[move.from]>=Division)))
      priority=std::max(priority,3);
    else if (outcome!=Outcome::DefenderWins) priority=std::max(priority,2);
  }
  return priority;
}

struct ReplyValueScore {
  double value=0, terminal_only=0;
  int moves=0, terminals=0, inspected=0, reassigned=0;
  int forced_replies=0, distinct_replies=0;
};

static Move lesson_alternative(const State& state,const Rules& rules,Move usual);

static void add_audit_leaf(ReplyValueScore& total,const ResponseBranch& branch,const Rules& rules) {
  ResponseValue leaf;
  record_leaf(leaf,branch.sample,branch.viewer,branch.root_ply,false,rules,RoutePriority::Old);
  total.value+=leaf.value;
  if (leaf.unknown==0) {
    total.terminal_only+=leaf.value;
    ++total.terminals;
  }
}

static ReplyValueScore audit_reply_value(const State& root,Move move,const Rules& rules,const ShortWeights& weights,
                                         uint64_t seed,int arm,int diversity=-1) {
  const OpeningMarks marks(observe(root,root.to_move,IdentityVisibility::AllIdentities),false);
  const PublicReplyHistory history(root,false);
  ReplyValueScore total;
  ResponseAudit work{0,0,0,false,true};
  if (arm==0 || arm==3) {
    for (int i=0;i<(arm==0?4:8);++i) {
      ResponseBranch branch(root,marks,history,mix_seed(seed,17930+i));
      branch.play(move,rules,true);
      ++work.plies;
      continue_response(branch,rules,weights,i%2,arm==0?6:16,true,&work);
      add_audit_leaf(total,branch,rules);
    }
  } else {
    std::vector<ResponseBranch> branches;
    std::array<int,4> replies{{-1,-1,-1,-1}};
    int priority[4]{};
    for (int i=0;i<4;++i) {
      auto& branch=branches.emplace_back(root,marks,history,mix_seed(seed,17930+i));
      branch.play(move,rules,true);
      ++work.plies;
      if (diversity>=0 && branch.sample.result==Result::Ongoing) {
        const int actor=branch.sample.to_move;
        auto decision=seat_decision(branch.sample);
        decision.observation=observe(branch.sample,actor,IdentityVisibility::AllIdentities);
        const auto usual=branch.reply(decision,rules,weights,i%2,true);
        const auto actual=(diversity==1 && i>=2 && actor!=branch.viewer)?
                           lesson_alternative(branch.sample,rules,usual):usual;
        total.forced_replies+=!(actual==usual);
        replies[i]=early_action_key(actual);
        branch.play(actual,rules,true);
        ++work.plies;
      }
      continue_response(branch,rules,weights,i%2,3,true,&work);
      priority[i]=arm==1?response_priority(branch,rules,total.inspected):0;
    }
    std::sort(replies.begin(),replies.end());
    total.distinct_replies=int(std::unique(replies.begin(),replies.end())-replies.begin())-(replies.front()==-1?1:0);
    for (int i=0;i<2;++i) {
      const int selected=priority[i+2]>priority[i]?i+2:i;
      total.reassigned+=selected!=i;
      continue_response(branches[selected],rules,weights,i,9,true,&work);
    }
    for (const auto& branch:branches) add_audit_leaf(total,branch,rules);
  }
  total.moves=work.plies;
  single_case_check(total.moves<=(arm==3?128:24),"reply analysis exceeded declared budget");
  return total;
}

static void review_visited_choice(State state,Move action,const Rules& rules,int learner,int table,uint64_t seed,
                                  const std::string& key,int stream) {
  const int cut=state.ply+1;
  auto rng=review_streams(seed);
  step(state,action,rules);
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    step(state,full_study_fallback(state,rules,learner,table,rng[actor]),rules);
  }
  std::cout<<"G "<<key<<' '<<early_action_key(action)<<' '<<table<<' '<<stream<<' '<<cut<<' '
           <<review_outcome(state.result,learner)<<' '<<state.ply<<' '<<state_hash(state)<<'\n';
}

static void emit_visited_reply_value(const State& root,Move path,const Rules& rules,const FrozenReplyPlayers& parent,
                                     const StudentNetwork& child,uint64_t seed,const std::string& key,int reference) {
  const auto d=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto shortlist=budget_candidates(d,path,rules,4);
  if (shortlist.size()<2) {
    std::cout<<"M "<<key<<' '<<root.ply<<" single-action\n";
    return;
  }
  const auto old=quiet_student(root,path,rules,parent.window);
  const auto updated=quiet_student(root,path,rules,child);
  std::cout<<"R "<<key<<' '<<root.ply+1<<' '<<state_hash(root)<<' '<<int(root.to_move)<<' '
           <<early_action_key(old)<<' '<<early_action_key(updated)<<' '<<shortlist.size()<<' '<<reference<<'\n';
  for (int arm=0;arm<3+reference;++arm) {
    for (Move action:shortlist) {
      const auto start=std::chrono::steady_clock::now();
      const auto value=audit_reply_value(root,action,rules,parent.peer,seed,arm);
      const auto micros=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-start).count();
      std::cout<<"S "<<key<<' '<<arm<<' '<<early_action_key(action)<<' '<<std::setprecision(17)
               <<value.value<<' '<<value.terminal_only<<' '<<value.moves<<' '<<value.terminals<<' '
               <<value.inspected<<' '<<value.reassigned<<' '<<micros<<'\n';
    }
  }
  for (Move action:shortlist) for (int table:{0,1,2}) for (int i=0;i<8;++i)
    review_visited_choice(root,action,rules,root.to_move,table,mix_seed(seed,44213+i),key,i);
}

static State read_visited_start(char format,uint64_t seed,const Rules& rules) {
  if (format=='R') return opening(seed,rules);
  State root;
  if (format=='H') {
    int8_t layout[kSeats][kPiecesPerSeat]{};
    std::vector<std::array<int,4>> moves;
    single_case_check(read_qh(layout,moves),"cannot read new full-information archive opening");
    for (const auto& row:layout) single_case_check(valid_layout(row),"invalid full-information archive layout");
    const auto& row=moves.front();
    const int node=point(row[0],row[1]);
    single_case_check(node>=0 && Board::get().node(node).seat>=0,"invalid first archive seat");
    init_state(root,layout,Board::get().node(node).seat,rules);
  } else {
    std::string tape;
    single_case_check(format=='G' && bool(std::cin>>tape),"invalid teacher audit source");
    const ArchiveRows raw(decode_archive_tape(tape));
    init_state(root,raw.layouts,archive_first_seat(raw),rules);
  }
  return root;
}

static Move lesson_alternative(const State& state,const Rules& rules,Move usual) {
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  DecisionScorer rank(d.observation,rules,DecisionKind::CollisionHeuristic);
  double highest=-1e9;
  Move chosen=usual;
  for (Move move:d.actions) if (!(move==usual)) {
    const double score=rank.score(move).total();
    if (score>highest) {
      highest=score;
      chosen=move;
    }
  }
  return chosen;
}

static void review_lesson(State state,Move first,const Rules& rules,int learner,int table,bool pressure,
                           uint64_t seed,int stream,const std::string& key) {
  const int cut=state.ply+1;
  auto rng=review_streams(seed);
  step(state,first,rules);
  int reply=-1, actor=kNone, changed=0;
  if (state.result==Result::Ongoing) {
    actor=state.to_move;
    const auto usual=full_study_fallback(state,rules,learner,table,rng[actor]);
    const auto choice=pressure && actor!=learner?lesson_alternative(state,rules,usual):usual;
    changed=!(choice==usual);
    reply=early_action_key(choice);
    step(state,choice,rules);
  }
  while (state.result==Result::Ongoing) {
    const int seat=state.to_move;
    step(state,full_study_fallback(state,rules,learner,table,rng[seat]),rules);
  }
  std::cout<<"V "<<key<<' '<<early_action_key(first)<<' '<<table<<' '<<int(pressure)<<' '<<stream<<' '<<cut<<' '
           <<review_outcome(state.result,learner)<<' '<<actor<<' '<<reply<<' '<<changed<<' '<<state_hash(state)<<'\n';
}

static void emit_visited_lesson(const State& root,Move path,const Rules& rules,const FrozenReplyPlayers& parent,
                                uint64_t seed,const std::string& key) {
  const auto d=compare_visibility(root,IdentityVisibility::AllIdentities);
  const auto shortlist=budget_candidates(d,path,rules,4);
  std::cout<<"R "<<key<<' '<<root.ply+1<<' '<<state_hash(root)<<' '<<int(root.to_move)<<' '
           <<early_action_key(quiet_student(root,path,rules,parent.window))<<' '<<shortlist.size()<<'\n';
  for (Move action:shortlist) {
    auto x=student_features(d,OpeningMarks(d.observation,false),action,&rules);
    const auto f=reply_window_features(d.observation,action,rules);
    single_case_check(f.has_value(),"lesson lost full-observation features");
    x.insert(x.end(),f->begin(),f->end());
    std::cout<<"F "<<key<<' '<<early_action_key(action)<<std::setprecision(17);
    for (double value:x) std::cout<<' '<<value;
    std::cout<<'\n';
    for (int table:{0,1,2}) for (bool pressure:{false,true}) for (int i=0;i<8;++i)
      review_lesson(root,action,rules,root.to_move,table,pressure,mix_seed(seed,89271+i),i,key);
  }
}

template<class Evaluate>
static void visit_reached_roots(const Rules& rules,Evaluate evaluate) {
  FrozenReplyPlayers parent;
  parent.load();
  StudentNetwork child;
  single_case_check(child.load(std::cin,true,true),"teacher audit child missing");
  char format;
  std::string key;
  uint64_t seed;
  int cut,policy,table,reference;
  while (std::cin>>format>>key>>seed>>cut>>policy>>table>>reference) {
    single_case_check(cut>=1 && policy>=0 && policy<=2 && table>=0 && table<=2 &&
                      (reference==0 || reference==1),"invalid teacher audit manifest");
    auto root=read_visited_start(format,seed,rules);
    auto rng=review_streams(seed);
    const int learner=int(seed%4);
    bool found=false;
    while (root.result==Result::Ongoing) {
      const int actor=root.to_move;
      auto move=full_study_fallback(root,rules,learner,table,rng[actor]);
      if (actor==learner && root.ply+1>=cut) {
        evaluate(root,move,parent,child,seed,key,reference);
        found=true;
        break;
      }
      if (actor==learner && policy)
        move=quiet_student(root,move,rules,policy==1?parent.window:child);
      step(root,move,rules);
    }
    if (!found) std::cout<<"M "<<key<<' '<<root.ply<<" natural-end\n";
  }
  single_case_check(std::cin.eof(),"incomplete teacher audit manifest");
}

static void run_visited_reply_value(const Rules& rules,bool lessons=false) {
  visit_reached_roots(rules,[&](const State& root,Move move,const FrozenReplyPlayers& parent,
                                const StudentNetwork& child,uint64_t seed,const std::string& key,int reference) {
    if (lessons) emit_visited_lesson(root,move,rules,parent,seed,key);
    else emit_visited_reply_value(root,move,rules,parent,child,seed,key,reference);
  });
}
