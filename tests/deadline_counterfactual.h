#pragma once

struct CounterfactualPosition {
  jq::State state;
  jq::PublicReplyHistory history;
  std::array<jq::OpeningMarks,jq::kSeats> marks;
  std::array<jq::Rng,jq::kSeats> streams;

  jq::EndReason play(jq::Move move,const jq::Rules& rules,bool track) {
    const auto before=state;
    const auto event=jq::step(state,move,rules);
    if (track) {
      history.advance(before,state,event,rules);
      for (int seat=0;seat<jq::kSeats;++seat) if (!state.out[seat])
        advance_contact_ranks(marks[seat],observe(before,seat),observe(state,seat),move,event.outcome,rules);
    }
    return event.end_reason;
  }
};

struct CounterfactualEnd {
  int score=0, ply=0;
  jq::EndReason ending=jq::EndReason::None;
  int first_change=0, prior_move=0, new_move=0;
};

static CounterfactualEnd counterfactual_tail(const CounterfactualPosition& start,jq::Move first,int root,int table,
    const jq::Rules& rules,const jq::ShortWeights& parent,bool replan) {
  using namespace jq;
  auto branch=start;
  CounterfactualEnd result;
  result.ending=branch.play(first,rules,replan);
  while (branch.state.result==Result::Ongoing) {
    const auto d=seat_decision(branch.state);
    const int actor=d.observation.viewer;
    auto move=review_move(d,root,table,true,parent,branch.streams[actor]);
    if (actor==root && replan) {
      const auto evidence=branch.history.for_actor(d.observation,branch.marks[actor]);
      const auto baseline=move;
      CriticalReply discarded;
      move=critical_replan(d,evidence,move,branch.history.last_moves()[actor],rules,discarded);
      if (!result.first_change && !(move==baseline)) {
        result.first_change=branch.state.ply+1;
        result.prior_move=early_action_key(baseline);
        result.new_move=early_action_key(move);
      }
    }
    result.ending=branch.play(move,rules,replan);
  }
  result.score=review_outcome(branch.state.result,root);
  result.ply=branch.state.ply;
  return result;
}

static void evaluator_controls(const jq::Rules& rules,const jq::ShortWeights& parent) {
  using namespace jq;
  auto state=selfplay_fixture();
  place_single_case_piece(state,1,51,Engineer);
  place_single_case_piece(state,26,30,Brigade);
  const Move win{51,56}, quiet{51,52};
  single_case_check(is_legal(state,win) && is_legal(state,quiet),"calibration control has an illegal move");
  state.no_progress=rules.no_progress_limit-1;
  PublicReplyHistory history(state,false);
  const auto shared=history.marks();
  CounterfactualPosition point{state,history,{shared,shared,shared,shared},review_streams(713)};
  auto cap=rules;
  cap.no_progress_limit=0;
  cap.truncation_cap=1;
  for (bool plan:{false,true}) {
    const auto positive=counterfactual_tail(point,win,0,1,rules,parent,plan);
    const auto negative=counterfactual_tail(point,quiet,0,1,rules,parent,plan);
    const auto unresolved=counterfactual_tail(point,quiet,0,1,cap,parent,plan);
    single_case_check(positive.score==2 && positive.ending==EndReason::TeamEliminated && positive.ply==1 &&
                      negative.score==1 && negative.ending==EndReason::NoProgressDraw && negative.ply==1 &&
                      unresolved.score==-1 && unresolved.ending==EndReason::PlyCap && unresolved.ply==1,
                      "evaluation does not distinguish terminal win, rule draw and cap");
  }
  std::cerr<<"PASS evaluator distinguishes referee win, draw, and cutoff in both profiles\n";
}

struct ReplyOptions {
  std::vector<jq::Move> known_hits;
  std::vector<std::pair<int,jq::Move>> quiet;
};

static ReplyOptions collect_reply_options(const jq::SeatDecision& decision,const jq::OpeningMarks& evidence,
                                         const jq::Rules& rules) {
  using namespace jq;
  ReplyOptions result;
  const auto& view=decision.observation;
  for (const auto move:decision.actions) {
    if (view.seat_at(move.to)<0) {
      result.quiet.emplace_back(score_move(view,move).total(),move);
      continue;
    }
    const auto ranks=compatible_rank_mask(view,evidence,move.to);
    const auto outcomes=collision_outcome_mask(view.rank_at(move.from),ranks,rules);
    const bool proved_flag=ranks==(1u<<Flag) && outcomes && !(outcomes&outcome_bit(Outcome::DefenderWins));
    if (proved_flag || outcomes==outcome_bit(Outcome::AttackerWins)) result.known_hits.push_back(move);
  }
  return result;
}

static std::vector<std::pair<char,jq::Move>> nominate_enemy_replies(ReplyOptions options,
    const jq::SeatDecision& decision,const jq::OpeningMarks& evidence,jq::Move fallback,
    jq::Move last,uint64_t key,const jq::Rules& rules) {
  using namespace jq;
  std::vector<std::pair<char,Move>> moves{{'A',fallback}};
  const auto insert=[&](char kind,Move move) {
    if (std::none_of(moves.begin(),moves.end(),[&](auto item){return item.second==move;})) moves.emplace_back(kind,move);
  };
  const auto& view=decision.observation;
  if (!options.known_hits.empty()) {
    const auto order=[&](Move m) {
      return std::make_tuple(compatible_rank_mask(view,evidence,m.to)==(1u<<Flag),score_move(view,m).total(),
                             -early_action_key(m));
    };
    const auto best=std::max_element(options.known_hits.begin(),options.known_hits.end(),[&](Move a,Move b) {
      return order(a)<order(b);
    });
    insert('C',*best);
  }
  CriticalReply discarded;
  insert('P',critical_replan(decision,evidence,fallback,last,rules,discarded));
  std::sort(options.quiet.begin(),options.quiet.end(),[](const auto& a,const auto& b) {
    return std::make_pair(a.first,early_action_key(a.second))<std::make_pair(b.first,early_action_key(b.second));
  });
  options.quiet.resize((options.quiet.size()+1)/2);
  options.quiet.erase(std::remove_if(options.quiet.begin(),options.quiet.end(),[&](const auto& candidate) {
    return std::any_of(moves.begin(),moves.end(),[&](const auto& item){return item.second==candidate.second;});
  }),options.quiet.end());
  if (!options.quiet.empty()) {
    const auto i=Rng(mix_seed(key,9101)).below(static_cast<uint32_t>(options.quiet.size()));
    insert('Q',options.quiet[i].second);
  }
  return moves;
}

static void confirm_enemy_reply(const CounterfactualPosition& start,jq::Move fallback,jq::Move move,int root,int table,
    const jq::Rules& rules,const jq::ShortWeights& parent,uint64_t seed,char kind) {
  using namespace jq;
  auto point=start;
  for (int index=0;index<8;++index) {
    const auto tail_seed=mix_seed(seed,19101+index);
    point.streams=review_streams(tail_seed);
    const auto old=counterfactual_tail(point,fallback,root,table,rules,parent,false);
    const auto reply=counterfactual_tail(point,move,root,table,rules,parent,false);
    std::cout<<"V "<<kind<<' '<<tail_seed<<' '<<old.score<<' '<<old.ply<<' '<<reply.score<<' '<<reply.ply<<'\n';
  }
}

static void calibrate_enemy_reply(const CounterfactualPosition& start,jq::Move fallback,jq::Move move,int root,int table,
    const jq::Rules& rules,const jq::ShortWeights& parent,uint64_t seed,char kind,char source,bool flag) {
  using namespace jq;
  auto point=start;
  for (int index=0;index<8;++index) {
    const auto tail_seed=mix_seed(seed,34901+index);
    point.streams=review_streams(tail_seed);
    std::cout<<"T "<<kind<<' '<<source<<' '<<flag<<' '<<tail_seed;
    for (bool replan:{false,true}) for (auto first:{fallback,move}) {
      const auto end=counterfactual_tail(point,first,root,table,rules,parent,replan);
      std::cout<<' '<<end.score<<' '<<end.ply<<' '<<int(end.ending)<<' '<<end.first_change
               <<' '<<end.prior_move<<' '<<end.new_move;
    }
    std::cout<<'\n';
  }
}

static void test_enemy_reply(const jq::State& start,jq::Move first,const jq::Rules& rules,int table,
    const jq::ShortWeights& parent,uint64_t seed,const jq::OpeningMarks& evidence,
    const jq::PublicReplyHistory& history,uint64_t world,size_t arm,bool calibration) {
  using namespace jq;
  const int root=start.to_move;
  const auto shared=history.marks();
  CounterfactualPosition point{start,history,{shared,shared,shared,shared},review_streams(seed)};
  point.marks[root]=evidence;
  point.play(first,rules,true);
  while (point.state.result==Result::Ongoing && point.state.ply<start.ply+9) {
    const auto d=seat_decision(point.state);
    const auto move=review_move(d,root,table,true,parent,point.streams[point.state.to_move]);
    point.play(move,rules,true);
  }
  std::cout<<"J "<<world<<' '<<table<<' '<<arm;
  if (point.state.result!=Result::Ongoing) {std::cout<<" ended\n"; return;}
  const auto d=seat_decision(point.state);
  const auto& view=d.observation;
  const int actor=view.viewer;
  if (view.team(actor)==view.team(root)) {std::cout<<" friendly\n"; return;}
  const auto visible=point.history.for_actor(view,point.marks[actor]);
  const auto fallback=review_move(d,root,table,true,parent,point.streams[actor]);
  const auto options=collect_reply_options(d,visible,rules);
  const auto key=public_rollout_seed(view,visible,view.quiet_clock);
  if (std::find(options.known_hits.begin(),options.known_hits.end(),fallback)!=options.known_hits.end()) {
    std::cout<<" already-chose\n"; return;
  }
  if (options.known_hits.empty() && key%8!=0) {std::cout<<" unsampled-control\n"; return;}
  std::cout<<" test "<<actor<<' '<<view.quiet_clock<<' '<<d.actions.size()<<' '<<options.known_hits.size()
           <<' '<<key<<' '<<early_action_key(fallback)<<'\n';
  const auto moves=nominate_enemy_replies(options,d,visible,fallback,point.history.last_moves()[actor],key,rules);
  CounterfactualEnd original{}, original_replan{};
  for (const auto& [kind,move]:moves) {
    const auto old=counterfactual_tail(point,move,root,table,rules,parent,false);
    const auto replan=counterfactual_tail(point,move,root,table,rules,parent,true);
    std::cout<<"K "<<kind<<' '<<early_action_key(move)<<' '<<score_move(view,move).total()
             <<' '<<old.score<<' '<<old.ply<<' '<<replan.score<<' '<<replan.ply<<'\n';
    if (kind=='A') {original=old; original_replan=replan; continue;}
    const bool frozen_hit=old.score>=0 && old.score<original.score;
    if (frozen_hit) confirm_enemy_reply(point,fallback,move,root,table,rules,parent,seed,kind);
    const bool moved_hit=replan.score>=0 && replan.score<original_replan.score;
    const bool control=mix_seed(key,8000+early_action_key(move))%16==0;
    if (calibration && (frozen_hit || moved_hit || control)) {
      const bool flag=std::any_of(options.known_hits.begin(),options.known_hits.end(),[&](Move m) {
        return compatible_rank_mask(view,visible,m.to)==(1u<<Flag);
      });
      calibrate_enemy_reply(point,fallback,move,root,table,rules,parent,seed,kind,frozen_hit || moved_hit?'H':'O',flag);
    }
  }
}
