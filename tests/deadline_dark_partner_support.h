#pragma once

struct PartnerSupportCounts {
  int possible=0,nominated=0,selected=0,resolved=0;
};

static OrdinaryFork support_fork(const EarlyPrefix& game,const SeatDecision& d,const std::array<Rng,kSeats>& streams,
                                 Move baseline,const std::array<bool,kNodes>& targets,const StudentNetwork& model,
                                 const Rules& rules) {
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[d.observation.viewer]);
  std::array<double,2> best{-1.,-1.};
  std::array<Move,2> actions{};
  for (const auto move:d.actions) {
    const int side=targets[move.to]?1:0;
    const auto value=model.score(student_features(d,evidence,move,&rules),false);
    if (value>best[side]) { best[side]=value; actions[side]=move; }
  }
  single_case_check(best[0]>=0 && best[1]>=0,"support fork lacks a competing action");
  return {game,streams,actions,baseline};
}

static std::optional<OrdinaryFork> replay_partner_support(uint64_t seed,int learner,int table,const Rules& rules,
    const StudentNetwork& model,OrdinaryDarkPeers peers,int arm) {
  auto game=EarlyPrefix(opening(seed,rules),seed);
  auto streams=review_streams(mix_seed(seed,70719));
  std::array<bool,kNodes> targets{};
  PartnerSupportCounts count;
  std::optional<OrdinaryFork> fork;
  const int ally=(learner+2)%kSeats;
  while (game.state.result==Result::Ongoing && game.state.ply<720) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    const auto base=ordinary_dark_move(game,d,rules,peers.policy(actor,learner,table,rules),streams[actor]);
    auto move=base;
    bool support=false;
    if (actor==learner) {
      move=ordinary_dark_student_move(game,d,base,&model,rules);
      for (int n=0;n<kNodes;++n) if (d.observation.seat_at(n)<0) targets[n]=false;
      const auto response=[&](Move a){ return targets[a.to]; };
      if (std::any_of(d.actions.begin(),d.actions.end(),response)) {
        ++count.possible;
        const auto candidates=budget_candidates(d,base,rules,4);
        count.nominated+=response(base) || std::any_of(candidates.begin(),candidates.end(),response);
        support=response(move);
        count.selected+=support;
        if (!arm && !fork && std::any_of(d.actions.begin(),d.actions.end(),[&](Move a){ return !response(a); }))
          fork=support_fork(game,d,streams,base,targets,model,rules);
      }
      targets.fill(false);
    }
    const int defender=d.observation.seat_at(move.to);
    const auto outcome=game.play(move,learner,rules);
    targets[move.from]=targets[move.to]=false;
    if (defender==ally && team_of(actor)!=team_of(learner) && outcome==Outcome::AttackerWins &&
        game.state.seat_at[move.to]==actor) targets[move.to]=true;
    if (support) count.resolved+=outcome==Outcome::AttackerWins || outcome==Outcome::BothRemoved;
  }
  std::cout<<"G "<<seed<<' '<<learner<<' '<<table<<' '<<arm<<' '<<review_outcome(game.state.result,learner)<<' '
           <<game.state.ply<<' '<<game.trace<<' '<<count.possible<<' '<<count.nominated<<' '<<count.selected<<' '
           <<count.resolved<<'\n';
  return fork;
}

static void partner_support_branches(const OrdinaryFork& fork,uint64_t seed,int learner,int table,const Rules& rules,
                                     const StudentNetwork& model,OrdinaryDarkPeers peers,int replicas) {
  const auto decision=seat_decision(fork.game.state);
  const auto& view=decision.observation;
  const auto evidence=fork.game.reply_history.for_actor(view,fork.game.marks[learner]);
  const auto capture=fork.actions[1];
  const auto ranks=compatible_rank_mask(view,evidence,capture.to);
  const auto outcomes=collision_outcome_mask(view.rank_at(capture.from),ranks,rules);
  const bool can_remove=outcomes&(outcome_bit(Outcome::AttackerWins)|outcome_bit(Outcome::BothRemoved));
  const bool can_lose=outcomes&outcome_bit(Outcome::DefenderWins);
  const int evidence_class=!outcomes?-1:!can_remove?0:can_lose?1:2;
  std::cout<<"F "<<seed<<' '<<learner<<' '<<table<<' '<<fork.game.state.ply<<' '
           <<early_action_key(fork.actions[0])<<' '<<early_action_key(fork.actions[1])<<' '<<evidence_class<<'\n';
  for (int i=0;i<replicas;++i) for (int ally_table:{0,2}) for (int a=0;a<2;++a) {
    auto game=fork.game;
    auto streams=review_streams(mix_seed(fork.game.trace,83100+i));
    game.play(fork.actions[a],learner,rules);
    while (game.state.result==Result::Ongoing && game.state.ply<720) {
      const auto d=seat_decision(game.state);
      const int actor=d.observation.viewer;
      auto move=ordinary_dark_move(game,d,rules,peers.policy(actor,learner,ally_table,rules),streams[actor]);
      if (actor==learner) move=ordinary_dark_student_move(game,d,move,&model,rules);
      game.play(move,learner,rules);
    }
    std::cout<<"R "<<seed<<' '<<learner<<' '<<table<<' '<<i<<' '<<ally_table<<' '<<a<<' '
             <<review_outcome(game.state.result,learner)<<' '<<game.state.ply<<' '<<game.trace<<'\n';
  }
}

static void run_dark_partner_support(const char* path) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(path),rules,error),"invalid partner support rules");
  StudentNetwork original,mixed,historical;
  single_case_check(original.load(std::cin,true) && mixed.load(std::cin,true) && historical.load(std::cin),
                    "invalid partner support models");
  const OrdinaryDarkPeers peers{&historical,true};
  uint64_t seed;
  int learner,table,replicas;
  while (std::cin>>seed>>learner>>table>>replicas) {
    single_case_check(learner>=0 && learner<4 && (table==0 || table==2) && replicas>=0 && replicas<=8,
                      "invalid partner support assignment");
    const auto fork=replay_partner_support(seed,learner,table,rules,original,peers,0);
    replay_partner_support(seed,learner,table,rules,mixed,peers,1);
    if (fork && replicas) partner_support_branches(*fork,seed,learner,table,rules,original,peers,replicas);
  }
  single_case_check(std::cin.eof(),"incomplete partner support input");
}
