#pragma once

struct OrdinaryFork {
  EarlyPrefix game;
  std::array<Rng,kSeats> streams;
  std::array<Move,2> actions;
  Move baseline;
};

static std::optional<OrdinaryFork> first_ordinary_fork(uint64_t seed,int learner,int table,const Rules& rules,
                                                       const StudentNetwork& parent,const StudentNetwork& child) {
  auto game=EarlyPrefix(opening(seed,rules),seed);
  auto streams=review_streams(mix_seed(seed,70719));
  while (game.state.result==Result::Ongoing && game.state.ply<720) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    const auto baseline=ordinary_dark_move(game,d,rules,ordinary_dark_peer(actor,learner,table,rules),streams[actor]);
    auto move=baseline;
    if (actor==learner) {
      move=ordinary_dark_student_move(game,d,baseline,&parent,rules);
      const auto alternative=ordinary_dark_student_move(game,d,baseline,&child,rules);
      if (!(alternative==move)) return OrdinaryFork{std::move(game),streams,{move,alternative},baseline};
    }
    game.play(move,learner,rules);
  }
  return std::nullopt;
}

static EarlyPrefix ordinary_fork_game(const OrdinaryFork& fork,Move action,int learner,int table,const Rules& rules,
                                     const StudentNetwork& model,std::array<Rng,kSeats> streams) {
  auto game=fork.game;
  game.play(action,learner,rules);
  while (game.state.result==Result::Ongoing && game.state.ply<720) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    auto move=ordinary_dark_move(game,d,rules,ordinary_dark_peer(actor,learner,table,rules),streams[actor]);
    if (actor==learner) move=ordinary_dark_student_move(game,d,move,&model,rules);
    game.play(move,learner,rules);
  }
  return game;
}

static void emit_ordinary_fork(uint64_t seed,int learner,int table,const Rules& rules,const OrdinaryFork& fork,
                                const StudentNetwork& parent,const StudentNetwork& child,int replicas) {
  const auto d=seat_decision(fork.game.state);
  const auto list=budget_candidates(d,fork.baseline,rules,4);
  const auto evidence=fork.game.reply_history.for_actor(d.observation,fork.game.marks[learner]);
  std::cout<<"F "<<seed<<' '<<learner<<' '<<table<<' '<<d.observation.ply<<' '<<d.actions.size()<<' '<<list.size();
  for (const auto action:fork.actions) {
    const auto x=student_features(d,evidence,action,&rules);
    std::cout<<' '<<early_action_key(action)<<' '<<std::setprecision(17)<<parent.score(x,false)<<' '<<child.score(x,false);
  }
  std::cout<<'\n';
  for (int i=0;i<=replicas;++i) for (int a=0;a<2;++a) for (int m=0;m<2;++m) {
    const auto streams=i==0?fork.streams:review_streams(mix_seed(seed,250500+i));
    const auto game=ordinary_fork_game(fork,fork.actions[a],learner,table,rules,m?child:parent,streams);
    std::cout<<"R "<<seed<<' '<<learner<<' '<<table<<' '<<i<<' '<<a<<' '<<m<<' '
             <<review_outcome(game.state.result,learner)<<' '<<game.state.ply<<' '<<game.trace<<'\n';
  }
}

static void run_dark_first_loss(const char* path) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(path),rules,error),"invalid first-loss rules");
  StudentNetwork parent,child;
  single_case_check(parent.load(std::cin,true) && child.load(std::cin,true),"invalid first-loss weights");
  uint64_t seed;
  int learner,table,cut,a,b,replicas;
  while (std::cin>>seed>>learner>>table>>cut>>a>>b>>replicas) {
    single_case_check(learner>=0 && learner<kSeats && table>=0 && table<4 && replicas>=1 && replicas<=12,
                      "invalid first-loss assignment");
    const auto fork=first_ordinary_fork(seed,learner,table,rules,parent,child);
    if (!fork) {
      single_case_check(cut<0,"reported divergence is absent");
      std::cout<<"M "<<seed<<' '<<learner<<' '<<table<<'\n';
      continue;
    }
    single_case_check(cut<0 || (fork->game.state.ply==cut && early_action_key(fork->actions[0])==a &&
                          early_action_key(fork->actions[1])==b),"reported divergence or decision root changed");
    emit_ordinary_fork(seed,learner,table,rules,*fork,parent,child,replicas);
  }
  single_case_check(std::cin.eof(),"incomplete first-loss input");
}
