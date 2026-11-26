#pragma once

static bool replay_conversion_world(const State& initial,const ConversionJournal& witness,
                                    const EarlyPrefix& expected,const ExchangePolicy& p,EarlyPrefix& result) {
  ConversionJournal observed;
  result=EarlyPrefix(initial,expected.trace);
  for (const int key:witness.actions) {
    const Move move{static_cast<uint8_t>(key/kNodes),static_cast<uint8_t>(key%kNodes)};
    if (result.state.result!=Result::Ongoing || !is_legal(result.state,move)) return false;
    observed.play(result,move,p);
  }
  return observed.views[p.learner]==witness.views[p.learner] &&
         same_observation(observe(result.state,p.learner),observe(expected.state,p.learner));
}

static bool swap_hidden_deployment(State& world,int a,int b) {
  const int ra=world.rank_at[a],rb=world.rank_at[b];
  if (ra==rb || !deployment_rank_allowed(a,rb) || !deployment_rank_allowed(b,ra)) return false;
  std::swap(world.rank_at[a],world.rank_at[b]);
  std::swap(world.rank_of[world.id_at[a]],world.rank_of[world.id_at[b]]);
  return true;
}

static std::vector<EarlyPrefix> conversion_worlds(uint64_t seed,const ConversionPrefix& prefix,
    const ConversionJournal& witness,const ExchangePolicy& p,int& attempts,
    const std::vector<EarlyPrefix>* excluded=nullptr,uint64_t salt=575829) {
  const auto initial=opening(seed,p.rules);
  std::vector<std::pair<int,int>> swaps;
  for (int a=0;a<kNodes;++a) for (int b=a+1;b<kNodes;++b) {
    if (initial.seat_at[a]>=0 && initial.seat_at[a]!=p.learner && initial.seat_at[a]==initial.seat_at[b] &&
        initial.rank_at[a]!=initial.rank_at[b] && deployment_rank_allowed(a,initial.rank_at[b]) &&
        deployment_rank_allowed(b,initial.rank_at[a])) swaps.emplace_back(a,b);
  }
  Rng rng(mix_seed(prefix.game.trace,salt));
  shuffle_slots(swaps,rng);
  std::vector<EarlyPrefix> worlds{prefix.game};
  for (const auto& [a,b]:swaps) {
    if (attempts==128 || worlds.size()==5) break;
    ++attempts;
    auto initial_swap=initial;
    single_case_check(swap_hidden_deployment(initial_swap,a,b),"invalid preregistered deployment swap");
    EarlyPrefix result(initial_swap,seed);
    if (!replay_conversion_world(initial_swap,witness,prefix.game,p,result)) continue;
    single_case_check(seat_decision(result.state).actions==seat_decision(prefix.game.state).actions,
                      "compatible world changed the learner's legal actions");
    const auto same_ranks=[&](const EarlyPrefix& old) {
      return std::equal(std::begin(old.state.rank_at),std::end(old.state.rank_at),std::begin(result.state.rank_at));
    };
    const bool duplicate=std::any_of(worlds.begin(),worlds.end(),same_ranks) ||
                         (excluded && std::any_of(excluded->begin(),excluded->end(),same_ranks));
    if (!duplicate) worlds.push_back(std::move(result));
  }
  return worlds;
}

static void label_arm(const EarlyPrefix& start,Move first,const ExchangePolicy& p,uint64_t rng,bool original) {
  auto game=start;
  if (!original) game.streams=review_streams(rng);
  game.play(first,p.learner,p.rules);
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    game.play(early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner),p.learner,p.rules);
  }
  std::cout<<' '<<review_outcome(game.state.result,p.learner)<<' '<<game.state.ply<<' '<<int(game.ending);
}

static void label_features(const SeatDecision& d,const OpeningMarks& marks,Move action,const ExchangePolicy& p,
    const StudentNetwork& student) {
  const auto x=student_features(d,marks,action);
  const auto y=student_features(d,marks,action,&p.rules);
  std::cout<<" F "<<early_action_key(action)<<' '<<student.score(x,false);
  for (const auto v:y) std::cout<<' '<<v;
  std::cout<<'\n';
}

static void audit_conversion_label(uint64_t seed,const ExchangePolicy& p,bool early,const StudentNetwork& student,
    bool confirm=false,bool anchored=false) {
  ConversionJournal journal;
  auto prefix=conversion_prefix(seed,p,early,&journal);
  auto& game=prefix.game;
  const auto checkpoint=conversion_checkpoint(game);
  std::cout<<"R "<<seed<<' '<<p.learner<<' '<<p.partner<<' '<<p.table<<' '<<early<<' '<<checkpoint;
  if (game.state.result!=Result::Ongoing) {std::cout<<" 0\n";return;}
  if (anchored && game.state.ply>=140) {std::cout<<" 3\n";return;}
  const auto d=seat_decision(game.state);
  std::optional<FinishPolicy> independent;
  if (confirm) {
    auto student_game=game;
    independent=finish_policy(student_game,d,ExchangePolicy{p.rules,p.parent,p.history,p.learner,p.partner,p.table,&student},true);
  }
  std::vector<std::pair<Move,int>> scores;
  const auto ordinary=early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner,&scores);
  const auto choice=conversion_choice(game,d,ordinary,p);
  const bool different=!(choice.action==ordinary) || (anchored && !(independent->base==ordinary));
  if (!different || (choice.kind!=3 && !(anchored && choice.kind==4))) {std::cout<<" 1\n";return;}
  const auto& marks=game.marks[p.learner];
  const auto evidence=game.reply_history.for_actor(d.observation,marks);
  const auto shortlist=response_candidates(d,evidence,ordinary,p.rules,public_rollout_seed(d.observation,marks,d.observation.quiet_clock));
  const auto value=[&](Move move) {
    const auto pos=std::find_if(scores.begin(),scores.end(),[&](const auto& x){return x.first==move;});
    return pos==scores.end()?-10001:pos->second;
  };
  int attempts=0;
  auto development=conversion_worlds(seed,prefix,journal,p,attempts);
  if (confirm) attempts=0;
  auto worlds=confirm?conversion_worlds(seed,prefix,journal,p,attempts,&development,893409):development;
  if (anchored) {
    development.insert(development.end(),worlds.begin()+1,worlds.end());
    attempts=0;
    worlds=conversion_worlds(seed,prefix,journal,p,attempts,&development,1631113);
  }
  std::cout<<" 2 "<<d.observation.ply+1<<' '<<d.actions.size()<<' '<<shortlist.size()<<' '<<worlds.size()<<' '<<attempts
           <<' '<<value(ordinary)<<' '<<value(choice.action)<<' '<<choice.samples;
  if (independent) {
    single_case_check(independent->pressure==ordinary,"student and teacher share different initial proposal streams");
    std::cout<<' '<<early_action_key(independent->base)<<' '
             <<(std::find(independent->shortlist.begin(),independent->shortlist.end(),choice.action)!=independent->shortlist.end());
  }
  std::cout<<'\n';
  for (auto move:{ordinary,choice.action}) label_features(d,marks,move,p,student);
  if (anchored) label_features(d,marks,independent->base,p,student);
  std::cout<<"O";
  for (auto move:{ordinary,choice.action}) label_arm(worlds.front(),move,p,0,true);
  std::cout<<'\n';
  const std::array<int,2> partners=anchored?std::array<int,2>{1,3}:
                                   confirm?std::array<int,2>{1,2}:std::array<int,2>{2,3};
  for (size_t i=0;i<worlds.size();++i) for (int partner:partners) for (int table:{0,1}) for (int repeat:{0,1,2,3}) {
    const ExchangePolicy custom{p.rules,p.parent,p.history,p.learner,partner,table};
    const auto stream=mix_seed(game.trace,(anchored?1593301:confirm?1193301:820701)+repeat);
    std::cout<<"A "<<i<<' '<<partner<<' '<<table<<' '<<repeat;
    for (auto move:{ordinary,choice.action}) label_arm(worlds[i],move,custom,stream,false);
    if (independent) label_arm(worlds[i],independent->base,custom,stream,false);
    std::cout<<'\n';
  }
}
