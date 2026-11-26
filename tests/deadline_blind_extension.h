#pragma once

static void blind_extension_controls(const Rules& rules) {
  auto quiet=selfplay_fixture();
  place_single_case_piece(quiet,1,4,Brigade);
  place_single_case_piece(quiet,26,30,Engineer);
  const auto safe=seat_decision(quiet);
  OpeningMarks unmarked(safe.observation,false);
  single_case_check(response_extension_reasons(safe,unmarked,rules,false)==0 &&
                    response_extension_reasons(safe,unmarked,rules,true)==1,"settled and contact cases were conflated");
  auto threatened=quiet;
  place_single_case_piece(threatened,27,21,Platoon);
  const auto attack=seat_decision(threatened);
  OpeningMarks marks(attack.observation,false);
  marks.ranks[21]=1u<<Platoon;
  single_case_check(response_extension_reasons(attack,marks,rules,false)==4,"missed next eligible seat's flag threat");
  auto hidden=threatened;
  hidden.rank_at[30]=hidden.rank_of[26]=Battalion;
  const auto swapped=seat_decision(hidden);
  single_case_check(same_observation(attack.observation,swapped.observation) && attack.actions==swapped.actions &&
                    response_extension_reasons(swapped,marks,rules,false)==4,"hidden ranks changed the cutoff trigger");
  const PublicReplyHistory history(threatened,false);
  ResponseAudit audit{9};
  ResponseValue baseline,logged;
  const auto first=attack.actions.front();
  response_trial(threatened,first,marks,history,rules,ShortWeights{},0,161,baseline,false,RoutePriority::Joint,true);
  response_trial(threatened,first,marks,history,rules,ShortWeights{},0,161,logged,false,RoutePriority::Joint,true,&audit);
  single_case_check(std::tie(baseline.wins,baseline.draws,baseline.losses,baseline.unknown,baseline.value)==
                    std::tie(logged.wins,logged.draws,logged.losses,logged.unknown,logged.value),
                    "four-dark logging changed the continuation");
  std::cout<<"controls 4\n";
}

static std::vector<State> blind_extension_worlds(const SeatDecision& d,const OpeningMarks& marks,uint64_t seed) {
  std::vector<State> worlds;
  for (unsigned i=0;i<64 && worlds.size()<8;++i) {
    const auto sample=public_sample(d.observation,marks,mix_seed(seed,i));
    if (!sample) continue;
    if (std::none_of(worlds.begin(),worlds.end(),[&](const State& old) {
      return std::equal(std::begin(old.rank_at),std::end(old.rank_at),std::begin(sample->rank_at));
    })) {
      single_case_check(seat_decision(*sample).actions==d.actions,"world changed legal actions");
      worlds.push_back(*sample);
    }
  }
  return worlds;
}

static void print_blind_scores(const std::vector<State>& worlds,Move action,const OpeningMarks& marks,
    const EarlyPrefix& game,const ExchangePolicy& p,uint64_t seed) {
  for (int mode=0;mode<3;++mode) {
    ResponseValue value;
    ResponseAudit meter{mode==0?9:12,0,0,mode==2};
    const int n=mode==0?4:3;
    for (int i=0;i<n;++i)
      response_trial(worlds[i],action,marks,game.reply_history,p.rules,p.history,i%2,mix_seed(seed,i),
                     value,false,RoutePriority::Joint,true,&meter);
    single_case_check(meter.plies<=36,"matched rollout limit exceeded");
    std::cout<<' '<<value.value/n<<' '<<meter.plies<<' '<<meter.extended<<' '<<meter.cutoff_reasons;
  }
  // Four held-out assignments, paired across the two actions, with distinct reply streams.
  for (size_t i=4;i<worlds.size();++i) {
    ResponseAudit meter{p.rules.truncation_cap};
    ResponseValue value;
    response_trial(worlds[i],action,marks,game.reply_history,p.rules,p.history,int(i%2),mix_seed(seed,i+917),
                   value,false,RoutePriority::Joint,true,&meter);
    std::cout<<' '<<(value.wins?2:value.draws?1:value.losses?0:-1)<<' '<<meter.plies;
  }
}

static void blind_extension_at(const EarlyPrefix& game,const SeatDecision& d,Move baseline,const ExchangePolicy& p,
    uint64_t layout,int phase) {
  const auto& actor=game.marks[p.learner];
  const auto marks=game.reply_history.for_actor(d.observation,actor);
  const auto seed=public_rollout_seed(d.observation,actor,d.observation.quiet_clock);
  const auto actions=response_candidates(d,marks,baseline,p.rules,seed);
  std::cout<<layout<<' '<<p.learner<<' '<<p.table<<' '<<phase<<' '<<game.state.ply;
  if (actions.size()<2 || public_flag_finish(d,baseline,p.rules)) {std::cout<<" 0\n";return;}
  const auto worlds=blind_extension_worlds(d,marks,seed);
  std::cout<<' '<<worlds.size();
  if (worlds.size()==8) {
    for (auto action:{actions[0],actions[1]}) {
      std::cout<<' '<<early_action_key(action);
      print_blind_scores(worlds,action,marks,game,p,seed);
    }
  }
  std::cout<<'\n';
}

static void audit_blind_extension(uint64_t seed,const ExchangePolicy& p) {
  EarlyPrefix game(opening(seed,p.rules),seed);
  bool offered[2]{};
  while (game.state.result==Result::Ongoing && game.state.ply<280) {
    const auto d=seat_decision(game.state);
    const auto ordinary=early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner);
    const int phase=game.enemy_exit?1:0;
    if (d.observation.viewer==p.learner && !offered[phase] && (phase || game.state.ply>=80)) {
      blind_extension_at(game,d,ordinary,p,seed,phase);
      offered[phase]=true;
    }
    if (offered[0] && offered[1]) break;
    game.play(ordinary,p.learner,p.rules);
  }
  for (int phase:{0,1}) if (!offered[phase])
    std::cout<<seed<<' '<<p.learner<<' '<<p.table<<' '<<phase<<" 0 -1\n";
}
