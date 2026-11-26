#pragma once
#include <numeric>

struct ReplanRoot {
  DarkOpeningPath path;
  EarlyPrefix ordinary;
  std::array<Rng,kSeats> streams;
  std::optional<Move> anchor;
  int anchor_ply=-1;
  ReplanRoot(State start,uint64_t seed,int viewer):
    path(start,viewer),ordinary(start,seed),streams(review_streams(mix_seed(seed,70719))) {}
};

static ReplanRoot restore_replan_root(uint64_t seed,int viewer,int table,int cut,int forced,const Rules& rules,
                                      const StudentNetwork& model,OrdinaryDarkPeers peers,bool keep_path=true,int cap=720) {
  ReplanRoot root(opening(seed,rules),seed,viewer);
  auto& game=root.ordinary;
  while (game.state.result==Result::Ongoing && game.state.ply<cap) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    if (actor==viewer && root.anchor) return root;
    auto move=ordinary_dark_move(game,d,rules,peers.policy(actor,viewer,table,rules),root.streams[actor]);
    if (peers.model_seat(actor,viewer,table,rules)) move=ordinary_dark_student_move(game,d,move,peers.historical,rules);
    if (actor==viewer) {
      move=ordinary_dark_student_move(game,d,move,&model,rules);
      if (d.observation.ply>=cut) {
        if (forced>=0) {
          const auto it=std::find_if(d.actions.begin(),d.actions.end(),[&](Move a){return early_action_key(a)==forced;});
          single_case_check(d.observation.ply==cut && it!=d.actions.end(),"historical first move did not restore");
          move=*it;
        }
        root.anchor=move;
        root.anchor_ply=d.observation.ply;
      }
    }
    game.play(move,viewer,rules);
    if (keep_path) root.path.play(move,viewer,rules);
    if (game.state.out[viewer]) break;
  }
  return root;
}

static EarlyPrefix ordinary_replan_world(const EarlyPrefix& strict,const DarkOpeningPath& path,int viewer,
                                         const Rules& rules) {
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (int s=0;s<kSeats;++s)
    std::copy_n(strict.state.rank_of+s*kPiecesPerSeat,kPiecesPerSeat,layouts[s]);
  State start;
  init_state(start,layouts,path.first,rules);
  EarlyPrefix game(start,0);
  for (const auto move:path.moves) game.play(move,viewer,rules);
  return game;
}

struct ReplanWorlds {
  std::vector<EarlyPrefix> games;
  std::optional<EarlyPrefix> last_strict;
  int solved=0,rejected=0,duplicates=0,nodes=0;
  std::array<int,6> reasons{};
};

static ReplanWorlds assemble_replan_worlds(const DarkOpeningPath& path,const SeatDecision& d,const Rules& rules,
                                          uint64_t salt,size_t target=4,int square=-1,unsigned ranks=0) {
  ReplanWorlds worlds;
  const auto& game=path.game;
  const auto& view=d.observation;
  std::array<unsigned,kNodes> pins{};
  if (square>=0) pins[square]=ranks;
  std::vector<std::array<int8_t,kNodes>> openings;
  for (size_t i=0;i<6*target && worlds.games.size()<target;++i) {
    std::array<int8_t,kNodes> initial{};
    Rng rng(mix_seed(salt,6170+i));
    CaptureSearch work{384,&rng,&initial};
    const auto status=game.reply_history.solve_candidate(view,game.marks[view.viewer],path.observations.front(),pins,work);
    worlds.nodes+=384-work.remaining;
    if (status!=CaptureCompatibility::Consistent) continue;
    ++worlds.solved;
    if (std::find(openings.begin(),openings.end(),initial)!=openings.end()) { ++worlds.duplicates; continue; }
    PrefixReplayWork report;
    auto world=replay_dark_opening(path,view.viewer,rules,initial,&report);
    if (!world) {
      ++worlds.rejected;
      const std::array<std::string_view,5> known{"seat-control","collision","flag-event","route","actor-history"};
      ++worlds.reasons[std::find(known.begin(),known.end(),report.reason)-known.begin()];
      continue;
    }
    if (std::any_of(worlds.games.begin(),worlds.games.end(),[&](const EarlyPrefix& prior){
          return std::equal(std::begin(prior.state.rank_at),std::end(prior.state.rank_at),std::begin(world->state.rank_at));
        })) { ++worlds.duplicates; continue; }
    openings.push_back(initial);
    worlds.games.push_back(ordinary_replan_world(*world,path,view.viewer,rules));
    worlds.last_strict=std::move(world);
  }
  return worlds;
}

struct ReplanPortfolio {
  Move base,wide;
  std::vector<Move> moves;
  int shortlist;
};

static ReplanPortfolio replan_portfolio(const SeatDecision& d,const OpeningMarks& evidence,Move fallback,
                                        const Rules& rules,const StudentNetwork& model,uint64_t salt) {
  auto moves=budget_candidates(d,fallback,rules,4);
  const auto base=student_pick(d,evidence,fallback,moves,model,rules,false);
  const int shortlist=int(moves.size());
  for (const auto move:response_candidates(d,evidence,fallback,rules,salt)) {
    if (std::find(moves.begin(),moves.end(),move)==moves.end()) moves.push_back(move);
    if (int(moves.size())==shortlist+2) break;
  }
  const auto wide=student_pick(d,evidence,fallback,moves,model,rules,false);
  return {base,wide,std::move(moves),shortlist};
}

struct ReplanPayoff {
  int result,steps;
  uint64_t trace;
};

static ReplanPayoff finish_replan(EarlyPrefix game,Move first,int viewer,int table,const Rules& rules,
    const StudentNetwork& model,OrdinaryDarkPeers peers,std::array<Rng,kSeats> streams,int cap=720) {
  const int ply=game.state.ply;
  game.play(first,viewer,rules);
  while (game.state.result==Result::Ongoing && game.state.ply<cap) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    auto move=ordinary_dark_move(game,d,rules,peers.policy(actor,viewer,table,rules),streams[actor]);
    if (actor==viewer || peers.model_seat(actor,viewer,table,rules))
      move=ordinary_dark_student_move(game,d,move,actor==viewer?&model:peers.historical,rules);
    game.play(move,viewer,rules);
  }
  return {review_outcome(game.state.result,viewer),game.state.ply-ply,game.trace};
}

static std::array<Move,4> choose_replan(const std::vector<EarlyPrefix>& worlds,const ReplanPortfolio& p,
    int viewer,const Rules& rules,const StudentNetwork& model,OrdinaryDarkPeers peers,uint64_t salt,int& steps) {
  std::array<Move,4> choices{p.base,p.wide,p.base,p.wide};
  if (worlds.size()<4) return choices;
  int best[2]{-1,-1};
  const int base=static_cast<int>(std::find(p.moves.begin(),p.moves.end(),p.base)-p.moves.begin());
  const int wide=static_cast<int>(std::find(p.moves.begin(),p.moves.end(),p.wide)-p.moves.begin());
  std::vector<int> lower,upper;
  for (const auto move:p.moves) {
    lower.push_back(0);
    upper.push_back(0);
    for (int world=0;world<2;++world) for (const int table:{0,2}) {
      const auto out=finish_replan(worlds[world],move,viewer,table,rules,model,peers,
                                   review_streams(mix_seed(salt,5608+world)));
      lower.back()+=out.result<0?0:out.result;
      upper.back()+=out.result<0?2:out.result;
      steps+=out.steps;
    }
  }
  for (int i=0;i<int(p.moves.size());++i) for (int arm=0;arm<2;++arm) {
    // Compare against the corresponding no-search arm when truncated outcomes are unresolved.
    if ((!arm && i>=p.shortlist) || lower[i]<=upper[arm?wide:base] || lower[i]<=best[arm]) continue;
    best[arm]=lower[i];
    choices[arm+2]=p.moves[i];
  }
  return choices;
}

struct ReplanWork {
  int runs[2]{},steps[2]{};
};

enum class ReplanMode { Factorial, Matched, Course };

struct ReplanRange {
  int low[2]{}, high[2]{};

  void add(int world,int score) {
    low[world]+=std::max(0,score);
    high[world]+=score<0?2:score;
  }

  int lower() const { return low[0]+low[1]; }
  int upper() const { return high[0]+high[1]; }
};

static std::array<Move,4> matched_replan(const std::vector<EarlyPrefix>& worlds,const ReplanPortfolio& p,
    int viewer,const Rules& rules,const StudentNetwork& model,OrdinaryDarkPeers peers,uint64_t salt,ReplanWork& work) {
  std::array<Move,4> choices{p.base,p.base,p.base,p.base};
  if (worlds.size()<4) return choices;
  const int narrow=p.shortlist,wide=int(p.moves.size());
  const int common=std::lcm(narrow,wide);
  const int budget=common*(common<2*wide?4:2);
  const int reps[2]{budget/narrow,budget/wide};
  const int base=int(std::find(p.moves.begin(),p.moves.end(),p.base)-p.moves.begin());
  std::vector<ReplanRange> values[2]{std::vector<ReplanRange>(narrow),std::vector<ReplanRange>(wide)};
  for (int i=0;i<wide;++i) {
    const int repeats=i<narrow?reps[0]:reps[1];
    for (int repeat=0;repeat<repeats;++repeat) {
      const int world=repeat%2,table=2*((repeat/2+world)%2);
      const auto payoff=finish_replan(worlds[world],p.moves[i],viewer,table,rules,model,peers,
                                     review_streams(mix_seed(salt,5608+repeat)));
      for (int arm:{0,1}) if (i<int(values[arm].size()) && repeat<reps[arm]) {
        values[arm][i].add(world,payoff.result);
        ++work.runs[arm];
        work.steps[arm]+=payoff.steps;
      }
    }
  }
  for (int arm=0;arm<3;++arm) {
    const auto& rows=values[arm==0?0:1];
    const auto reference=rows[base];
    int best=reference.upper();
    for (size_t i=0;i<rows.size();++i) {
      const auto& entry=rows[i];
      if (entry.lower()<=best) continue;
      if (arm==2 && (entry.low[0]<=reference.high[0] || entry.low[1]<=reference.high[1])) continue;
      best=entry.lower();
      choices[arm+1]=p.moves[i];
    }
  }
  return choices;
}

static void audit_replan_candidates(uint64_t seed,int viewer,int table,const SeatDecision& d,const OpeningMarks& evidence,
    const std::vector<EarlyPrefix>& worlds,const ReplanPortfolio& p,const Rules& rules,const StudentNetwork& model,
    OrdinaryDarkPeers peers,uint64_t salt) {
  if (worlds.size()!=4) return;
  for (const auto move:p.moves) {
    const int id=early_action_key(move);
    std::cout<<"Q "<<seed<<' '<<viewer<<' '<<table<<' '<<id;
    for (double x:student_features(d,evidence,move,&rules)) std::cout<<' '<<std::setprecision(17)<<x;
    std::cout<<'\n';
    for (int w:{2,3}) for (int desk:{0,4}) for (int i=0;i<2;++i) {
      const auto result=finish_replan(worlds[w],move,viewer,desk,rules,model,peers,
                                     review_streams(mix_seed(salt,67226+i)));
      std::cout<<"V "<<seed<<' '<<viewer<<' '<<table<<' '<<id<<' '<<w<<' '<<desk<<' '<<i<<' '
               <<result.result<<' '<<result.steps<<'\n';
    }
  }
}

static void judge_replan_arms(uint64_t seed,int viewer,int table,const EarlyPrefix& game,const ReplanWorlds& worlds,
    const std::array<Move,4>& choices,const Rules& rules,const StudentNetwork& model,OrdinaryDarkPeers peers,
    uint64_t salt,int replicas) {
  for (int domain=0;domain<(worlds.games.size()==4?3:1);++domain)
    for (int i=0;i<replicas;++i) for (int arm=0;arm<4;++arm) {
      const auto& state=domain?worlds.games[domain+1]:game;
      const auto out=finish_replan(state,choices[arm],viewer,table,rules,model,peers,
                                   review_streams(mix_seed(salt,8140+i)));
      std::cout<<"R "<<seed<<' '<<viewer<<' '<<table<<' '<<domain<<' '<<i<<' '<<arm<<' '<<out.result<<' '
               <<out.steps<<' '<<out.trace<<'\n';
    }
}

static void emit_replan_receipt(uint64_t seed,int viewer,int table,const ReplanRoot& root,const Rules& rules,
    const StudentNetwork& model,OrdinaryDarkPeers peers,int replicas,ReplanMode mode) {
  const auto& game=root.ordinary;
  const int ply=game.state.ply;
  if (!root.anchor || game.state.result!=Result::Ongoing || game.state.to_move!=viewer || game.state.out[viewer]) {
    std::cout<<"M "<<seed<<' '<<viewer<<' '<<table<<' '<<root.anchor_ply<<' '<<ply<<' '
             <<(game.state.out[viewer]?2:game.state.result!=Result::Ongoing?1:3)<<'\n';
    return;
  }
  const auto d=seat_decision(game.state);
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[viewer]);
  const auto salt=public_rollout_seed(d.observation,evidence,d.observation.quiet_clock);
  auto streams=root.streams;
  const auto fallback=ordinary_dark_move(game,d,rules,OrdinaryDarkPolicy::Tactical,streams[viewer]);
  const auto portfolio=replan_portfolio(d,evidence,fallback,rules,model,salt);
  const auto worlds=assemble_replan_worlds(root.path,d,rules,salt);
  int steps=0;
  ReplanWork work;
  const bool matched=mode!=ReplanMode::Factorial;
  const auto choices=matched?matched_replan(worlds.games,portfolio,viewer,rules,model,peers,salt,work):
                             choose_replan(worlds.games,portfolio,viewer,rules,model,peers,salt,steps);
  if (matched) {
    steps=work.steps[0]+work.steps[1];
    std::cout<<"K "<<seed<<' '<<viewer<<' '<<table<<' '<<work.runs[0]<<' '<<work.runs[1]<<' '
             <<work.steps[0]<<' '<<work.steps[1]<<'\n';
  }
  int isolation=0;
  if (worlds.games.size()==4 && seed%8==0) {
    auto alternate=root.path;
    alternate.game=*worlds.last_strict;
    const auto view=seat_decision(worlds.games.back().state);
    const auto marks=worlds.games.back().reply_history.for_actor(view.observation,worlds.games.back().marks[viewer]);
    const auto blinded=replan_portfolio(view,marks,fallback,rules,model,salt);
    const auto reference=assemble_replan_worlds(alternate,view,rules,salt);
    int check_steps=0;
    ReplanWork check;
    const auto selected=matched?matched_replan(reference.games,blinded,viewer,rules,model,peers,salt,check):
                                choose_replan(reference.games,blinded,viewer,rules,model,peers,salt,check_steps);
    if (matched) check_steps=check.steps[0]+check.steps[1];
    single_case_check(portfolio.moves==blinded.moves && reference.games.size()==worlds.games.size() &&
        selected==choices && steps==check_steps,
        "true hidden layout affected post-reply selection");
    isolation=1;
  }
  std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<root.anchor_ply<<' '<<ply<<' '<<early_action_key(*root.anchor)
           <<' '<<d.actions.size()<<' '<<portfolio.shortlist<<' '<<portfolio.moves.size()<<' '<<worlds.games.size()
           <<' '<<worlds.solved<<' '<<worlds.rejected<<' '<<worlds.duplicates<<' '<<worlds.nodes<<' '<<steps<<' '<<isolation;
  for (int c:worlds.reasons) std::cout<<' '<<c;
  for (auto move:choices) std::cout<<' '<<early_action_key(move);
  const auto baseline=finish_replan(game,portfolio.base,viewer,table,rules,model,peers,streams);
  std::cout<<' '<<baseline.result<<' '<<ply+baseline.steps<<' '<<baseline.trace<<'\n';
  judge_replan_arms(seed,viewer,table,game,worlds,choices,rules,model,peers,salt,replicas);
  if (mode==ReplanMode::Course)
    audit_replan_candidates(seed,viewer,table,d,evidence,worlds.games,portfolio,rules,model,peers,salt);
}

static void run_dark_replanning(const char* path,ReplanMode mode) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(path),rules,error),"invalid post-reply rules");
  StudentNetwork model,history;
  single_case_check(model.load(std::cin,true) && history.load(std::cin),"invalid post-reply models");
  const OrdinaryDarkPeers peers{&history,true};
  uint64_t seed;
  int viewer,table,cut,forced,replicas;
  while (std::cin>>seed>>viewer>>table>>cut>>forced>>replicas) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==2 || table==4) && cut>=0 &&
                      replicas>=1 && replicas<=8,"invalid post-reply assignment");
    const auto root=restore_replan_root(seed,viewer,table,cut,forced,rules,model,peers);
    emit_replan_receipt(seed,viewer,table,root,rules,model,peers,replicas,mode);
  }
  single_case_check(std::cin.eof(),"incomplete post-reply assignment");
}
