#pragma once

enum class OrdinaryDarkPolicy { Route, Tactical, Terminal };

static OrdinaryDarkPolicy ordinary_dark_peer(int actor,int learner,int table,const Rules& rules) {
  const bool partner=actor==(learner+2)%kSeats;
  if (actor==learner) return OrdinaryDarkPolicy::Tactical;
  if (table==1 && partner) return OrdinaryDarkPolicy::Route;
  if (table==2 && actor==next_seat(learner,rules)) return OrdinaryDarkPolicy::Route;
  if (table==3 && !partner) return OrdinaryDarkPolicy::Terminal;
  return OrdinaryDarkPolicy::Tactical;
}

static Move ordinary_dark_move(const EarlyPrefix& game,const SeatDecision& decision,const Rules& rules,
                                OrdinaryDarkPolicy policy,Rng& rng) {
  const DecisionKind kind=policy==OrdinaryDarkPolicy::Route?DecisionKind::FlagRoute:
      policy==OrdinaryDarkPolicy::Tactical?DecisionKind::CollisionHeuristic:DecisionKind::TerminalHeuristic;
  const int actor=decision.observation.viewer;
  Move move;
  single_case_check(DecisionScorer(decision.observation,rules,kind).choose(rng,move),"ordinary game lost valid moves");
  if (policy==OrdinaryDarkPolicy::Tactical)
    return plan(decision,game.marks[actor],move,game.reply_history.last_moves()[actor],rules);
  return move;
}

struct OrdinaryDarkActions {
  int turns=0,changed=0,first=-1,base=-1,choice=-1;
};

struct OrdinaryDarkPeers {
  const StudentNetwork* historical=nullptr;
  bool mixed_tables=false;

  bool model_seat(int actor,int learner,int table,const Rules& rules) const {
    if (!historical || actor==learner) return false;
    const bool partner=actor==(learner+2)%kSeats;
    const bool next_enemy=actor==next_seat(learner,rules);
    if (mixed_tables) return (table==1 && partner) || ((table==3 || table==4) && next_enemy);
    return ((table&1) && partner) || ((table&2) && next_enemy);
  }

  OrdinaryDarkPolicy policy(int actor,int learner,int table,const Rules& rules) const {
    if (!historical) return ordinary_dark_peer(actor,learner,table,rules);
    const bool partner=actor==(learner+2)%kSeats;
    if (mixed_tables && actor!=learner && (partner?(table==2 || table==4):table==5))
      return OrdinaryDarkPolicy::Terminal;
    return OrdinaryDarkPolicy::Tactical;
  }
};

static Move ordinary_dark_student_move(const EarlyPrefix& game,const SeatDecision& d,Move baseline,
                                       const StudentNetwork* model,const Rules& rules) {
  const int actor=d.observation.viewer;
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[actor]);
  if (!model) return baseline;
  const auto candidates=budget_candidates(d,baseline,rules,4);
  return student_pick(d,evidence,baseline,candidates,*model,rules,false);
}

struct OrdinaryDarkCollection {
  int remaining;
  bool explore;

  Move choose(const SeatDecision& d,const OpeningMarks& marks,Move prior,const Rules& rules,
              uint64_t seed,int table) const {
    const auto& view=d.observation;
    if (view.ply<8 || view.ply>=120 || public_terminal_flag(view,prior,rules,occupied_enemies(view)))
      return prior;
    auto candidates=budget_candidates(d,prior,rules,4);
    candidates.erase(std::remove_if(candidates.begin()+1,candidates.end(),[&](Move move) {
      const auto target=compatible_rank_mask(view,marks,move.to);
      return target && collision_outcome_mask(view.rank_at(move.from),target,rules)==outcome_bit(Outcome::DefenderWins);
    }),candidates.end());
    if (candidates.size()<2) return prior;
    Rng random(mix_seed(seed,508819+view.ply));
    const bool sample=explore && random.below(8)==0;
    const int alternatives=int(candidates.size()-1);
    const Move picked=sample?candidates[1+random.below(alternatives)]:prior;
    const int denominator=explore?8*alternatives:1;
    const int numerator=sample?1:explore?7*alternatives:1;
    std::cout<<"J "<<seed<<' '<<view.viewer<<' '<<table<<' '<<view.ply<<' '<<early_action_key(prior)<<' '
             <<early_action_key(picked)<<' '<<alternatives<<' '<<numerator<<' '<<denominator<<'\n';
    return picked;
  }
};

static int ordinary_dark_game(uint64_t seed,int learner,int table,int arm,const Rules& rules,
                             const StudentNetwork* model,bool export_examples,std::vector<int>* actions=nullptr,
                             bool whole_game_samples=false,OrdinaryDarkPeers peers={},OrdinaryDarkCollection* course=nullptr) {
  const auto start=opening(seed,rules);
  auto game=EarlyPrefix(start,seed);
  auto streams=review_streams(mix_seed(seed,70719));
  OrdinaryDarkActions trace;
  std::vector<std::vector<double>> examples;
  int next_sample=0;
  int peer_turns=0,peer_changes=0;
  const int cap=course?std::min(720,course->remaining):720;
  while (game.state.result==Result::Ongoing && game.state.ply<cap) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    const auto policy=actor==learner && arm==0?OrdinaryDarkPolicy::Route:peers.policy(actor,learner,table,rules);
    const auto baseline=ordinary_dark_move(game,d,rules,policy,streams[actor]);
    auto move=baseline;
    if (actor==learner) {
      ++trace.turns;
      move=ordinary_dark_student_move(game,d,baseline,model,rules);
      if (course) {
        const auto evidence=game.reply_history.for_actor(d.observation,game.marks[actor]);
        move=course->choose(d,evidence,move,rules,seed,table);
      }
      if (trace.changed==0 && !(move==baseline)) {
        trace.first=game.state.ply;
        trace.base=early_action_key(baseline);
        trace.choice=early_action_key(move);
      }
      trace.changed+=!(move==baseline);
      if (export_examples && game.state.ply>=next_sample && (whole_game_samples || examples.size()<6)) {
        const auto evidence=game.reply_history.for_actor(d.observation,game.marks[actor]);
        examples.push_back(student_features(d,evidence,move,&rules));
        next_sample=game.state.ply+24;
      }
    } else if (peers.model_seat(actor,learner,table,rules)) {
      move=ordinary_dark_student_move(game,d,baseline,peers.historical,rules);
      ++peer_turns;
      peer_changes+=!(move==baseline);
    }
    if (actions) actions->push_back(early_action_key(move));
    game.play(move,learner,rules);
  }
  if (examples.size()>6) {
    std::vector<std::vector<double>> evenly_spaced;
    for (size_t i=0;i<6;++i) evenly_spaced.push_back(std::move(examples[i*(examples.size()-1)/5]));
    examples=std::move(evenly_spaced);
  }
  const int outcome=review_outcome(game.state.result,learner);
  if (course) course->remaining-=game.state.ply;
  std::cout<<"G "<<seed<<' '<<learner<<' '<<table<<' '<<arm<<' '<<outcome<<' '<<game.state.ply<<' '
           <<game.trace<<' '<<trace.turns<<' '<<trace.changed<<' '<<trace.first<<' '<<trace.base<<' '<<trace.choice
           <<' '<<examples.size();
  if (peers.historical) std::cout<<' '<<peer_turns<<' '<<peer_changes;
  std::cout<<'\n';
  for (const auto& x:examples) {
    std::cout<<"E "<<seed<<' '<<learner<<' '<<table<<' '<<arm<<' '<<outcome;
    for(double v:x) std::cout<<' '<<std::setprecision(17)<<v;
    std::cout<<'\n';
  }
  return outcome;
}

static void ordinary_dark_pair(uint64_t seed,int learner,int table,const Rules& rules,
                               const StudentNetwork& parent,const StudentNetwork& child,
                               OrdinaryDarkPeers peers={}) {
  std::vector<int> a,b;
  ordinary_dark_game(seed,learner,table,2,rules,&parent,false,&a,false,peers);
  ordinary_dark_game(seed,learner,table,3,rules,&child,false,&b,false,peers);
  const auto [old_move,new_move]=std::mismatch(a.begin(),a.end(),b.begin(),b.end());
  const bool identical=old_move==a.end() && new_move==b.end();
  single_case_check(identical || (old_move!=a.end() && new_move!=b.end()),"equal dark action prefixes reached different terminals");
  std::cout<<"R "<<seed<<' '<<learner<<' '<<table<<' '<<(identical?-1:static_cast<int>(old_move-a.begin()))
           <<' '<<(identical?-1:*old_move)<<' '<<(identical?-1:*new_move)<<'\n';
}

static void run_dark_ordinary_learning(const char* path) {
  Rules rules;
  std::string error;
  single_case_check(parse_rules(read_text(path),rules,error),"invalid ordinary four-dark rules");
  char mode;
  single_case_check(bool(std::cin>>mode) && std::string("PDLEHMXY").find(mode)!=std::string::npos,
                    "invalid ordinary learning wire");
  StudentNetwork parent,child,historical;
  const bool sample=mode=='D' || mode=='L' || mode=='M';
  const bool pair=mode=='E' || mode=='H' || mode=='X';
  const bool with_peer=mode=='H' || mode=='M' || mode=='X';
  if (mode!='P') single_case_check(parent.load(std::cin,true),"invalid ordinary parent weights");
  if (pair) single_case_check(child.load(std::cin,true),"invalid ordinary child weights");
  if (with_peer) single_case_check(historical.load(std::cin),"invalid historical 41-feature peer");
  const OrdinaryDarkPeers peers{with_peer?&historical:nullptr,mode=='M' || mode=='X'};
  OrdinaryDarkCollection course{0,false};
  if (mode=='Y')
    single_case_check(bool(std::cin>>course.remaining>>course.explore) && course.remaining>0 &&
                      course.remaining<=120000,"invalid direct-experience budget");
  uint64_t seed;
  int learner,table;
  while (std::cin>>seed>>learner>>table) {
    const int tables=mode=='M'?2:mode=='X'?6:4;
    single_case_check(learner>=0 && learner<4 && table>=0 && table<tables,"invalid ordinary game assignment");
    if (mode=='P') {
      for (int arm=0;arm<2;++arm) ordinary_dark_game(seed,learner,table,arm,rules,nullptr,false);
    } else if (mode=='Y') {
      if (course.remaining)
        ordinary_dark_game(seed,learner,table,2,rules,&parent,true,nullptr,false,peers,&course);
    } else if (sample) {
      const bool use_parent=(seed%2)==0;
      ordinary_dark_game(seed,learner,table,use_parent?2:1,rules,use_parent?&parent:nullptr,true,nullptr,mode=='L',peers);
    } else {
      ordinary_dark_pair(seed,learner,table,rules,parent,child,peers);
    }
  }
  single_case_check(std::cin.eof(),"incomplete ordinary learning assignments");
  if (mode=='Y') std::cout<<"B "<<course.remaining<<'\n';
}
