#pragma once

constexpr const char* student_encoding="junqi-public-early-action-v1";
constexpr const char* exchange_encoding="junqi-public-early-action-collision-v1";
constexpr const char* reply_window_encoding="junqi-full-reply-window-action-v1";
constexpr size_t student_inputs=41, exchange_inputs=44, student_heads=7;
constexpr size_t reply_window_inputs=exchange_inputs+8;

static bool active_full_research() {
  static const bool enabled=[] {
#ifdef _MSC_VER
    char* value=nullptr;
    size_t size=0;
    const bool present=_dupenv_s(&value,&size,"JUNQI_FULL_RESEARCH_ACTIVE")==0 && size>0;
    std::free(value);
    return present;
#else
    return std::getenv("JUNQI_FULL_RESEARCH_ACTIVE")!=nullptr;
#endif
  }();
  return enabled;
}

static std::vector<double> student_features(const jq::SeatDecision& d,const jq::OpeningMarks& marks,jq::Move move,
                                            const jq::Rules* collision_rules=nullptr) {
  using namespace jq;
  const auto& v=d.observation;
  const auto& b=Board::get();
  const int flag=own_flag(v);
  const int owner=v.seat_at(move.to);
  const auto target=owner>=0?compatible_rank_mask(v,marks,move.to):0u;
  const auto goals=flag_targets(v,marks);
  const auto geometry=[&](int n) {
    const auto& node=b.node(n);
    return std::array<double,5>{node.seat<0?1.:double((node.seat-v.viewer+kSeats)%kSeats)/4.,
      node.r/5.,std::abs(node.c-2)/3.,double(node.rail),double(node.kind==NodeKind::HQ)};
  };
  std::vector<double> x{v.ply/140.,v.quiet_clock/70.,std::max(0,100-v.ply)/100.,
    double(d.actions.size())/200.,v.rank_at(move.from)/11.,double(owner>=0),
    flag<0?0.:distance(move.from,flag)/20.,flag<0?0.:distance(move.to,flag)/20.,
    goals.empty()?0.:flag_distance(move.from,goals)/20.,goals.empty()?0.:flag_distance(move.to,goals)/20.,
    double(marks.moved[move.to])};
  for (int n:{move.from,move.to}) for (const auto f:geometry(n)) x.push_back(f);
  for (const int rank:{Flag,Mine,Bomb,Engineer,Marshal})
    x.push_back(double((target&(1u<<rank))!=0));
  for (int seat=0;seat<kSeats;++seat) {
    int pieces=0,mobile=0,engineer=0;
    const int relative=(v.viewer+seat)%kSeats;
    for (int n=0;n<kNodes;++n) if (v.seat_at(n)==relative) {
      ++pieces;
      const auto ranks=compatible_rank_mask(v,marks,n);
      if (can_leave_node(Engineer,n) && (ranks&~immobile_ranks)) {
        ++mobile;
        engineer+=(ranks&(1u<<Engineer))!=0;
      }
    }
    x.insert(x.end(),{pieces/25.,mobile/21.,engineer/21.});
  }
  const double contact=owner>=0?1.:0.;
  x.insert(x.end(),{contact*x[0],contact*x[1],(x[6]-x[7])*contact});
  if (collision_rules) {
    const auto possible=collision_outcome_mask(v.rank_at(move.from),target,*collision_rules);
    for (const auto result:{Outcome::AttackerWins,Outcome::DefenderWins,Outcome::BothRemoved})
      x.push_back(double((possible & outcome_bit(result))!=0));
  }
  single_case_check(x.size()==(collision_rules?exchange_inputs:student_inputs),"student feature width mismatch");
  return x;
}

static void exchange_feature_fixtures(const jq::Rules& rules) {
  using namespace jq;
  const Move move{4,30};
  auto state=single_case_state();
  place_single_case_piece(state,1,move.from,Division);
  place_single_case_piece(state,26,move.to,Brigade);
  auto lower=seat_decision(state);
  state.rank_at[move.to]=state.rank_of[26]=Corps;
  auto higher=seat_decision(state);
  const OpeningMarks marks(lower.observation,false);
  single_case_check(is_legal(state,move) && lower.actions==higher.actions &&
      student_features(lower,marks,move,&rules)==student_features(higher,marks,move,&rules),
      "exchange feature leaked unobserved rank");
  lower.observation.ranks[move.to]=Brigade;
  higher.observation.ranks[move.to]=Corps;
  const auto a=student_features(lower,marks,move,&rules);
  const auto b=student_features(higher,marks,move,&rules);
  single_case_check(student_features(lower,marks,move)==student_features(higher,marks,move) &&
      a[41]==1 && a[42]==0 && b[41]==0 && b[42]==1,"public collision contrast was not resolved");
}

struct StudentNetwork {
  std::vector<double> mean, scale, inner, bias, outer, output;
  size_t width=0;
  bool exchange=false;

  bool load(std::istream& stream,bool use_exchange=false,bool use_reply_window=false) {
    std::string encoding;
    size_t dimension=0,heads=0;
    exchange=use_exchange;
    const auto format=use_reply_window?reply_window_encoding:exchange?exchange_encoding:student_encoding;
    const auto inputs=use_reply_window?reply_window_inputs:exchange?exchange_inputs:student_inputs;
    if (!(stream>>encoding>>dimension>>heads>>width) || encoding!=format ||
        dimension!=inputs || heads!=student_heads ||
        (width!=0 && width!=24)) return false;
    const auto vector=[&](std::vector<double>& list,size_t size) {
      list.resize(size);
      for (auto& x:list) if (!(stream>>x) || !std::isfinite(x)) return false;
      return true;
    };
    const bool loaded=vector(mean,dimension) && vector(scale,dimension) &&
           std::all_of(scale.begin(),scale.end(),[](double x) { return x>0; }) &&
           vector(inner,dimension*width) && vector(bias,width) &&
           vector(outer,(width?width:dimension)*heads) && vector(output,heads);
    if (loaded && active_full_research()) {
      auto hash=jq::fnv1a(encoding.data(),encoding.size());
      for (const auto* row:{&mean,&scale,&inner,&bias,&outer,&output})
        hash=jq::fnv1a(row->data(),row->size()*sizeof(double),hash);
      std::cerr<<"JUNQI_ACTIVE_MODEL "<<encoding<<' '<<hash<<'\n';
    }
    return loaded;
  }

  std::vector<double> logits(std::vector<double> x) const {
    for (size_t i=0;i<x.size();++i) x[i]=std::clamp((x[i]-mean[i])/scale[i],-6.,6.);
    if (width) {
      auto hidden=bias;
      for (size_t i=0;i<x.size();++i) for (size_t j=0;j<width;++j) hidden[j]+=x[i]*inner[i*width+j];
      for (auto& value:hidden) value=std::tanh(value);
      x=std::move(hidden);
    }
    auto result=output;
    for (size_t i=0;i<x.size();++i) for (size_t j=0;j<student_heads;++j) result[j]+=x[i]*outer[i*student_heads+j];
    return result;
  }

  double score(std::vector<double> x,bool deadline=true) const {
    const auto result=logits(std::move(x));
    const auto m=*std::max_element(result.begin(),result.begin()+3);
    const double win=std::exp(result[2]-m), draw=std::exp(result[1]-m);
    const auto sigmoid=[](double x) { return 1./(1.+std::exp(-std::clamp(x,-35.,35.))); };
    const double team=(win+.5*draw)/(win+draw+std::exp(result[0]-m));
    return deadline?team+.05*sigmoid(result[3])+.15*sigmoid(result[4])-.10*sigmoid(result[5]):team;
  }
};

static void export_student_features(std::istream& in,const jq::Rules* collision_rules=nullptr) {
  using namespace jq;
  int count=0;
  if (!(in>>count) || count<1 || count>4096) throw std::runtime_error("invalid student feature batch");
  for (int row=0;row<count;++row) {
    SeatDecision d{};
    int legal=0;
    auto& v=d.observation;
    if (!(in>>v.viewer>>v.ply>>v.quiet_clock>>legal) || v.viewer<0 || v.viewer>=kSeats ||
        legal<1 || legal>kNodes*kNodes || v.ply<0 || v.quiet_clock<0)
      throw std::runtime_error("invalid student feature prefix");
    v.turn=v.viewer;
    std::fill(std::begin(v.seats),std::end(v.seats),static_cast<int8_t>(kNone));
    std::fill(std::begin(v.ranks),std::end(v.ranks),static_cast<int8_t>(Unknown));
    OpeningMarks marks(v,false);
    for (int n=0;n<kNodes;++n) {
      int seat,rank,moved;
      if (!(in>>seat>>rank>>marks.ranks[n]>>moved)) throw std::runtime_error("short student board");
      v.seats[n]=static_cast<int8_t>(seat);
      v.ranks[n]=static_cast<int8_t>(rank);
      marks.moved[n]=moved!=0;
    }
    for (int i=0;i<legal;++i) {
      int m=0;
      if (!(in>>m) || m<0 || m>=kNodes*kNodes) throw std::runtime_error("invalid student legal action");
      d.actions.push_back({static_cast<uint8_t>(m/kNodes),static_cast<uint8_t>(m%kNodes)});
    }
    int choices=0;
    if (!(in>>choices) || choices<1 || choices>6) throw std::runtime_error("invalid student shortlist");
    for (int i=0;i<choices;++i) {
      int m=0;
      if (!(in>>m)) throw std::runtime_error("incomplete student shortlist");
      const auto move=std::find_if(d.actions.begin(),d.actions.end(),[&](auto a) { return early_action_key(a)==m; });
      if (move==d.actions.end()) throw std::runtime_error("student action absent from legal set");
      for (const double x:student_features(d,marks,*move,collision_rules)) std::cout<<std::setprecision(17)<<x<<' ';
      std::cout<<'\n';
    }
  }
}

static jq::Move student_pick(const jq::SeatDecision& d,const jq::OpeningMarks& marks,jq::Move fallback,
                            const std::vector<jq::Move>& actions,const StudentNetwork& model,
                            const jq::Rules& rules,bool deadline=true) {
  const auto* collision_rules=model.exchange?&rules:nullptr;
  auto choice=fallback;
  double best=model.score(student_features(d,marks,choice,collision_rules),deadline);
  for (const auto action:actions) {
    const auto value=model.score(student_features(d,marks,action,collision_rules),deadline);
    if (value>best) { best=value; choice=action; }
  }
  return choice;
}

static void student_trial(uint64_t seed,int learner,int partner,int table,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& history,const StudentNetwork* student,
    bool contact_detail=false) {
  using namespace jq;
  EarlyPrefix game(opening(seed,rules),seed);
  const int first=game.state.to_move;
  bool visited[2]{};
  int changes=0, opportunities=0;
  int legal_contact=0, listed_contact=0, selected_contact=0, won_contact=0;
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    const int actor=d.observation.viewer;
    auto choice=early_normal(game,d,learner,table,rules,parent,history,partner);
    const int ply=d.observation.ply, window=ply>=56?1:0;
    bool inspect_contact=false;
    if (actor==learner && ply>=24 && ply<100 && !visited[window]) {
      visited[window]=true;
      const auto& marks=game.marks[actor];
      const auto actions=response_candidates(d,marks,choice,rules,public_rollout_seed(d.observation,marks,
                                                                      d.observation.quiet_clock));
      opportunities+=actions.size()>1;
      if (student) {
        const auto fallback=choice;
        choice=student_pick(d,marks,fallback,actions,*student,rules);
        changes+=!(choice==fallback);
      }
      if (contact_detail) {
        const auto contact=[&](Move m) { return d.observation.seat_at(m.to)>=0; };
        legal_contact+=std::any_of(d.actions.begin(),d.actions.end(),contact);
        // The fallback is always scored even if the shortlist is a singleton.
        listed_contact+=contact(choice) || std::any_of(actions.begin(),actions.end(),contact);
        selected_contact+=contact(choice);
        inspect_contact=contact(choice);
      }
    }
    const auto outcome=game.play(choice,learner,rules);
    won_contact+=inspect_contact && outcome==Outcome::AttackerWins;
  }
  std::cout<<seed<<' '<<learner<<' '<<partner<<' '<<table<<' '<<game.state.ply<<' '
           <<review_outcome(game.state.result,learner)<<' '<<changes<<' '<<opportunities<<' '<<game.trace
           <<' '<<first<<' '<<game.friendly_exit<<' '<<game.enemy_exit;
  if (contact_detail) std::cout<<' '<<legal_contact<<' '<<listed_contact<<' '<<selected_contact<<' '<<won_contact;
  std::cout<<'\n';
}

static void student_feedback_trial(uint64_t seed,int learner,int table,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& history,const StudentNetwork* student,
    uint64_t reply_salt=0) {
  using namespace jq;
  EarlyPrefix game(opening(seed,rules),seed);
  bool exported[2]{};
  while (game.state.result==Result::Ongoing && game.state.ply<100) {
    const auto d=seat_decision(game.state);
    const auto base=early_normal(game,d,learner,table,rules,parent,history);
    auto chosen=base;
    const int window=game.state.ply>=56?1:0;
    if (d.observation.viewer==learner && game.state.ply>=24 && !exported[window]) {
      const auto& marks=game.marks[learner];
      if (!window && student) {
        const auto actions=response_candidates(d,marks,base,rules,public_rollout_seed(d.observation,marks,
                                                                      d.observation.quiet_clock));
        chosen=student_pick(d,marks,base,actions,*student,rules);
      }
      early_export(game,d,base,learner,table,window,seed,rules,parent,history,1,reply_salt);
      if (!window) std::cout<<"S "<<seed<<' '<<learner<<' '<<table<<' '<<early_action_key(base)
                            <<' '<<early_action_key(chosen)<<'\n';
      exported[window]=true;
    }
    game.play(chosen,learner,rules);
  }
  std::cout<<"G "<<seed<<' '<<learner<<' '<<table<<' '<<exported[0]<<' '<<exported[1]<<'\n';
}
