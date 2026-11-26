#pragma once
#include "../sim/public_flag_reply.h"
#include "deadline_public_flag_attribution.h"

struct ReplyNomination {
  Move base, candidate, selected;
  FlagReply risk=FlagReply::Unresolved;
  bool existing=false, proposed=false;
};

static ReplyNomination nominate_flag_reply(const SeatDecision& d,const OpeningMarks& marks,Move fallback,
                                           const StudentNetwork& model,const Rules& rules) {
  auto list=budget_candidates(d,fallback,rules,4);
  auto base=student_pick(d,marks,fallback,list,model,rules,false);
  ReplyNomination result{base,base,base};
  const auto& view=d.observation;
  if (public_terminal_flag(view,base,rules,occupied_enemies(view))) return result;
  result.risk=public_flag_reply(view,marks,base,rules);
  if (result.risk!=FlagReply::Risk) return result;
  for (auto move:list)
    if (!(move==base) && public_flag_reply(view,marks,move,rules)==FlagReply::Clear) {
      result.existing=true;
      return result;
    }
  const auto victim=std::find_if(list.rbegin(),list.rend(),[&](Move move) {
    return !(move==base) && !(move==fallback);
  });
  if (victim==list.rend()) return result;
  DecisionScorer order(view,rules,DecisionKind::FlagRoute);
  int best=-100000;
  for (auto move:d.actions) {
    if (std::find(list.begin(),list.end(),move)!=list.end() ||
        public_flag_reply(view,marks,move,rules)!=FlagReply::Clear) continue;
    const int value=order.score(move).total();
    if (value<=best) continue;
    best=value;
    result.proposed=true;
    result.candidate=move;
  }
  if (result.proposed) {
    *victim=result.candidate;
    result.selected=student_pick(d,marks,fallback,list,model,rules,false);
  }
  return result;
}

static bool actual_flag_reply(State world,Move first,const Rules& rules) {
  const int team=team_of(world.to_move);
  single_case_check(is_legal(world,first),"classifier returned an illegal move");
  step(world,first,rules);
  if (world.result!=Result::Ongoing || team_of(world.to_move)==team) return false;
  std::vector<Move> legal;
  legal_moves(world,legal);
  return std::any_of(legal.begin(),legal.end(),[&](Move move) {
    return world.rank_at[move.to]==Flag &&
           resolve(world.rank_at[move.from],Flag,rules)!=Outcome::DefenderWins;
  });
}

static void flag_reply_controls(Rules rules) {
  auto root=single_case_state();
  root.ply=root.no_progress=0;
  place_single_case_piece(root,1,20,Marshal);
  place_single_case_piece(root,2,4,Engineer);
  place_single_case_piece(root,26,21,Engineer);
  place_single_case_piece(root,27,51,Engineer);
  const Move wait{4,3}, guard{20,21};
  const auto view=observe(root,0);
  OpeningMarks marks(view,false);
  marks.ranks[21]=(1u<<Engineer)|(1u<<Corps);
  marks.ranks[51]=1u<<Engineer;
  const auto verify=[&](const State& world,const OpeningMarks& evidence,FlagReply expected,Move move) {
    const auto claim=public_flag_reply(observe(world,0),evidence,move,rules);
    single_case_check(claim==expected,"unexpected public flag-reply control");
    if (claim==FlagReply::Clear)
      single_case_check(!actual_flag_reply(world,move,rules),"certified defense allows an immediate flag capture");
  };
  single_case_check(is_legal(root,wait) && actual_flag_reply(root,wait,rules),"missing hostile flag reply");
  verify(root,marks,FlagReply::Risk,wait);
  verify(root,marks,FlagReply::Clear,guard);
  auto alternate=root;
  alternate.rank_at[21]=alternate.rank_of[26]=Corps;
  single_case_check(same_observation(view,observe(alternate,0)),"fixture hidden swap leaked");
  verify(alternate,marks,FlagReply::Clear,guard);
  marks.ranks[21]|=1u<<Marshal;
  verify(root,marks,FlagReply::Clear,guard);
  marks.ranks[21]|=1u<<Bomb;
  verify(root,marks,FlagReply::Clear,guard);
  auto weaker=alternate;
  weaker.rank_at[20]=weaker.rank_of[1]=Brigade;
  single_case_check(actual_flag_reply(weaker,guard,rules),"missing failed interceptor reply");
  verify(weaker,marks,FlagReply::Risk,guard);
  auto last=root;
  last.alive[27]=false;
  last.rank_at[51]=last.id_at[51]=last.seat_at[51]=kNone;
  verify(last,marks,FlagReply::Unresolved,guard);
  marks.ranks[21]=(1u<<Engineer)|(1u<<Corps);
  rules.no_progress_limit=1;
  verify(root,marks,FlagReply::Unresolved,wait);
  rules.no_progress_limit=70;
  auto skipped=root;
  skipped.out[3]=false;
  place_single_case_piece(skipped,75,116,Flag);
  skipped.id_at[21]=skipped.seat_at[21]=kNone;
  skipped.rank_at[21]=Unknown;
  skipped.alive[26]=false;
  skipped.id_at[51]=skipped.seat_at[51]=kNone;
  skipped.rank_at[51]=Unknown;
  skipped.alive[27]=false;
  place_single_case_piece(skipped,76,21,Engineer);
  OpeningMarks skip_marks(observe(skipped,0),false);
  skip_marks.ranks[21]=1u<<Engineer;
  single_case_check(actual_flag_reply(skipped,wait,rules),"missing flag attack after skipped seats");
  verify(skipped,skip_marks,FlagReply::Risk,wait);
  auto terminal=root;
  place_single_case_piece(terminal,3,55,Engineer);
  const auto last_flag=observe(terminal,0,IdentityVisibility::AllIdentities);
  const Move win{55,56};
  single_case_check(is_legal(terminal,win) &&
    public_terminal_flag(last_flag,win,rules,occupied_enemies(last_flag)) &&
    public_flag_reply(last_flag,OpeningMarks(last_flag,false),win,rules)==FlagReply::Unresolved,
    "terminal flag capture must be left to the existing finish path");
  std::cout<<"CONTROL public-flag-reply\n";
}

struct FlagReplyTelemetry {
  int plies=0, contacts=0, flag_captures=0, known_flag_offers=0, known_flag_taken=0, partner_replies=0;
  int risk=0, clear=0, unresolved=0;
  Move last_contact[kSeats]{};
  int contact_ply[kSeats]{};

  void action(const SeatDecision& d,Move move,const Rules& rules,FlagReply claim) {
    const auto& view=d.observation;
    const int actor=view.viewer, partner=(actor+2)%kSeats;
    const bool offer=std::any_of(d.actions.begin(),d.actions.end(),[&](Move m) {
      return view.rank_at(m.to)==Flag && resolve(view.rank_at(m.from),Flag,rules)!=Outcome::DefenderWins;
    });
    ++plies;
    known_flag_offers+=offer;
    known_flag_taken+=view.rank_at(move.to)==Flag;
    risk+=claim==FlagReply::Risk;
    clear+=claim==FlagReply::Clear;
    unresolved+=claim==FlagReply::Unresolved;
    if (view.seat_at(move.to)<0) return;
    ++contacts;
    partner_replies+=contact_ply[partner]>0 && view.ply+1-contact_ply[partner]<=3 &&
                     last_contact[partner].to==move.to;
    last_contact[actor]=move;
    contact_ply[actor]=view.ply+1;
  }

  void print() const {
    std::cout<<' '<<plies<<' '<<contacts<<' '<<flag_captures<<' '<<known_flag_offers<<' '
             <<known_flag_taken<<' '<<partner_replies<<' '<<risk<<' '<<clear<<' '<<unresolved;
  }
};

static void pressure_flag_game(State start,uint64_t seed,int learner,int table,int arm,
                               const StudentNetwork& model,const Rules& rules,FlagAttribution* audit=nullptr) {
  EarlyPrefix game(start,seed);
  auto streams=review_streams(mix_seed(seed,70719));
  FlagReplyTelemetry board[2];
  int danger=0,in_list=0,nominated=0,selected=0,exact_base_risk=0,validated_selected=0;
  while (game.state.result==Result::Ongoing && game.state.ply<720) {
    const auto d=seat_decision(game.state);
    const auto& view=d.observation;
    const auto actor=view.viewer;
    const bool own=view.team(actor)==view.team(learner);
    bool next_enemy=false;
    const auto policy=!own && table==0?OrdinaryDarkPolicy::Route:OrdinaryDarkPolicy::Tactical;
    auto action=ordinary_dark_move(game,d,rules,policy,streams[actor]);
    const auto marks=game.reply_history.for_actor(view,game.marks[actor]);
    if (own) {
      const auto choice=nominate_flag_reply(d,marks,action,model,rules);
      danger+=choice.risk==FlagReply::Risk;
      if (audit && choice.risk==FlagReply::Risk)
        next_enemy=audit->report_risk(game.state,d,marks,action,choice.base,rules);
      in_list+=choice.existing;
      nominated+=choice.proposed;
      selected+=!(choice.selected==choice.base);
      if (choice.proposed) {
        single_case_check(!actual_flag_reply(game.state,choice.candidate,rules),"native nomination disproved");
        const bool attack=actual_flag_reply(game.state,choice.base,rules);
        exact_base_risk+=attack;
        validated_selected+=attack && !(choice.selected==choice.base);
        std::cout<<"N "<<seed<<' '<<table<<' '<<arm<<' '<<view.ply+1<<' '<<actor<<' '
                 <<early_action_key(choice.base)<<' '<<early_action_key(choice.candidate)<<' '
                 <<early_action_key(choice.selected)<<' '<<attack<<'\n';
      }
      action=arm?choice.selected:choice.base;
    } else if (const auto win=public_flag_finish(d,action,rules)) {
      action=*win;
    }
    const auto claim=public_flag_reply(view,marks,action,rules);
    if (claim==FlagReply::Clear)
      single_case_check(!actual_flag_reply(game.state,action,rules),"native flag reply disproved");
    auto& meter=board[!own];
    meter.action(d,action,rules,claim);
    const bool flag=game.state.rank_at[action.to]==Flag;
    const auto outcome=game.play(action,learner,rules);
    const bool captured=flag && outcome!=Outcome::DefenderWins;
    meter.flag_captures+=captured;
    if (audit) audit->after_play(d,action,own,outcome,captured,next_enemy,rules);
  }
  std::cout<<"G "<<seed<<' '<<learner<<' '<<table<<' '<<arm<<' '<<start.to_move+0<<' '
           <<state_hash(start)<<' '<<game.trace<<' '<<review_outcome(game.state.result,learner)<<' '
           <<game.state.ply<<' '<<danger<<' '<<in_list<<' '<<nominated<<' '<<selected<<' '<<exact_base_risk<<' '
           <<validated_selected;
  board[0].print();
  board[1].print();
  std::cout<<'\n';
  if (audit) audit->print();
}

struct HumanFlagReplies {
  FlagReplyTelemetry board[2];
  std::array<std::optional<OpeningMarks>,kSeats> history;
  int alternatives=0, matched=0, contrast=0, both=0, risk_roots=0;

  void advance(const State& before,const StepInfo& event,const State& after,const Rules& rules) {
    for (int seat=0; seat<kSeats; ++seat) {
      if (after.out[seat]) continue;
      const auto old=observe(before,seat);
      if (!history[seat]) history[seat].emplace(old,true);
      advance_contact_ranks(*history[seat],old,observe(after,seat),event.move,event.outcome,rules,
                            settled_flag_capture(old,event,rules));
    }
  }

  static bool near_flag(const Observation& view) {
    const auto known=friendly_flag_slots(view);
    const auto& board=Board::get();
    for (int node=0; node<kNodes; ++node) {
      if (!threatens_team_flag(view,node,known)) continue;
      if (board.node(node).rail) return true;
      for (auto origin:board.road(node))
        if (view.seat_at(origin)>=0 && view.team(view.seat_at(origin))!=view.team(view.viewer))
          return true;
    }
    return false;
  }

  void inspect(const State& state,Move human,const Rules& rules) {
    const auto d=seat_decision(state);
    auto& record=history[d.observation.viewer];
    if (!record) record.emplace(d.observation,true);
    const auto& evidence=*record;
    const auto claim=public_flag_reply(d.observation,evidence,human,rules);
    auto& meter=board[team_of(state.to_move)];
    meter.action(d,human,rules,claim);
    meter.flag_captures+=state.rank_at[human.to]==Flag &&
                         resolve(state.rank_at[human.from],Flag,rules)!=Outcome::DefenderWins;
    if (claim==FlagReply::Clear)
      single_case_check(!actual_flag_reply(state,human,rules),"strict replay disproved public flag claim");
    if (!near_flag(d.observation)) return;
    bool risk=false,clear=false;
    for (auto move:d.actions) {
      if (move==human) continue;
      const auto sibling=public_flag_reply(d.observation,evidence,move,rules);
      risk|=sibling==FlagReply::Risk;
      clear|=sibling==FlagReply::Clear;
      if (risk && clear) break;
    }
    alternatives+=risk || clear;
    both+=risk && clear;
    contrast+=risk && !clear && claim==FlagReply::Clear;
    matched+=risk && claim==FlagReply::Clear;
    risk_roots+=risk || claim==FlagReply::Risk;
  }
};

static void run_flag_reply_pressure(const char* path,const Rules& rules,const std::string& mode) {
  if (mode=="public-flag-reply-controls") return flag_reply_controls(rules);
  const bool attribute=mode=="public-flag-reply-attribute";
  if (mode=="public-flag-reply-games" || attribute) {
    StudentNetwork model;
    single_case_check(model.load(std::cin,true),"incompatible frozen student");
    uint64_t seed;
    int learner,table,count=0;
    auto stress=rules;
    stress.marshal_reveals_flag=true;
    std::cout<<"RULES "<<rules_hash(stress)<<'\n';
    while (std::cin>>seed>>learner>>table) {
      single_case_check(learner>=0 && learner<kSeats && table>=0 && table<2 && ++count<=30,
                        "invalid pressure assignment");
      const auto start=opening(seed,stress);
      if (attribute) {
        FlagAttribution audit{seed,table};
        pressure_flag_game(start,seed,learner,table,0,model,stress,&audit);
      } else {
        for (int arm:{0,1}) pressure_flag_game(start,seed,learner,table,arm,model,stress);
      }
    }
    single_case_check(std::cin.eof() && count>0,"incomplete pressure assignment");
    return;
  }
  std::string tag,key;
  int ccw,cap;
  const auto config=read_text(path);
  while (std::cin>>tag>>key>>ccw>>cap) {
    single_case_check((ccw==0 || ccw==1) && cap>=100 && cap<=140,"invalid archive validation prefix");
    HumanFlagReplies scan;
    auto local=rules;
    local.ccw=ccw!=0;
    const auto advance=[&](const State& before,const StepInfo& event,const State& after) {
      scan.advance(before,event,after,local);
    };
    RuleComparison result;
    if (tag=="H") result=scan_jqh_pressure(scan,local,cap,advance);
    else {
      Rules archive;
      std::string tape,error;
      single_case_check(tag=="G" && bool(std::cin>>tape) &&
                        parse_rules(archive_review_rules(config,local.ccw),archive,error),"invalid archive tape");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      result=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cap,advance);
    }
    std::cout<<"H "<<key<<' '<<cap<<' '<<result.agrees<<' '<<result.divergence<<' '<<result.reason<<' '
             <<scan.alternatives<<' '<<scan.matched<<' '<<scan.contrast<<' '<<scan.both<<' '<<scan.risk_roots;
    scan.board[0].print();
    scan.board[1].print();
    std::cout<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete archive comparison");
}
