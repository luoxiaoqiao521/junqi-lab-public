#pragma once

#include "../sim/archive_review.h"

struct ExchangePolicy {
  const Rules& rules;
  const ShortWeights& parent;
  const ShortWeights& history;
  int learner,partner,table;
  const StudentNetwork* student=nullptr;
  bool recovery_only=false;
};

struct ExchangeTime {
  int decisions=0,legal_contact=0,selected_contact=0;
  int max_quiet=0,last_collision=0,reverse=0,last_reverse=0;
  void note(const EarlyPrefix& game,Move action,Outcome outcome,Move previous) {
    max_quiet=std::max(max_quiet,game.state.no_progress);
    if (outcome!=Outcome::Move) last_collision=game.state.ply;
    if (action.from==previous.to && action.to==previous.from && outcome==Outcome::Move &&
        previous.from!=previous.to) {++reverse; last_reverse=game.state.ply;}
  }
};

struct ExchangeChoice { Move action; int kind=0, samples=0; };

static ExchangeChoice exchange_alternative(const SeatDecision& d,const OpeningMarks& marks,
                                            const PublicReplyHistory& prefix,Move base,
                                            const Rules& rules,const ShortWeights& history) {
  if (const auto finish=public_flag_finish(d,base,rules)) return {*finish,1,0};
  const auto threat=visible_flag_threat(d.observation,marks,rules);
  if (const auto guard=visible_intercept(d,marks,rules,base,threat)) return {*guard,2,0};
  ResponseWork work;
  const auto move=response_choice(d,marks,prefix,base,rules,history,d.observation.quiet_clock,work,
                                  true,RoutePriority::Joint,false,false,true);
  return {move,work.samples?3:4,work.samples};
}

static Move exchange_parent(EarlyPrefix& game,const SeatDecision& d,const ExchangePolicy& policy) {
  auto action=early_normal(game,d,policy.learner,policy.table,policy.rules,policy.parent,policy.history,policy.partner);
  const int ply=game.state.ply;
  if (!policy.student || d.observation.viewer!=policy.learner || ply<24 || ply>=100) return action;
  auto& mark=game.marks[policy.learner];
  const bool first=game.student_window[ply>=56];
  if (first) return action;
  game.student_window[ply>=56]=true;
  const auto acts=response_candidates(d,mark,action,policy.rules,
                                     public_rollout_seed(d.observation,mark,d.observation.quiet_clock));
  return student_pick(d,mark,action,acts,*policy.student,policy.rules);
}

static Outcome exchange_step(EarlyPrefix& game,Move action,const ExchangePolicy& policy,ExchangeTime& clock) {
  const int seat=game.state.to_move;
  const auto previous=game.reply_history.last_moves()[seat];
  const auto outcome=game.play(action,policy.learner,policy.rules);
  clock.note(game,action,outcome,previous);
  return outcome;
}

static Move exchange_decide(EarlyPrefix& game,const SeatDecision& d,const ExchangePolicy& policy,ExchangeTime& clock) {
  const auto action=exchange_parent(game,d,policy);
  if (game.state.to_move==policy.learner) {
    ++clock.decisions;
    const auto& v=d.observation;
    clock.legal_contact+=std::any_of(d.actions.begin(),d.actions.end(),[&](Move m){return v.seat_at(m.to)>=0;});
    clock.selected_contact+=v.seat_at(action.to)>=0;
  }
  return action;
}

static void print_exchange_end(const EarlyPrefix& game,const ExchangePolicy& policy,const ExchangeTime& clock) {
  std::cout<<' '<<game.state.ply<<' '<<review_outcome(game.state.result,policy.learner)<<' '
           <<int(game.ending)<<' '<<game.trace<<' '<<clock.decisions<<' '<<clock.legal_contact<<' '
           <<clock.selected_contact<<' '<<clock.max_quiet<<' '<<clock.last_collision<<' '<<clock.reverse
           <<' '<<clock.last_reverse<<' '<<game.friendly_exit<<' '<<game.enemy_exit<<' '
           <<game.friendly_reason<<' '<<game.enemy_reason<<'\n';
}

static void exchange_suffix(EarlyPrefix game,Move action,const ExchangePolicy& policy,ExchangeTime clock) {
  // The parent action has consumed its normal RNG draw on the shared prefix.
  exchange_step(game,action,policy,clock);
  while (game.state.result==Result::Ongoing) {
    const auto decision=seat_decision(game.state);
    const auto move=exchange_decide(game,decision,policy,clock);
    exchange_step(game,move,policy,clock);
  }
  print_exchange_end(game,policy,clock);
}

struct ExchangeContact {
  int ply=0,outcome=0,id=kNone,target=kNone,replies=0,first=kNone,enemy_contacts=0;
  bool ally_contact=false;
  void note(const State& state,Move action,int learner,Outcome result) {
    const int actor=state.to_move;
    if (ply) {
      if (first==kNone) first=actor;
      replies|=1<<actor;
      if (actor==(learner+2)%kSeats) ally_contact|=result!=Outcome::Move;
      if (team_of(actor)!=team_of(learner)) enemy_contacts+=result!=Outcome::Move;
    } else if (state.ply>=24 && actor==learner && result!=Outcome::Move) {
      ply=state.ply+1;
      outcome=int(result);
      id=state.id_at[action.from];
      target=action.to;
    }
  }
};

static void exchange_run(const std::string& key,EarlyPrefix game,const ExchangePolicy& policy,
                         ExchangeTime clock,bool archive,const ExchangeContact& contact) {
  const auto decision=seat_decision(game.state);
  const auto before=game.trace;
  const auto base=exchange_decide(game,decision,policy,clock);
  const auto& marks=game.marks[policy.learner];
  const bool survives=decision.observation.seat_at(contact.target)==policy.learner;
  expect_value(survives==game.state.alive[contact.id],"public attacker tracking diverged from referee");
  ExchangeChoice choice{base};
  if (policy.recovery_only && survives) choice.kind=5;
  else if (decision.actions.size()>1)
    choice=exchange_alternative(decision,marks,game.reply_history,base,policy.rules,policy.history);
  expect_value(std::find(decision.actions.begin(),decision.actions.end(),choice.action)!=decision.actions.end(),
               "one-step alternative not legal");
  std::cout<<"P "<<key<<' '<<contact.ply<<' '<<contact.outcome<<' '<<game.state.ply+1<<' '
           <<decision.actions.size()<<' '<<early_action_key(base)<<' '<<early_action_key(choice.action)<<' '
           <<choice.kind<<' '<<choice.samples<<' '<<before<<' '<<contact.replies<<' '
           <<survives<<' '<<contact.first<<' '<<contact.ally_contact<<' '<<contact.enemy_contacts<<'\n';
  for (int rep=0;rep<(archive?8:1);++rep) {
    auto root=game;
    if (archive) root.streams=review_streams(mix_seed(game.trace,8701+rep));
    for (const int arm:{0,1}) {
      auto branch=root;
      const auto action=arm?choice.action:base;
      // Identical action in a fork must generate an identical continuation.
      clock.selected_contact+=int(decision.observation.seat_at(action.to)>=0)-
                              int(decision.observation.seat_at(base.to)>=0);
      std::cout<<"F "<<key<<' '<<rep<<' '<<arm;
      exchange_suffix(branch,action,policy,clock);
      clock.selected_contact+=int(decision.observation.seat_at(base.to)>=0)-
                              int(decision.observation.seat_at(action.to)>=0);
    }
  }
}

static std::string exchange_key(uint64_t seed,int learner,int partner,int table) {
  return std::to_string(seed)+"/"+std::to_string(learner)+"/"+std::to_string(partner)+"/"+std::to_string(table);
}

static void natural_exchange(uint64_t seed,const ExchangePolicy& policy) {
  EarlyPrefix game(opening(seed,policy.rules),seed);
  std::cout<<"N "<<exchange_key(seed,policy.learner,policy.partner,policy.table)
           <<' '<<int(game.state.to_move)<<'\n';
  const auto key=exchange_key(seed,policy.learner,policy.partner,policy.table);
  ExchangeTime clock;
  ExchangeContact contact;
  while (game.state.result==Result::Ongoing) {
    if (contact.ply && game.state.to_move==policy.learner) {
      exchange_run(key,game,policy,clock,false,contact);
      return;
    }
    const auto before=game.state;
    const auto decision=seat_decision(game.state);
    const auto action=exchange_decide(game,decision,policy,clock);
    const auto outcome=exchange_step(game,action,policy,clock);
    contact.note(before,action,policy.learner,outcome);
  }
  std::cout<<"X "<<key<<' '<<contact.ply<<' '<<game.state.out[policy.learner]<<' '<<contact.outcome<<'\n';
  for (const int arm:{0,1}) {
    std::cout<<"F "<<key<<" 0 "<<arm;
    print_exchange_end(game,policy,clock);
  }
}

static void archive_observe(EarlyPrefix& game,const ArchiveReview& review,const State& before,
                             const Rules& rules) {
  const auto event=review.last;
  game.reply_history.advance(before,review.state,event,rules);
  game.state=review.state;
  game.trace=mix_seed(game.trace,uint64_t(early_action_key(event.move))+uint64_t(int(event.outcome))*32768u);
  for (int seat=0;seat<kSeats;++seat)
    if (!game.state.out[seat])
      advance_contact_ranks(game.marks[seat],observe(before,seat),observe(game.state,seat),
                            event.move,event.outcome,rules);
}

static void archived_exchange(const std::string& key,const ArchiveRows& raw,const Rules& rules,
                              const ShortWeights& parent,const ShortWeights& history) {
  ArchiveReview review;
  const auto seed=std::stoull(key.substr(0,16),nullptr,16);
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  EarlyPrefix game(review.state,seed);
  ExchangeContact contact;
  int learner=kNone,index=0;
  std::cout<<"N "<<key<<' '<<int(game.state.to_move)<<'\n';
  while (index<raw.entries && raw.byte(raw.row(index))==245) {
    review.control(raw,raw.row(index),index+1,rules); ++index;
    game.reply_history.after_control(review.state);
    game.state=review.state;
  }
  while (index<raw.entries && !review.ended) {
    if (contact.ply && review.state.to_move==learner) {
      const ExchangePolicy policy{rules,parent,history,learner,1+int(seed%2),int((seed>>1)%2)};
      exchange_run(key,game,policy,{},true,contact);
      return;
    }
    const auto before=review.state;
    const auto row=raw.row(index++);
    expect_value(raw.byte(row)==95,"invalid archive prefix row");
    review.play(raw,row,index,rules);
    review.frames.str("");
    while (index<raw.entries && raw.byte(raw.row(index))==245) {
      const auto notice=raw.row(index);
      ++index;
      review.control(raw,notice,index,rules);
    }
    if (learner==kNone && before.ply>=24 && review.last.outcome!=Outcome::Move) learner=before.to_move;
    contact.note(before,review.last.move,learner,review.last.outcome);
    archive_observe(game,review,before,rules);
  }
  std::cout<<"X "<<key<<' '<<contact.ply<<' '<<(learner>=0 && game.state.out[learner])
           <<' '<<contact.outcome<<'\n';
}
