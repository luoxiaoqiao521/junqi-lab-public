#pragma once

static int first_enemy_seat(const State& state,int learner,const Rules& rules) {
  for (int seat=next_seat(learner,rules);seat!=learner;seat=next_seat(seat,rules))
    if (!state.out[seat] && team_of(seat)!=team_of(learner)) return seat;
  return kNone;
}

static void replay_follow_case(const std::string& key,const Root& root,Rules rules,const Policies& models,
                               int cap,int table,int rep,bool human,bool alternate) {
  auto state=root.state;
  const int learner=state.to_move,initial_ply=state.ply;
  const int tested_enemy=table==2 ? first_enemy_seat(state,learner,rules) : kNone;
  const auto seed=mix_seed(std::stoull(key.substr(0,16),nullptr,16),uint64_t(8*initial_ply+rep));
  auto streams=review_streams(seed);
  rules.truncation_cap=std::min(rules.truncation_cap,initial_ply+cap);
  Timing timing;
  int first=-1, returned=0, normal=-1, second=-1, ending=0;
  int first_reply=-1, reply_seat=kNone;
  uint64_t trace=seed, before_return=0;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const int actor=d.observation.viewer;
    auto move=actor==tested_enemy ? review_move(d,learner,2,true,models.peer,streams[actor])
                                  : select(d,learner,table==0?0:1,0,models,streams[actor]);
    if (state.ply==initial_ply) {
      if (human) move=root.human;
      first=move.from*kNodes+move.to;
    } else if (actor==learner && !returned) {
      returned=state.ply-initial_ply;
      before_return=trace;
      normal=move.from*kNodes+move.to;
      if (alternate) {
        auto tie_stream=streams[actor];
        move=review_move(d,learner,0,false,models.parent,tie_stream);
      }
      second=move.from*kNodes+move.to;
    }
    expect_value(std::find(d.actions.begin(),d.actions.end(),move)!=d.actions.end(),"illegal replay follow-through");
    const auto event=step(state,move,rules);
    if (state.ply==initial_ply+2) {
      first_reply=move.from*kNodes+move.to;
      reply_seat=event.seat;
    }
    trace=mix_seed(trace,uint64_t(move.from) | (uint64_t(move.to)<<8) | (uint64_t(actor)<<16) |
        (uint64_t(event.outcome)<<18) | (uint64_t(event.eliminated_mask)<<22));
    timing.note(event,team_of(learner),state.ply-initial_ply);
    ending=int(event.end_reason);
  }
  std::cout<<"L "<<key<<' '<<initial_ply<<' '<<table<<' '<<rep<<' '<<human<<' '<<alternate<<' '
           <<first<<' '<<reply_seat<<' '<<first_reply<<' '<<returned<<' '<<normal<<' '<<second
           <<' '<<before_return<<' '<<trace<<' '<<state.ply-initial_ply<<' '
           <<review_outcome(state.result,learner)<<' '<<ending;
  timing.print();
  std::cout<<'\n';
}
