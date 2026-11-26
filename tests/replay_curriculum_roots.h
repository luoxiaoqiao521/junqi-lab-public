#pragma once

struct CurriculumFollow {
  uint64_t trace=0;
  int opening_move=0, elapsed=0, outcome=-1, next=kNone, exits=0;
};

static void check_archive_marks(const State& state,const std::vector<OpeningMarks>& marks) {
  expect_value(marks.size()==kSeats,"missing per-seat history");
  for (int seat=0;seat<kSeats;++seat) {
    const auto view=observe(state,seat);
    for (int node=0;node<kNodes;++node) {
      if (view.seat_at(node)<0) continue;
      const auto mask=compatible_rank_mask(view,marks[seat],node);
      expect_value((mask & (1u<<state.rank_at[node]))!=0,"restored public evidence contradicts referee");
    }
  }
}

static CurriculumFollow follow_curriculum_root(const Root& root,const Rules& rules,const Policies& models,
                                              uint64_t seed,int table,bool human) {
  auto state=root.state;
  auto marks=root.all_history;
  auto streams=review_streams(seed);
  const int learner=state.to_move;
  CurriculumFollow result;
  result.trace=seed;
  do {
    const auto d=seat_decision(state);
    expect_value(!d.actions.empty(),"curriculum actor cannot move");
    auto choice=select(d,learner,table,0,models,streams[state.to_move]);
    if (!result.elapsed) {
      if (human) choice=root.human;
      result.opening_move=choice.from*kNodes+choice.to;
    }
    expect_value(std::find(d.actions.begin(),d.actions.end(),choice)!=d.actions.end(),"illegal branch choice");
    const auto before=state;
    const auto event=step(state,choice,rules);
    for (int seat=0;seat<kSeats;++seat)
      advance_contact_ranks(marks[seat],observe(before,seat),observe(state,seat),choice,event.outcome,rules);
    check_archive_marks(state,marks);
    result.exits|=event.eliminated_mask;
    ++result.elapsed;
    result.trace=mix_seed(result.trace,uint64_t(choice.from) | (uint64_t(choice.to)<<8) |
        (uint64_t(event.seat)<<16) | (uint64_t(event.outcome)<<18) | (uint64_t(event.eliminated_mask)<<22));
  } while (state.result==Result::Ongoing && !state.out[learner] &&
           state.to_move!=learner && result.elapsed<kSeats);
  result.outcome=review_outcome(state.result,learner);
  result.next=state.result==Result::Ongoing ? state.to_move : kNone;
  return result;
}

static void curriculum_root(const std::string& key,const Root& root,const Rules& rules,
                            const Policies& models,int human_result) {
  const auto d=seat_decision(root.state);
  const int learner=d.observation.viewer;
  check_archive_marks(root.state,root.all_history);
  const auto& history=root.all_history[learner];
  int moved=0, restricted=0;
  for (int node=0;node<kNodes;++node) {
    if (d.observation.seat_at(node)<0 || d.observation.rank_at(node)!=Unknown) continue;
    moved+=history.moved[node];
    restricted+=history.ranks[node]!=all_evidence_ranks;
  }
  std::array<CurriculumFollow,4> branches;
  const auto seed=mix_seed(std::stoull(key.substr(0,16),nullptr,16),root.state.ply);
  for (int table:{0,1}) for (int arm:{0,1})
    branches[2*table+arm]=follow_curriculum_root(root,rules,models,seed,table,arm==1);
  std::cout<<"K "<<key<<' '<<root.state.ply<<' '<<learner<<' '<<root.state.no_progress
           <<' '<<d.actions.size()<<' '<<root.human.from*kNodes+root.human.to<<' '
           <<moved<<' '<<restricted<<' '<<human_result<<'\n';
  for (int table:{0,1}) for (int arm:{0,1}) {
    const auto& r=branches[2*table+arm];
    std::cout<<"B "<<key<<' '<<root.state.ply<<' '<<table<<' '<<arm<<' '
             <<r.opening_move<<' '<<r.elapsed<<' '<<r.outcome<<' '<<r.next<<' '<<r.exits<<' '<<r.trace<<'\n';
  }
}
