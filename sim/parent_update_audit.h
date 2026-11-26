#pragma once
#include "credit_window.h"
#include "flag_interception.h"
#include "transfer_probe.h"

namespace jq {
constexpr int kAuditConfirmationPairs=24;
struct ParentAudit {
  Rules rules;
  uint64_t seed;
  int learner, anchor;
  std::array<FrozenSeat,4> seats;
  ShortWeights child;
  int confirm_choice=0;
  int witness=-1;
  int other_hq=-1;
};

struct AuditReplyTrace {
  StepInfo first{};
  std::optional<StepInfo> next;
  std::optional<Move> eligible;
};

struct AuditPoint {
  State state;
  std::vector<Move> history;
  Move original, alternate;
};

struct AuditTrace {
  int score, plies, decisions;
  std::optional<AuditPoint> point;
};

inline AuditTrace audit_trace(State state,const ParentAudit& settings,bool child) {
  auto streams=review_streams(settings.seed);
  std::vector<Move> history;
  std::optional<AuditPoint> point;
  int decisions=0;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto seat=state.to_move;
    auto& rng=streams[seat];
    const auto copy=rng;
    const auto action=seat==settings.learner && child
      ? short_move(d.observation,d.actions,settings.child,rng)
      : frozen_seat_move(d,settings.seats[seat],rng);
    if (seat==settings.learner) {
      ++decisions;
      if (!child && !point) {
        auto other_rng=copy;
        const auto alternate=short_move(d.observation,d.actions,settings.child,other_rng);
        double variation=0;
        if (settings.anchor==0 && !(action==alternate)) {
          const auto old=review_learned(d,settings.seats[seat].weights);
          const auto proposed=review_learned(d,settings.child);
          for (size_t i=0;i<old.size();++i) variation+=std::abs(old[i]-proposed[i])/2;
        }
        if ((settings.anchor==0 && variation>=.01) || decisions==settings.anchor)
          point=AuditPoint{state,history,action,alternate};
      }
    }
    if (!point && !child) history.push_back(action);
    step(state,action,settings.rules);
  }
  return {review_outcome(state.result,settings.learner),state.ply,decisions,std::move(point)};
}

inline std::optional<Move> audit_pressure_reply(const SeatDecision& d,bool extended) {
  const auto attack=public_hq_attack(d);
  if (attack || !extended) return attack;
  for (const auto action : d.actions) {
    const int owner=d.observation.seat_at(action.to);
    if (owner>=0 && team_of(owner)!=team_of(d.observation.viewer)) return action;
  }
  return std::nullopt;
}

inline int audit_rollout(State state,Move first,const ParentAudit& settings,
                         uint64_t seed,bool child,bool pressure,int* replacements=nullptr,
                         AuditReplyTrace* trace=nullptr) {
  auto streams=review_streams(seed);
  const auto next_enemy=(settings.learner+(settings.rules.ccw ? 1 : 3))%4;
  const auto first_event=step(state,first,settings.rules);
  if (trace) trace->first=first_event;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto seat=state.to_move;
    auto& rng=streams[seat];
    auto move=seat==settings.learner && child
      ? short_move(d.observation,d.actions,settings.child,rng)
      : frozen_seat_move(d,settings.seats[seat],rng);
    if (seat==settings.learner) pressure=false;
    const auto attack=seat==next_enemy && (pressure || (trace && !trace->next))
      ? audit_pressure_reply(d,settings.confirm_choice!=0) : std::nullopt;
    if (trace && !trace->next) trace->eligible=attack;
    if (seat==next_enemy && pressure) {
      if (attack && !(move==*attack)) {
        move=*attack;
        if (replacements) ++*replacements;
      }
      pressure=false;
    }
    const auto event=step(state,move,settings.rules);
    if (trace && !trace->next) trace->next=event;
  }
  return review_outcome(state.result,settings.learner);
}

inline std::vector<size_t> audit_candidates(const SeatDecision& d,const AuditPoint& point,
                                            const std::vector<double>& probabilities) {
  const auto original=std::find(d.actions.begin(),d.actions.end(),point.original)-d.actions.begin();
  std::vector<size_t> choices{size_t(original)};
  const auto add=[&](Move move) {
    const auto found=std::find(d.actions.begin(),d.actions.end(),move);
    if (found==d.actions.end()) return;
    const auto i=static_cast<size_t>(found-d.actions.begin());
    if (std::find(choices.begin(),choices.end(),i)==choices.end()) choices.push_back(i);
  };
  add(point.alternate);
  std::vector<size_t> by_policy(d.actions.size());
  for (size_t i=0;i<by_policy.size();++i) by_policy[i]=i;
  std::stable_sort(by_policy.begin(),by_policy.end(),[&](size_t a,size_t b) {
    return probabilities[a]>probabilities[b];
  });
  if (choices.size()==1 && d.actions.size()>1)
    add(d.actions[by_policy[0]==size_t(original) ? by_policy[1] : by_policy[0]]);
  for (bool collision : {true,false}) {
    for (size_t i : by_policy) {
      if ((d.observation.seat_at(d.actions[i].to)!=kNone)==collision &&
          std::find(choices.begin(),choices.end(),i)==choices.end()) {
        add(d.actions[i]); break;
      }
    }
  }
  for (size_t i=0;i<d.actions.size() && choices.size()<5;++i) add(d.actions[i]);
  return choices;
}

inline void print_audit_pair(int original,int alternate) {
  std::cout << '[' << original << ',' << alternate << ']';
}

inline void audit_confirmation(const ParentAudit& settings,Move original,Move candidate,
                               const std::vector<State>& worlds) {
  bool comma=false;
  std::vector<std::array<int,2>> replacements;
  for (const auto& world : worlds)
    for (bool child : {false,true})
      for (bool pressure : {false,true}) {
        std::cout << (comma ? ",[" : "["); comma=true;
        std::array<int,2> changes{};
        const uint64_t offset=settings.confirm_choice ? 5000 : 2000;
        const uint64_t count=settings.confirm_choice ? kAuditConfirmationPairs : 3;
        for (uint64_t k=0;k<count;++k) {
          const auto seed=mix_seed(settings.seed,offset+k);
          const auto a=audit_rollout(world,original,settings,seed,child,pressure,&changes[0]);
          const auto b=audit_rollout(world,candidate,settings,seed,child,pressure,&changes[1]);
          if (k) std::cout << ',';
          print_audit_pair(a,b);
        }
        std::cout << ']';
        replacements.push_back(changes);
      }
  std::cout << ']';
  if (settings.confirm_choice) {
    std::cout << ",\"reply_edits\":[";
    for (size_t i=0;i<replacements.size();++i) {
      if (i) std::cout << ',';
      print_audit_pair(replacements[i][0],replacements[i][1]);
    }
    std::cout << ']';
  }
}

inline void audit_method(const ParentAudit& settings,const SeatDecision& d,
                         const std::vector<State>& worlds,const std::vector<size_t>& choices,size_t tested) {
  const auto started=std::chrono::steady_clock::now();
  const auto original=d.actions[choices[0]];
  std::cout << "{\"screen\":[";
  size_t winner=1;
  int best=-100;
  for (size_t i=1;i<=tested;++i) {
    int net=0;
    std::cout << (i==1 ? "[" : ",[");
    for (uint64_t k=0;k<12/tested;++k) {
      const auto seed=mix_seed(settings.seed,1000+k);
      const auto a=audit_rollout(worlds[0],original,settings,seed,false,false);
      const auto b=audit_rollout(worlds[0],d.actions[choices[i]],settings,seed,false,false);
      if (a>=0 && b>=0) net+=b-a;
      if (a<0 || b<0) net-=2;
      if (k) std::cout << ',';
      print_audit_pair(a,b);
    }
    if (net>best) { winner=i; best=net; }
    std::cout << ']';
  }
  std::cout << "],\"picked\":" << winner << ",\"confirm\":[";
  audit_confirmation(settings,original,d.actions[choices[winner]],worlds);
  const auto elapsed=std::chrono::steady_clock::now()-started;
  std::cout << ",\"us\":" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << '}';
}

inline void print_exercise_event(const StepInfo& event) {
  std::cout << '[' << int(event.seat) << ',' << int(event.move.from) << ',' << int(event.move.to)
            << ',' << int(event.outcome) << ',' << int(event.eliminated_mask)
            << ',' << int(event.revealed_mask) << ']';
}

inline void print_exercise_question(State initial,const AuditPoint& point,const ParentAudit& settings,
                                    const std::array<Move,2>& moves) {
  const auto view=observe(point.state,settings.learner);
  const auto& board=Board::get();
  std::cout << "{\"seat\":" << settings.learner << ",\"ply\":" << view.ply << ",\"choices\":[";
  print_audit_pair(moves[0].from,moves[0].to);
  std::cout << ',';
  print_audit_pair(moves[1].from,moves[1].to);
  std::cout << "],\"nodes\":[";
  for (int id=0;id<kNodes;++id) {
    const auto node=board.node(id);
    if (id) std::cout << ',';
    std::cout << '[' << int(node.gr) << ',' << int(node.gc) << ',' << int(node.kind) << ','
              << view.seat_at(id) << ',' << view.rank_at(id) << ']';
  }
  std::cout << "],\"edges\":[";
  bool comma=false;
  for (int a=0;a<kNodes;++a) {
    const auto& rail=board.rail_adj(a);
    for (int b : board.road(a)) {
      if (b<=a) continue;
      if (comma) std::cout << ',';
      comma=true;
      std::cout << '[' << a << ',' << b << ',' << (std::find(rail.begin(),rail.end(),b)!=rail.end()) << ']';
    }
  }
  std::cout << "],\"prefix\":[";
  for (size_t i=0;i<point.history.size();++i) {
    if (i) std::cout << ',';
    print_exercise_event(step(initial,point.history[i],settings.rules));
  }
  expect_value(state_hash(initial)==state_hash(point.state),"exercise public prefix is not exact");
  std::cout << "]}";
}

inline void print_exercise_witness(const State& initial,const AuditPoint& point,
                                   const ParentAudit& settings,const std::vector<State>& worlds,
                                   const std::array<Move,2>& moves) {
  const auto panel=settings.witness/kAuditConfirmationPairs;
  expect_value(size_t(panel/4)<worlds.size(),"exercise compatible world is missing");
  std::cout << "\"exercise\":{\"question\":";
  print_exercise_question(initial,point,settings,moves);
  std::cout << ",\"witness\":[";
  for (size_t i=0;i<moves.size();++i) {
    AuditReplyTrace trace;
    int edits=0;
    const auto seed=mix_seed(settings.seed,5000+settings.witness%kAuditConfirmationPairs);
    const auto score=audit_rollout(worlds[panel/4],moves[i],settings,seed,(panel%4)>=2,
                                   (panel%2)==1,&edits,&trace);
    if (i) std::cout << ',';
    std::cout << "{\"score\":" << score << ",\"edited\":" << edits << ",\"first\":";
    print_exercise_event(trace.first);
    std::cout << ",\"next\":";
    if (trace.next) print_exercise_event(*trace.next);
    else std::cout << "null";
    std::cout << ",\"eligible\":";
    if (trace.eligible) print_audit_pair(trace.eligible->from,trace.eligible->to);
    else std::cout << "null";
    std::cout << '}';
  }
  std::cout << "]}";
}

inline void audit_target_hq(const State& initial,const AuditPoint& point,const ParentAudit& settings,
                            const SeatDecision& d,const std::array<Move,2>& moves) {
  const auto target=moves[1].to;
  const auto other=settings.other_hq;
  const auto owner=d.observation.seat_at(target);
  const auto& board=Board::get();
  expect_value(target!=other && owner>=0 && team_of(owner)!=team_of(settings.learner) &&
               d.observation.seat_at(other)==owner &&
               d.observation.rank_at(target)==Unknown && d.observation.rank_at(other)==Unknown &&
               board.node(target).kind==NodeKind::HQ && board.node(other).kind==NodeKind::HQ,
               "expected two occupied enemy HQs hidden from the learner");
  const auto& state=point.state;
  const auto id_a=state.id_at[target];
  const auto id_b=state.id_at[other];
  expect_value((state.rank_at[target]==Flag)!=(state.rank_at[other]==Flag),
               "HQ test did not include an alternative flag placement");
  std::vector<State> worlds{state};
  State alternative;
  auto status="illegal_layout";
  if (swap_in_layout(initial,alternative,owner,id_a%25,id_b%25)) {
    status="changes_public_prefix";
    if (replay_alternative(initial,alternative,settings.rules,point.history,settings.learner)) {
      const auto alt=seat_decision(alternative);
      expect_value((state.rank_at[target]==Flag)!=(alternative.rank_at[target]==Flag) &&
                   alt.actions==d.actions &&
                   review_learned(alt,settings.seats[settings.learner].weights)==
                   review_learned(d,settings.seats[settings.learner].weights) &&
                   review_learned(alt,settings.child)==review_learned(d,settings.child),
                   "compatible flag change affected legal actions or decision probabilities");
      worlds.push_back(alternative);
      status="compatible";
    }
  }
  std::cout << "\"targets\":[" << int(target) << ',' << other << "],\"swap\":\"" << status
            << "\",\"first_events\":[";
  for (size_t i=0;i<worlds.size();++i) {
    if (i) std::cout << ',';
    std::cout << '[';
    for (size_t j=0;j<moves.size();++j) {
      if (j) std::cout << ',';
      State trial=worlds[i];
      print_exercise_event(step(trial,moves[j],settings.rules));
    }
    std::cout << ']';
  }
  std::cout << "],\"confirm\":[";
  audit_confirmation(settings,moves[0],moves[1],worlds);
}

inline void audit_point(const State& initial,const AuditPoint& point,const ParentAudit& settings) {
  const auto d=seat_decision(point.state);
  const auto old=review_learned(d,settings.seats[settings.learner].weights);
  const auto child=review_learned(d,settings.child);
  const auto choices=audit_candidates(d,point,child);
  if (choices.size()<2) { std::cout << "null"; return; }
  for (size_t i : {choices[0],choices[1]}) {
    State trial=point.state, undo;
    StepInfo event;
    expect_value(apply_checked(trial,d.actions[i],settings.rules,undo,event),"audit move is illegal");
    restore(trial,undo);
    const auto restored=seat_decision(trial);
    expect_value(state_hash(trial)==state_hash(point.state) && restored.actions==d.actions &&
                 review_learned(restored,settings.seats[settings.learner].weights)==old &&
                 review_learned(restored,settings.child)==child,"future changed the restored decision");
  }
  const auto worlds=credit_worlds(initial,point.state,d,settings.rules,point.history,
                                  settings.seats[settings.learner].weights,true);
  for (const auto& world : worlds)
    expect_value(review_learned(seat_decision(world),settings.child)==child,"child saw hidden identity");
  std::cout << "{\"ply\":" << point.state.ply << ",\"legal\":" << d.actions.size()
            << ",\"same_sample\":" << (point.original==point.alternate)
            << ",\"worlds\":" << worlds.size() << ",\"choices\":[";
  for (size_t k=0;k<choices.size();++k) {
    if (k) std::cout << ',';
    const auto i=choices[k];
    const auto move=d.actions[i];
    const auto features=short_features(d.observation,move);
    std::cout << '[' << int(move.from) << ',' << int(move.to) << ',' << std::setprecision(17)
              << old[i] << ',' << child[i] << ",[" << int(features[0]) << ',' << int(features[1])
              << ',' << int(features[2]) << ',' << d.observation.rank_at(move.from) << ','
              << int(features[4]) << "]]";
  }
  std::cout << ']';
  if (settings.confirm_choice) {
    const size_t choice=settings.confirm_choice;
    expect_value(choice<choices.size(),"fixed audit choice is missing");
    std::cout << ",\"picked\":" << choice << ',';
    const std::array<Move,2> moves{d.actions[choices[0]],d.actions[choices[choice]]};
    if (settings.other_hq>=0) audit_target_hq(initial,point,settings,d,moves);
    else if (settings.witness>=0) print_exercise_witness(initial,point,settings,worlds,moves);
    else {
      std::cout << "\"confirm\":[";
      audit_confirmation(settings,moves[0],moves[1],worlds);
    }
  } else {
    std::cout << ",\"A\":";
    audit_method(settings,d,worlds,choices,1);
    std::cout << ",\"B\":";
    audit_method(settings,d,worlds,choices,choices.size()-1);
  }
  std::cout << '}';
}

inline void parent_update_audit(ParentAudit settings) {
  Rng layout_rng(settings.seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,layout_rng);
  const int first=settings.rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  State initial;
  init_state(initial,layouts,first,settings.rules);
  const auto parent=audit_trace(initial,settings,false);
  const auto child=audit_trace(initial,settings,true);
  const auto print=[](const AuditTrace& trace) {
    std::cout << '[' << trace.score << ',' << trace.plies << ',' << trace.decisions << ']';
  };
  std::cout << "{\"first\":" << first << ",\"parent\":";
  print(parent);
  std::cout << ",\"child\":";
  print(child);
  std::cout << ",\"point\":";
  if (parent.point) audit_point(initial,*parent.point,settings);
  else std::cout << "null";
  std::cout << "}\n";
}
}
