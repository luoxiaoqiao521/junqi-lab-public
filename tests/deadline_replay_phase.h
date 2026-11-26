#pragma once
#include "replay_qh_wire.h"
#include "deadline_replay_witness.h"

static void replay_phase_root(EarlyPrefix& game,Move archived,const ExchangePolicy& policy,
                              const std::string& key,int phase,uint64_t seed,
                              const std::vector<State>* exact=nullptr) {
  const auto d=seat_decision(game.state);
  single_case_check(d.observation.viewer==policy.learner && is_legal(game.state,archived),"invalid matched archive root");
  const auto choice=finish_policy(game,d,policy);
  const auto evidence=game.reply_history.for_actor(d.observation,game.marks[policy.learner]);
  seed=mix_seed(seed,public_rollout_seed(d.observation,evidence,d.observation.quiet_clock));
  const auto worlds=exact?*exact:blind_extension_worlds(d,evidence,seed);
  const auto offered=std::find(choice.shortlist.begin(),choice.shortlist.end(),archived)!=choice.shortlist.end();
  std::cout<<"R "<<key<<' '<<phase<<' '<<game.state.ply+1<<' '<<d.actions.size()<<' '<<offered
           <<' '<<early_action_key(choice.base)<<' '<<early_action_key(archived)<<' '<<std::min(size_t{6},worlds.size())<<'\n';
  for (size_t i=0;i<std::min(size_t{6},worlds.size());++i) for (int desk:{0,1}) for (int arm:{0,1}) {
    const auto measure=[&](Move first) {
      ResponseAudit work{140};
      ResponseValue score;
      response_trial(worlds[i],first,evidence,game.reply_history,policy.rules,policy.history,
                     desk,mix_seed(seed,i+uint64_t(desk)*1103),score,false,RoutePriority::Old,true,&work);
      return std::pair{score.wins?2:score.draws?1:score.losses?0:-1,work.plies};
    };
    const auto first=arm?archived:choice.base;
    const auto [outcome,plies]=measure(first);
    if (i==0 && desk==0 && arm==0)
      single_case_check(measure(first)==std::pair{outcome,plies},"equal-action continuation changed");
    std::cout<<"F "<<key<<' '<<phase<<' '<<i<<' '<<desk<<' '<<arm<<' '<<outcome<<' '<<plies<<'\n';
  }
}

static int replay_phase_index(int ply,const std::array<int,2>& cuts) {
  return ply+1==cuts[0]?0:ply+1==cuts[1]?1:kNone;
}

static void replay_phase_qq(const ArchiveRows& raw,const ExchangePolicy& policy,const std::string& key,
                            const std::array<int,2>& cuts,uint64_t seed,bool exact) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),policy.rules);
  EarlyPrefix game(review.state,seed);
  const int requested=static_cast<int>(std::count_if(cuts.begin(),cuts.end(),[](int cut) {return cut>0;}));
  std::vector<Observation> history;
  if (exact) history.push_back(observe(review.state,policy.learner));
  int found=0;
  for (int i=0;i<raw.entries && found<requested;) {
    const auto row=raw.row(i++);
    const auto before=review.state;
    review.play(raw,row,i,policy.rules);
    const int phase=replay_phase_index(before.ply,cuts);
    if (phase!=kNone) {
      const auto rng=mix_seed(seed,phase);
      if (exact) {
        const auto d=seat_decision(game.state);
        const auto evidence=game.reply_history.for_actor(d.observation,game.marks[policy.learner]);
        const auto worlds=replay_prefix_witnesses(raw,policy.rules,history,d,key,phase,
                         public_rollout_seed(d.observation,evidence,d.observation.quiet_clock));
        replay_phase_root(game,review.last.move,policy,key,phase,rng,&worlds);
      } else replay_phase_root(game,review.last.move,policy,key,phase,rng);
      ++found;
    }
    while (i<raw.entries && raw.byte(raw.row(i))==245) {
      const auto control=raw.row(i++);
      review.control(raw,control,i,policy.rules);
    }
    review.frames.str("");
    archive_observe(game,review,before,policy.rules);
    if (exact) history.push_back(observe(review.state,policy.learner));
  }
  single_case_check(found==requested,"missing QQ phase");
}

static void replay_phase_qh(const ExchangePolicy& policy,const std::string& key,
                            const std::array<int,2>& cuts,uint64_t seed) {
  int8_t layouts[4][25]{};
  std::vector<std::array<int,4>> rows;
  single_case_check(read_qh(layouts,rows),"invalid JQH parameters");
  State start;
  init_state(start,layouts,policy.learner,policy.rules);
  EarlyPrefix game(start,seed);
  int found=0;
  for (const auto& row:rows) {
    const int from=point(row[0],row[1]),to=point(row[2],row[3]);
    single_case_check(from>=0 && to>=0,"unknown event in JQH study prefix");
    const Move action{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    single_case_check(is_legal(game.state,action),"JQH prefix diverged");
    const int phase=replay_phase_index(game.state.ply,cuts);
    if (phase!=kNone) {
      replay_phase_root(game,action,policy,key,phase,mix_seed(seed,phase));
      if (++found==2) break;
    }
    game.play(action,policy.learner,policy.rules);
  }
  single_case_check(found==2,"missing JQH phase");
}

static void run_replay_phases(const char* path,const Rules& frozen,const ShortWeights& parent,
                             const ShortWeights& peer,const StudentNetwork& student,bool exact) {
  std::string error;
  Rules historical,qh;
  single_case_check(parse_rules(archive_review_rules(read_text(path),false),historical,error),"bad QQ profile");
  // The intake verified this explicit experimental JQH profile. No assumption of dual visibility.
  qh.ccw=false;
  qh.no_progress_limit=0;
  const Rules cohort[]{historical,qh};
  std::string kind,key;
  int learner,ccw;
  std::array<int,2> cuts{};
  while (std::cin>>kind>>key>>learner>>ccw>>cuts[0]>>cuts[1]) {
    single_case_check((kind=="G" || kind=="H") && learner>=0 && learner<kSeats && ccw>=0 && ccw<=1 &&
                      (exact?(kind=="G" && cuts[0]>=0 && cuts[1]>=0 && (cuts[0] || cuts[1])):
                             (cuts[0]>0 && cuts[1]>0)) && cuts[0]!=cuts[1],"invalid replay phase row");
    auto rules=cohort[kind=="H"];
    rules.ccw=bool(ccw);
    rules.truncation_cap=frozen.truncation_cap;
    const ExchangePolicy policy{rules,parent,peer,learner,1,1,&student};
    constexpr uint64_t seed=806681;
    if (kind=="H") replay_phase_qh(policy,key,cuts,seed);
    else {
      std::string hex;
      single_case_check(bool(std::cin>>hex),"missing QQ tape");
      replay_phase_qq(ArchiveRows(decode_archive_tape(hex)),policy,key,cuts,seed,exact);
    }
  }
  single_case_check(std::cin.eof(),"incomplete replay phase row");
}
