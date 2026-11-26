#pragma once

struct DirectedContact {
  Move move{};
  unsigned ranks=0;
  int kind=-1;
};

static DirectedContact directed_contact(const SeatDecision& decision,const OpeningMarks& evidence,
                                        const Rules& rules,int preferred) {
  const unsigned groups[]{1u<<Bomb,1u<<Engineer,(1u<<Marshal)|(1u<<Corps)};
  for (int i=0;i<3;++i) {
    const auto kind=(preferred+i)%3;
    for (const auto move:decision.actions) {
      const auto& view=decision.observation;
      if (view.seat_at(move.to)<0 || view.rank_at(move.to)!=Unknown) continue;
      const auto domain=compatible_rank_mask(view,evidence,move.to);
      const auto yes=collision_outcome_mask(view.rank_at(move.from),domain&groups[kind],rules);
      const auto no=collision_outcome_mask(view.rank_at(move.from),domain&~groups[kind],rules);
      if (yes && no && yes!=no) return {move,groups[kind],kind};
    }
  }
  return {};
}

static constexpr unsigned directed_node_cap=4096;
static constexpr int directed_trial_cap=48, directed_replay_cap=4;

static bool directed_budget(const CoverageTrial& trial) {
  return trial.nodes<directed_node_cap && trial.proposals<directed_trial_cap && trial.solved<directed_replay_cap;
}

static void directed_attempt(const CoverageRoot& root,uint64_t seed,bool old,int square,unsigned ranks,
                             CoverageTrial& trial) {
  const auto salt=mix_seed(seed,trial.proposals++);
  std::array<unsigned,kNodes> pins{};
  Rng rng(salt);
  if (old) {
    const auto proposal=public_sample(root.decision.observation,root.evidence,salt);
    if (!proposal) { ++trial.unavailable; return; }
    for (int n=0;n<kNodes;++n) if (proposal->rank_at[n]!=Unknown) pins[n]=1u<<proposal->rank_at[n];
  } else if (square>=0) pins[square]=ranks;
  coverage_attempt(root,pins,old?nullptr:&rng,std::min(384u,directed_node_cap-trial.nodes),trial);
}

static unsigned directed_sides(const CoverageTrial& trial,const DirectedContact& contact) {
  unsigned sides=0;
  if (contact.ranks) for (const auto& world:trial.worlds)
    sides|=(contact.ranks&(1u<<world.rank_at[contact.move.to]))?2u:1u;
  return sides;
}

struct DirectedPanel {
  CoverageTrial trial;
  int targeted=0;
  unsigned excluded=0;
};

static DirectedPanel run_directed_panel(const CoverageRoot& root,uint64_t seed,int arm,const DirectedContact& contact) {
  DirectedPanel panel;
  auto& trial=panel.trial;
  if (arm==2 && contact.ranks) {
    while (directed_budget(trial) && trial.proposals<12 && trial.solved<2)
      directed_attempt(root,seed,true,kNone,0,trial);
    for (unsigned side:{1u,2u}) {
      const auto ranks=side==2?contact.ranks:all_evidence_ranks&~contact.ranks;
      for (int i=0;i<4 && directed_budget(trial) && !(directed_sides(trial,contact)&side);++i) {
        ++panel.targeted;
        const int before=trial.contradicted;
        directed_attempt(root,seed,false,contact.move.to,ranks,trial);
        if (trial.contradicted>before) { panel.excluded|=side; break; }
      }
    }
  }
  while (directed_budget(trial)) {
    const bool old=arm==0 || (arm==2 && (!contact.ranks || !trial.worlds.empty()));
    directed_attempt(root,seed,old,kNone,0,trial);
  }
  return panel;
}

static unsigned directed_outcomes(const CoverageRoot& root,const CoverageTrial& trial,const DirectedContact& contact) {
  unsigned outcomes=0;
  if (contact.ranks) for (const auto& world:trial.worlds)
    outcomes|=outcome_bit(resolve(root.decision.observation.rank_at(contact.move.from),
                                  world.rank_at[contact.move.to],root.rules));
  return outcomes;
}

static void assess_directed_coverage(const ArchiveRows& raw,const Rules& rules,const std::string& key,
    int viewer,int cut,uint64_t seed,int preferred,const StudentNetwork& student) {
  using Clock=std::chrono::steady_clock;
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto decision=seat_decision(prefix.game.state);
  const auto evidence=prefix.game.reply_history.for_actor(decision.observation,prefix.game.marks[viewer]);
  const CoverageRoot root{raw,rules,prefix,decision,evidence};
  const auto contact=directed_contact(decision,evidence,rules,preferred);
  std::cout<<"T "<<key<<' '<<cut<<' '<<contact.kind<<' '<<(contact.ranks?early_action_key(contact.move):kNone)<<'\n';
  coverage_controls(root,key,seed,student);
  seed=mix_seed(seed,public_rollout_seed(decision.observation,evidence,decision.observation.quiet_clock));
  for (int i=0;i<3;++i) {
    const int arm=(i+preferred)%3;
    const auto start=Clock::now();
    const auto result=run_directed_panel(root,seed,arm,contact);
    const auto us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
    const auto& p=result.trial;
    std::cout<<"P "<<key<<' '<<arm<<' '<<p.proposals<<' '<<p.nodes<<' '<<p.solved<<' '<<p.replay_plies
             <<' '<<p.worlds.size()<<' '<<p.replay_failed<<' '<<p.contradicted<<' '<<p.unresolved<<' '<<p.unavailable
             <<' '<<directed_sides(p,contact)<<' '<<directed_outcomes(root,p,contact)<<' '<<result.targeted
             <<' '<<result.excluded<<' '<<us<<'\n';
  }
}

static void run_directed_coverage(const char* rules_path) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid directed coverage student");
  std::string key,hex,error;
  int viewer,cut,ccw,preferred;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>ccw>>seed>>preferred>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && preferred>=0 && preferred<3 && (ccw==0 || ccw==1) &&
        parse_rules(archive_review_rules(read_text(rules_path),bool(ccw)),rules,error),"invalid directed coverage wire");
    assess_directed_coverage(ArchiveRows(decode_archive_tape(hex)),rules,key,viewer,cut,seed,preferred,student);
  }
  single_case_check(std::cin.eof(),"incomplete directed coverage wire");
}
