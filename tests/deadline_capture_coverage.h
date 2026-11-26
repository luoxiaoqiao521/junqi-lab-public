#pragma once
#include <chrono>

struct CaptureCoveragePrefix {
  EarlyPrefix game;
  std::vector<Observation> history;
};

static CaptureCoveragePrefix coverage_prefix(const ArchiveRows& raw,const int8_t (&layouts)[kSeats][25],
    const Rules& rules,int viewer,int cut) {
  ArchiveReview replay;
  init_state(replay.state,layouts,archive_first_seat(raw),rules);
  CaptureCoveragePrefix prefix{EarlyPrefix(replay.state,0),{observe(replay.state,viewer)}};
  prefix.game.reply_history=PublicReplyHistory(replay.state,true,true);
  for (int i=0;replay.state.ply<cut && i<raw.entries;) {
    const auto row=raw.row(i++);
    const auto before=replay.state;
    replay.play(raw,row,i,rules);
    archive_observe(prefix.game,replay,before,rules);
    while (i<raw.entries && raw.byte(raw.row(i))==245) {
      const auto control=raw.row(i++);
      replay.control(raw,control,i,rules);
      prefix.game.reply_history.after_control(replay.state);
      prefix.game.state=replay.state;
    }
    if (rules.flag_eliminates && replay.captured_flag==kNone)
      prefix.game.reply_history.record_flag_resolution(replay.last);
    replay.frames.str("");
    prefix.history.push_back(observe(replay.state,viewer));
  }
  single_case_check(replay.state.ply==cut && replay.state.to_move==viewer,"capture coverage cut did not restore");
  return prefix;
}

static std::optional<State> verify_coverage_layout(const ArchiveRows& raw,const int8_t (&layouts)[kSeats][25],
    const Rules& rules,const CaptureCoveragePrefix& prefix,const SeatDecision& decision,PrefixReplayWork& work) {
  for (const auto& line:layouts) if (!valid_layout(line)) {
    work.failed_ply=0; work.reason="deployment"; return std::nullopt;
  }
  auto world=replay_prefix_world(raw,layouts,rules,prefix.history,&work);
  if (world && seat_decision(*world).actions!=decision.actions) {
    work.failed_ply=world->ply;
    work.reason="legal-set";
    return std::nullopt;
  }
  return world;
}

struct CoverageTrial {
  unsigned nodes=0;
  int proposals=0, unavailable=0, contradicted=0, unresolved=0, collision_conflicts=0, inventory_conflicts=0;
  int solved=0, replay_failed=0, replay_plies=0, first_conflict=-1, first_replay_failure=-1;
  std::vector<State> worlds;
};

struct CoverageRoot {
  const ArchiveRows& raw;
  const Rules& rules;
  const CaptureCoveragePrefix& prefix;
  const SeatDecision& decision;
  const OpeningMarks& evidence;

  CaptureCompatibility solve(const std::array<unsigned,kNodes>& pins,CaptureSearch& search) const {
    const auto& game=prefix.game;
    const auto& view=decision.observation;
    return game.reply_history.solve_candidate(view,game.marks[view.viewer],prefix.history.front(),pins,search);
  }

  std::optional<State> verify(const std::array<int8_t,kNodes>& opening,PrefixReplayWork& work) const {
    int8_t layouts[kSeats][25];
    for (int seat=0;seat<kSeats;++seat) for (int slot=0;slot<25;++slot)
      layouts[seat][slot]=opening[layout_slot_node(seat,slot)];
    return verify_coverage_layout(raw,layouts,rules,prefix,decision,work);
  }
};

static void coverage_attempt(const CoverageRoot& root,const std::array<unsigned,kNodes>& pins,
    Rng* rng,unsigned limit,CoverageTrial& result) {
  std::array<int8_t,kNodes> opening{};
  CaptureSearch search{limit,rng,&opening};
  const auto status=root.solve(pins,search);
  result.nodes+=limit-search.remaining;
  result.contradicted+=status==CaptureCompatibility::Contradicted;
  result.unresolved+=status==CaptureCompatibility::Unresolved;
  if (status==CaptureCompatibility::Contradicted) {
    result.collision_conflicts+=search.first_conflict_ply>0;
    result.inventory_conflicts+=search.inventory_conflict;
    if (result.first_conflict<0 && search.first_conflict_ply>0) result.first_conflict=search.first_conflict_ply;
  }
  if (status!=CaptureCompatibility::Consistent) return;
  ++result.solved;
  PrefixReplayWork work;
  const auto world=root.verify(opening,work);
  result.replay_plies+=work.moves;
  if (!world) {
    ++result.replay_failed;
    if (result.first_replay_failure<0) result.first_replay_failure=work.failed_ply;
    return;
  }
  if (std::none_of(result.worlds.begin(),result.worlds.end(),[&](const State& other) {
      return std::equal(std::begin(other.rank_at),std::end(other.rank_at),std::begin(world->rank_at));
    })) result.worlds.push_back(*world);
}

static CoverageTrial coverage_panel(const CoverageRoot& root,uint64_t seed,bool direct) {
  constexpr unsigned cap=4096;
  const auto& view=root.decision.observation;
  CoverageTrial result;
  for (int attempt=0;attempt<48 && result.worlds.size()<4 && result.nodes<cap;++attempt) {
    ++result.proposals;
    const auto salt=mix_seed(seed,attempt);
    std::array<unsigned,kNodes> pins{};
    Rng rng(salt);
    if (!direct) {
      const auto sample=public_sample(view,root.evidence,salt);
      if (!sample) { ++result.unavailable; continue; }
      for (int node=0;node<kNodes;++node) if (sample->rank_at[node]!=Unknown) pins[node]=1u<<sample->rank_at[node];
    }
    coverage_attempt(root,pins,direct?&rng:nullptr,std::min(384u,cap-result.nodes),result);
  }
  return result;
}

static bool same_capture_evidence(const CoverageRoot& root,const State& world,const StudentNetwork& student) {
  int8_t layouts[kSeats][25];
  for (int seat=0;seat<kSeats;++seat) for (int slot=0;slot<25;++slot)
    layouts[seat][slot]=world.rank_of[seat*kPiecesPerSeat+slot];
  const auto& view=root.decision.observation;
  const auto hidden=coverage_prefix(root.raw,layouts,root.rules,view.viewer,view.ply);
  const auto alternative=seat_decision(hidden.game.state);
  std::vector<CaptureEvidence> before,after;
  root.prefix.game.reply_history.for_actor(view,root.prefix.game.marks[view.viewer],&before);
  const auto marks=hidden.game.reply_history.for_actor(alternative.observation,hidden.game.marks[view.viewer],&after);
  const auto fields=[](const CaptureEvidence& row) {
    return std::make_tuple(row.ply,row.attacker,row.defender,row.attacker_seat,row.defender_seat,
                           row.outcome,row.attacker_ranks,row.defender_ranks);
  };
  if (before.size()!=after.size() || !std::equal(before.begin(),before.end(),after.begin(),
      [&](const auto& a,const auto& b){return fields(a)==fields(b);} )) return false;
  for (int seat=0;seat<kSeats;++seat)
    if (marks.removed[seat].masks!=root.evidence.removed[seat].masks) return false;
  if (alternative.actions!=root.decision.actions) return false;
  for (const auto move:alternative.actions) {
    const auto a=student_features(root.decision,root.evidence,move,&root.rules);
    const auto b=student_features(alternative,marks,move,&root.rules);
    if (a!=b || student.score(a)!=student.score(b)) return false;
  }
  return marks.ranks==root.evidence.ranks && marks.moved==root.evidence.moved;
}

static void print_coverage_panel(const std::string& key,int mode,int target,const CoverageTrial& result,long long us) {
  unsigned ranks=0;
  for (const auto& world:result.worlds) if (target>=0) ranks|=1u<<world.rank_at[target];
  std::cout<<"P "<<key<<' '<<mode<<' '<<result.proposals<<' '<<result.nodes<<' '<<result.unavailable<<' '
           <<result.contradicted<<' '<<result.unresolved<<' '<<result.solved<<' '<<result.replay_failed<<' '
           <<result.replay_plies<<' '<<result.worlds.size()<<' '<<ranks<<' '<<result.collision_conflicts<<' '
           <<result.inventory_conflicts<<' '<<result.first_conflict<<' '<<result.first_replay_failure<<' '<<us<<'\n';
}

static int coverage_target(const SeatDecision& decision,const OpeningMarks& evidence) {
  const auto& view=decision.observation;
  for (const auto move:decision.actions) {
    const auto mask=compatible_rank_mask(view,evidence,move.to);
    if (view.rank_at(move.to)==Unknown && (mask&(1u<<Bomb)) && (mask&~(1u<<Bomb))) return move.to;
  }
  return kNone;
}

static void coverage_controls(const CoverageRoot& root,const std::string& key,uint64_t seed,
                              const StudentNetwork& student) {
  const auto& view=root.decision.observation;
  PrefixReplayWork work;
  const auto truth=verify_coverage_layout(root.raw,root.raw.layouts,root.rules,root.prefix,root.decision,work);
  single_case_check(truth.has_value(),"original replay did not pass the common verifier");
  const auto& game=root.prefix.game;
  const auto& actor=game.marks[view.viewer];
  const auto control=game.reply_history.check_candidate(view,actor,truth->rank_at,4096);
  single_case_check(control==CaptureCompatibility::Consistent,"collision ledger rejected the original replay");
  const auto nearby=replay_prefix_witnesses(root.raw,root.rules,root.prefix.history,root.decision,key,0,seed);
  int invariant=0;
  for (const auto& world:nearby) {
    single_case_check(game.reply_history.check_candidate(view,actor,world.rank_at,4096)==CaptureCompatibility::Consistent,
                      "collision ledger rejected a strict replay witness");
    invariant+=same_capture_evidence(root,world,student);
  }
  single_case_check(invariant==int(nearby.size()),"hidden witness changed visible ledger or 44-input student scores");
  int negative_ply=-1;
  Rng rng(seed);
  for (int attempt=0;attempt<96 && negative_ply<0;++attempt) {
    const int seat=(view.viewer+1+int(rng.below(3)))%kSeats;
    const auto a=rng.below(25), b=rng.below(25);
    int8_t layouts[kSeats][25];
    for (int s=0;s<kSeats;++s) std::copy_n(root.raw.layouts[s],25,layouts[s]);
    std::swap(layouts[seat][a],layouts[seat][b]);
    PrefixReplayWork bad;
    verify_coverage_layout(root.raw,layouts,root.rules,root.prefix,root.decision,bad);
    if (bad.collision_mismatch) negative_ply=bad.failed_ply;
  }
  std::cout<<"C "<<key<<' '<<work.moves<<' '<<nearby.size()<<' '<<invariant<<' '<<negative_ply<<'\n';
}

static void assess_capture_coverage(const ArchiveRows& raw,const Rules& rules,const std::string& key,
    int viewer,int cut,uint64_t seed,const StudentNetwork& student) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto decision=seat_decision(prefix.game.state);
  const auto evidence=prefix.game.reply_history.for_actor(decision.observation,prefix.game.marks[viewer]);
  const CoverageRoot root{raw,rules,prefix,decision,evidence};
  const auto target=coverage_target(decision,evidence);
  std::cout<<"T "<<key<<' '<<cut<<' '<<target<<' '<<decision.actions.size()<<'\n';
  coverage_controls(root,key,seed,student);
  for (bool direct:{false,true}) {
    const auto start=std::chrono::steady_clock::now();
    const auto source_seed=public_rollout_seed(decision.observation,evidence,decision.observation.quiet_clock);
    const auto panel=coverage_panel(root,mix_seed(seed,source_seed),direct);
    const auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
    print_coverage_panel(key,int(direct),target,panel,us);
  }
  if (target<0) return;
  for (bool bomb:{false,true}) {
    std::array<unsigned,kNodes> pins{};
    pins[target]=bomb?(1u<<Bomb):all_evidence_ranks&~(1u<<Bomb);
    CoverageTrial result;
    for (int attempt=0;attempt<4 && result.worlds.empty() && result.nodes<1536;++attempt) {
      ++result.proposals;
      Rng rng(mix_seed(seed,attempt));
      coverage_attempt(root,pins,&rng,std::min(384u,1536u-result.nodes),result);
      if (result.contradicted) break;
    }
    print_coverage_panel(key,2+int(bomb),target,result,0);
  }
}

static void run_capture_coverage(const char* rules_path) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid capture coverage student");
  std::string key,hex,error;
  int viewer,cut,ccw;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>ccw>>seed>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && (ccw==0 || ccw==1) &&
        parse_rules(archive_review_rules(read_text(rules_path),bool(ccw)),rules,error),"invalid capture coverage wire");
    assess_capture_coverage(ArchiveRows(decode_archive_tape(hex)),rules,key,viewer,cut,seed,student);
  }
  single_case_check(std::cin.eof(),"incomplete capture coverage wire");
}
