#pragma once

using WitnessOpening=std::array<int8_t,kNodes>;

static WitnessOpening witness_layout(const State& start) {
  WitnessOpening ranks{};
  for (int s=0;s<kSeats;++s) for (int i=0;i<kPiecesPerSeat;++i)
    ranks[layout_slot_node(s,i)]=start.rank_of[s*kPiecesPerSeat+i];
  return ranks;
}

template<class Verify>
static void witness_controls(const std::string& key,WitnessOpening original,int viewer,Verify verify) {
  PrefixReplayWork same,valid,invalid;
  single_case_check(verify(original,same),"original layout failed its replay contract");
  bool accepted=false,rejected=false;
  for (int seat=0;seat<kSeats && !(accepted && rejected);++seat) if (seat!=viewer) {
    for (int a=0;a<25 && !(accepted && rejected);++a) for (int b=a+1;b<25 && !(accepted && rejected);++b) {
      const int x=layout_slot_node(seat,a),y=layout_slot_node(seat,b);
      if (original[x]==original[y] || !deployment_rank_allowed(x,original[y]) || !deployment_rank_allowed(y,original[x])) continue;
      auto other=original;
      std::swap(other[x],other[y]);
      PrefixReplayWork result;
      if (verify(other,result)) { valid=result; accepted=true; }
      else { invalid=result; rejected=true; }
    }
  }
  single_case_check(accepted && rejected,"replay controls did not discriminate legal alternatives");
  std::cout<<"C "<<key<<' '<<same.moves<<' '<<valid.moves<<' '<<invalid.failed_ply<<' '<<invalid.reason<<'\n';
}

template<class Solve,class Verify>
static void witness_attempts(const std::string& key,int focus,uint64_t seed,bool generated,Solve solve,Verify verify) {
  for (int side=0;side<2;++side) for (int i=0;i<3;++i) {
    std::array<unsigned,kNodes> pins{};
    pins[focus]=side?(1u<<Bomb):all_evidence_ranks&~(1u<<Bomb);
    WitnessOpening result{};
    Rng rng(mix_seed(seed,generated?200+i:i));
    CaptureSearch attempt{generated?384u:512u,&rng,&result};
    const auto status=solve(pins,attempt);
    PrefixReplayWork report;
    const bool ok=status==CaptureCompatibility::Consistent && verify(result,report);
    std::cout<<"F "<<key<<' '<<side<<' '<<i<<' '<<static_cast<int>(status)<<' '<<int(ok)<<' '
             <<report.failed_ply<<' '<<report.reason<<'\n';
  }
}

static void audit_generated_witness(const std::string& key,uint64_t seed,int learner,int table,int cut,const Rules& rules) {
  const auto start=opening(seed,rules);
  DarkOpeningPath path(start,learner);
  auto streams=review_streams(mix_seed(seed,53017));
  while (path.game.state.result==Result::Ongoing && path.game.state.ply<cut) {
    const auto d=seat_decision(path.game.state);
    path.play(dark_actor_move(path.game,d,rules,learner,table,streams[d.observation.viewer]),learner,rules);
  }
  single_case_check(path.game.state.ply==cut && path.game.state.to_move==learner,"generated witness root did not restore");
  const auto d=seat_decision(path.game.state);
  const auto& view=d.observation;
  const auto& history=path.game.reply_history;
  const auto& marks=path.game.marks[learner];
  const auto evidence=history.for_actor(view,marks);
  const auto focus=directed_contact(d,evidence,rules,0);
  single_case_check(focus.kind==0,"generated focus changed");
  const auto verify=[&](const WitnessOpening& layout,PrefixReplayWork& w) {
    return replay_dark_opening(path,learner,rules,layout,&w).has_value();
  };
  witness_controls(key,witness_layout(start),learner,verify);
  const auto solve=[&](const auto& pins,CaptureSearch& work) {
    return history.solve_candidate(view,marks,path.observations.front(),pins,work);
  };
  witness_attempts(key,focus.move.to,mix_seed(seed,public_rollout_seed(view,evidence,view.quiet_clock)),true,solve,verify);
}

static void audit_archived_witness(const std::string& key,const ArchiveRows& raw,int viewer,int cut,uint64_t seed,const Rules& rules) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto d=seat_decision(prefix.game.state);
  const auto& view=d.observation;
  const auto evidence=prefix.game.reply_history.for_actor(view,prefix.game.marks[viewer]);
  const auto focus=directed_contact(d,evidence,rules,0);
  single_case_check(focus.kind==0,"archived focus changed");
  CoverageRoot root{raw,rules,prefix,d,evidence};
  const auto verify=[&](const WitnessOpening& layout,PrefixReplayWork& w) {return root.verify(layout,w).has_value();};
  witness_controls(key,witness_layout(prefix.game.state),viewer,verify);
  const auto solve=[&](const auto& pins,CaptureSearch& work){return root.solve(pins,work);};
  witness_attempts(key,focus.move.to,mix_seed(seed,public_rollout_seed(view,evidence,view.quiet_clock)),false,solve,verify);
}

static void run_dark_witness_audit(const char* path) {
  const auto config=read_text(path);
  std::string tag,key,error,hex;
  uint64_t seed;
  int viewer,cut,table;
  while (std::cin>>tag>>key>>seed>>viewer>>cut>>table) {
    Rules rules;
    if (tag=="G") {
      single_case_check(parse_rules(config,rules,error),"invalid generated witness rules");
      audit_generated_witness(key,seed,viewer,table,cut,rules);
    } else {
      single_case_check(tag=="J" && (table==0 || table==1) && bool(std::cin>>hex) &&
                            parse_rules(archive_review_rules(config,bool(table)),rules,error),"invalid archive witness wire");
      audit_archived_witness(key,ArchiveRows(decode_archive_tape(hex)),viewer,cut,seed,rules);
    }
  }
  single_case_check(std::cin.eof(),"incomplete strict witness audit request");
}
