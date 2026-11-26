#pragma once

static int opening_engineer_exits(const State& state,int learner) {
  std::vector<Move> legal;
  gen_moves(StateView{state},learner,legal);
  return int(std::count_if(legal.begin(),legal.end(),[&](Move m) {
    return state.rank_at[m.from]==Engineer && Board::get().node(m.to).seat!=learner;
  }));
}

static void run_engineer_formation(const Rules& rules,bool telemetry=false) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true,true),"invalid engineer-formation student");
  std::array<int,2> admitted{};
  char format;
  std::string key;
  uint64_t seed;
  while (std::cin>>format>>key>>seed) {
    int8_t deployment[kSeats][kPiecesPerSeat]{};
    const int source=int(format!='R');
    if (format=='R') {
      const auto generated=opening(seed,rules);
      for (int seat=0;seat<kSeats;++seat) for (int slot=0;slot<kPiecesPerSeat;++slot)
        deployment[seat][slot]=generated.rank_at[layout_slot_node(seat,slot)];
    } else if (format=='H') {
      std::vector<std::array<int,4>> rows;
      single_case_check(read_qh(deployment,rows),"cannot restore archived deployment");
    } else {
      std::string tape;
      single_case_check(format=='G' && bool(std::cin>>tape),"unknown archive layout format");
      const ArchiveRows archived(decode_archive_tape(tape));
      for (int seat=0;seat<kSeats;++seat)
        std::copy_n(archived.layouts[seat],kPiecesPerSeat,deployment[seat]);
    }
    for (const auto& seat:deployment) single_case_check(valid_layout(seat),"invalid complete opening layout");
    const int i=admitted[source];
    if (i>=8) {
      std::cout<<"S "<<key<<' '<<source<<'\n';
      continue;
    }
    const int learner=i%kSeats;
    const int first=(learner+source+i/kSeats)%kSeats;
    const int table=(i%2+i/kSeats)%2;
    State initial;
    init_state(initial,deployment,first,rules);
    auto& own=deployment[learner];
    const int original=opening_engineer_exits(initial,learner);
    int best=original, from=kNone, into=kNone, rank=kNone;
    State shifted=initial;
    for (int a=0;a<kPiecesPerSeat;++a) if (own[a]==Engineer) {
      for (int b=0;b<kPiecesPerSeat;++b) if (own[b]>=Platoon && own[b]<=Brigade) {
        std::swap(own[a],own[b]);
        if (valid_layout(own)) {
          State modified;
          init_state(modified,deployment,first,rules);
          const int exits=opening_engineer_exits(modified,learner);
          if (exits>best) {
            from=a;
            into=b;
            rank=own[a];
            best=exits;
            shifted=modified;
          }
        }
        std::swap(own[a],own[b]);
      }
    }
    if (from==kNone) {
      std::cout<<"M "<<key<<' '<<source<<' '<<original<<'\n';
      continue;
    }
    int differences=0;
    for (int n=0;n<kNodes;++n) differences+=initial.rank_at[n]!=shifted.rank_at[n];
    single_case_check(differences==2 && initial.to_move==shifted.to_move &&
                      initial.rank_at[layout_slot_node(learner,from)]==Engineer &&
                      shifted.rank_at[layout_slot_node(learner,from)]==rank &&
                      shifted.rank_at[layout_slot_node(learner,into)]==Engineer,
                      "engineer perturbation changed more than one exchange");
    ++admitted[source];
    std::cout<<"A "<<key<<' '<<source<<' '<<learner<<' '<<first<<' '<<table<<' '<<from<<' '<<into<<' '
             <<rank<<' '<<original<<' '<<best<<' '<<state_hash(initial)<<' '<<state_hash(shifted)<<'\n';
    for (int arm=0;arm<4;++arm)
      run_full_student_episode(arm%2==0?initial:shifted,rules,seed,learner,table,arm,
                               FullStudentPlan{arm>=2?&student:nullptr,arm>=2},key,telemetry);
  }
  single_case_check(std::cin.eof() && admitted==std::array<int,2>{8,8},"insufficient eligible paired opening blocks");
}
