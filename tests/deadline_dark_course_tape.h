#pragma once

using CourseEvent=std::array<int,13>;

static CourseEvent course_event(const State& state,const StepInfo& info) {
  return {info.move.from,info.move.to,info.seat,int(info.outcome),info.eliminated_mask,info.revealed_mask,
          info.skipped_mask,int(info.exit_reason[0]),int(info.exit_reason[1]),int(info.exit_reason[2]),
          int(info.exit_reason[3]),int(info.end_reason),int(state.result)};
}

struct RecordedCourse {
  uint64_t seed;
  int viewer,table,arm,outcome;
  State start;
  std::vector<int> picks;
  std::vector<CourseEvent> events;
};

static RecordedCourse read_recorded_course(std::istream& in,uint64_t seed,const Rules& rules) {
  RecordedCourse course{};
  course.seed=seed;
  int first,length,picks;
  single_case_check(bool(in>>course.viewer>>course.table>>course.arm>>course.outcome>>first>>length>>picks) &&
     course.viewer>=0 && course.viewer<kSeats && first>=0 && first<kSeats &&
     course.outcome>=-1 && course.outcome<=2 && length>0 && length<=720 && picks>=0 && picks<=6,
     "invalid recorded course header");
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout:layouts) {
    for (auto& slot:layout) {
      int rank;
      single_case_check(bool(in>>rank) && rank>=Flag && rank<RankCount,"invalid recorded opening rank");
      slot=static_cast<int8_t>(rank);
    }
    single_case_check(valid_layout(layout),"recorded course opening violates deployment rules");
  }
  init_state(course.start,layouts,first,rules);
  course.picks.resize(picks);
  int prior=-1;
  for (auto& p:course.picks) {
    single_case_check(bool(in>>p) && p>prior && p<length,"duplicate or future recorded sample");
    prior=p;
  }
  course.events.resize(length);
  for (auto& event:course.events) for (auto& field:event)
    single_case_check(bool(in>>field),"incomplete course event tape");
  return course;
}

enum class CourseReplay { Same, Illegal, EventChanged, OutcomeChanged };
struct CourseReplayEvidence {
  int matched=0;
  int outcome=-1;
  CourseReplay status=CourseReplay::Same;
  std::vector<std::vector<double>> features;
};

static CourseReplayEvidence replay_recorded_course(const RecordedCourse& course,const Rules& rules) {
  EarlyPrefix game(course.start,course.seed);
  CourseReplayEvidence result;
  for (const auto& wanted:course.events) {
    const int from=wanted[0],to=wanted[1];
    if (from<0 || from>=kNodes || to<0 || to>=kNodes || !is_legal(game.state,{uint8_t(from),uint8_t(to)})) {
      result.status=CourseReplay::Illegal;
      return result;
    }
    const Move move{uint8_t(from),uint8_t(to)};
    auto next=game.state;
    const auto expected=step(next,move,rules);
    if (course_event(next,expected)!=wanted) {
      result.status=CourseReplay::EventChanged;
      return result;
    }
    if (std::find(course.picks.begin(),course.picks.end(),game.state.ply)!=course.picks.end()) {
      const auto d=seat_decision(game.state);
      single_case_check(d.observation.viewer==course.viewer,"recorded sample belongs to another observer");
      const auto marks=game.reply_history.for_actor(d.observation,game.marks[course.viewer]);
      result.features.push_back(student_features(d,marks,move,&rules));
    }
    game.play(move,course.viewer,rules);
    ++result.matched;
  }
  result.outcome=review_outcome(game.state.result,course.viewer);
  if (result.outcome!=course.outcome) result.status=CourseReplay::OutcomeChanged;
  single_case_check(result.features.size()==course.picks.size(),"course could not reconstruct every requested sample");
  return result;
}

static void run_dark_course_tape(const Rules& rules) {
  uint64_t seed;
  while (std::cin>>seed) {
    const auto course=read_recorded_course(std::cin,seed,rules);
    const auto result=replay_recorded_course(course,rules);
    std::cout<<"V "<<seed<<' '<<course.viewer<<' '<<course.table<<' '<<course.arm<<' '
             <<int(result.status)<<' '<<result.matched<<' '<<result.outcome<<'\n';
    if (result.status!=CourseReplay::Same || result.outcome<0) continue;
    for (const auto& x:result.features) {
      std::cout<<"E "<<seed<<' '<<course.viewer<<' '<<course.table<<' '<<course.arm<<' '<<result.outcome;
      for (double value:x) std::cout<<' '<<std::setprecision(17)<<value;
      std::cout<<'\n';
    }
  }
  single_case_check(std::cin.eof(),"incomplete recorded course receipt");
}


static void run_dark_course_openings(const Rules& rules) {
  const auto& board=Board::get();
  std::cout<<"M";
  for (int i=0;i<kPiecesPerSeat;++i) {
    const auto cell=board.node(layout_slot_node(0,i));
    const int reflected=board.seat_node(0,cell.r,4-cell.c);
    int twin=0;
    while (twin<kPiecesPerSeat && layout_slot_node(0,twin)!=reflected) ++twin;
    single_case_check(twin<kPiecesPerSeat,"recorded layout is not mirror symmetric");
    std::cout<<' '<<twin;
  }
  std::cout<<'\n';
  uint64_t seed;
  while (std::cin>>seed) {
    const auto start=opening(seed,rules);
    std::cout<<"L "<<seed;
    for (const int rank:start.rank_of) std::cout<<' '<<rank;
    std::cout<<'\n';
  }
  single_case_check(std::cin.eof(),"incomplete recorded opening request");
}
