#include "../sim/archive_review.h"
#include <iostream>
#include "replay_qh_wire.h"

using namespace jq;

static std::string unpack(const std::string& hex) {
  expect_value(hex.size()%2==0 && hex.size()<=131072,"invalid replay hex length");
  const auto nibble=[](char c) {
    expect_value((c>='0' && c<='9') || (c>='a' && c<='f'),"invalid replay hex digit");
    return c<='9'?c-'0':c-'a'+10;
  };
  std::string result(hex.size()/2,'\0');
  for(size_t i=0;i<result.size();++i) result[i]=static_cast<char>(16*nibble(hex[2*i])+nibble(hex[2*i+1]));
  return result;
}

struct IntakeTrace {
  int plies=0, contacts=0, mid=0, post=0, notice=0, stopped=0, reason=0;
  Result result=Result::Ongoing;
  int learner=kNone;

  void before(const State& state) {
    if (state.to_move!=learner || notice) return;
    const int foes=int(!state.out[(learner+1)%kSeats])+int(!state.out[(learner+3)%kSeats]);
    if (foes==2 && !mid && state.ply>=72 && state.ply<140) mid=state.ply+1;
    if (foes==1 && !post && state.ply<140) post=state.ply+1;
  }
};

static IntakeTrace qh_trace(const int8_t (&layouts)[4][25],const std::vector<std::array<int,4>>& rows,const Rules& rules) {
  IntakeTrace trace;
  for (const auto& layout:layouts) if(!valid_layout(layout)) {trace.stopped=1;return trace;}
  if(rows.empty() || rows.front()[0]>16) {trace.stopped=4;return trace;}
  const auto& board=Board::get();
  const int origin=point(rows.front()[0],rows.front()[1]);
  if(origin<0 || board.node(origin).seat<0) {trace.stopped=2;return trace;}
  State state;
  trace.learner=board.node(origin).seat;
  init_state(state,layouts,trace.learner,rules);
  for (const auto& a:rows) {
    if(a[0]>16) {trace.stopped=4;trace.notice=state.ply+1;break;}
    const int from=point(a[0],a[1]),to=point(a[2],a[3]);
    if(from<0 || to<0) {trace.stopped=2;break;}
    const Move action{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    if(!is_legal(state,action)) {trace.stopped=state.result==Result::Ongoing?3:5;break;}
    trace.before(state);
    const auto event=step(state,action,rules);
    trace.contacts+=event.outcome!=Outcome::Move;
    trace.reason=int(event.end_reason);
  }
  trace.plies=state.ply;
  trace.result=state.result;
  return trace;
}

static IntakeTrace qq_trace(const ArchiveRows& rows,const Rules& rules) {
  ArchiveReview review;
  IntakeTrace trace;
  trace.learner=archive_first_seat(rows);
  init_state(review.state,rows.layouts,trace.learner,rules);
  for(int i=0;i<rows.entries;++i) {
    const auto row=rows.row(i);
    if (rows.byte(row)==245) {
      const int kind=rows.byte(row+1),cause=rows.byte(row+3);
      const bool routine=(kind==3 && (cause==1 || cause==2)) || (kind==5 && cause==4);
      if (!routine && !trace.notice) trace.notice=review.state.ply+1;
      review.control(rows,row,i+1,rules);
    } else {
      trace.before(review.state);
      review.play(rows,row,i+1,rules);
      trace.contacts+=review.last.outcome!=Outcome::Move;
    }
    review.frames.str("");
  }
  trace.plies=review.state.ply;
  trace.result=review.state.result;
  trace.stopped=review.ended && review.captured_flag==kNone?0:6;
  return trace;
}

static void print_trace(const std::string& key,const IntakeTrace& t,int direction) {
  std::cout<<key<<' '<<direction<<' '<<t.plies<<' '<<t.contacts<<' '<<t.mid<<' '<<t.post<<' '<<t.notice
           <<' '<<t.stopped<<' '<<int(t.result)<<' '<<t.reason<<' '<<t.learner<<'\n';
}

int main(int argc,char** argv) {
  if (argc!=2) return 2;
  Rules archive,qh;
  std::string error;
  if (!parse_rules(archive_review_rules(read_text(argv[1]),false),archive,error)) return 3;
  qh.ccw=false;
  qh.no_progress_limit=0;
  std::string tag,key,raw;
  while(std::cin>>tag>>key) {
    if(tag=="H") {
      int8_t layouts[4][25]{};
      std::vector<std::array<int,4>> rows;
      if (!read_qh(layouts,rows)) return 4;
      print_trace(key,qh_trace(layouts,rows,qh),0);
      continue;
    }
    if(tag!="G" || !(std::cin>>raw)) return 5;
    bool done=false;
    try {
      const ArchiveRows rows(unpack(raw));
      for (bool ccw:{false,true}) {
        archive.ccw=ccw;
        try {print_trace(key,qq_trace(rows,archive),ccw);done=true;break;}
        catch(const std::runtime_error&) {}
      }
    } catch(const std::runtime_error&) {}
    if(!done) {IntakeTrace missing;missing.stopped=7;print_trace(key,missing,-1);}
  }
  return 0;
}
