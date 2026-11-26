#include "../sim/archive_review.h"
#include <iostream>

using namespace jq;

static const char* differing(const State& next,const StepInfo& step_info,const ArchiveReview& review,
                             bool compare_result) {
  if (step_info.outcome!=review.last.outcome) return "collision";
  if (step_info.revealed_mask!=review.last.revealed_mask) return "reveal";
  if (step_info.eliminated_mask!=review.last.eliminated_mask ||
      !std::equal(std::begin(next.id_at),std::end(next.id_at),std::begin(review.state.id_at))) return "clear";
  if (compare_result && next.result!=review.state.result) return "result";
  if (next.result==Result::Ongoing && review.state.result==Result::Ongoing &&
      next.to_move!=review.state.to_move) return "next";
  return "same";
}

static std::array<Rules,6> variants(const Rules& archive,Rules training) {
  training.ccw=archive.ccw;
  std::array<Rules,6> rules{archive,training,archive,archive,archive,archive};
  rules[2].check_involved_no_move=false;
  rules[3].no_progress_limit=70;
  rules[4].marshal_reveals_flag=false;
  rules[5].bomb_captures_flag=!archive.bomb_captures_flag;
  return rules;
}

static bool independent_control(const ArchiveRows& raw,size_t row) {
  const int kind=raw.byte(row+1), cause=raw.byte(row+3);
  return (kind==3 && (cause==1 || cause==2)) || (kind==5 && cause==4);
}

static void tally(const State& before,Move move,const ArchiveReview& review,const Rules& archive,
                  const std::array<Rules,6>& rules,bool controlled,std::ostringstream& output) {
  const auto ply=review.state.ply;
  const bool draw=review.state.result==Result::Draw;
  if (draw || controlled) {
    output<<"C "<<ply<<' '<<static_cast<int>(review.state.result)<<' '<<review.state.no_progress<<'\n';
    return;
  }
  for (size_t i=0;i<rules.size();++i) {
    auto state=before;
    const auto e=step(state,move,rules[i]);
    const auto reason=differing(state,e,review,true);
    if (std::string_view(reason)!="same")
      output<<"M "<<i<<' '<<ply<<' '<<reason<<' '<<static_cast<int>(e.end_reason)
            <<' '<<unsigned(e.eliminated_mask)<<' '<<unsigned(review.last.eliminated_mask)<<'\n';
  }
  if (before.rank_at[move.to]==Flag && review.last.outcome!=Outcome::DefenderWins) {
    auto state=before;
    const auto action=step(state,move,archive);
    const int defender=before.seat_at[move.to];
    const bool cleared=state.out[defender] &&
      std::find(std::begin(state.seat_at),std::end(state.seat_at),defender)==std::end(state.seat_at);
    const bool locked=action.outcome==Outcome::AttackerWins && !can_leave_node(state.rank_at[move.to],move.to);
    output<<"F "<<ply<<' '<<cleared<<' '<<locked<<' '<<before.flag_revealed[defender]
          <<' '<<(before.rank_at[move.from]==Bomb)<<'\n';
  }
}

static std::string review_game(const ArchiveRows& raw,const Rules& archive,const Rules& training) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),archive);
  const auto profiles=variants(archive,training);
  std::ostringstream output;
  int index=0;
  while (index<raw.entries && raw.byte(raw.row(index))==245) {
    review.control(raw,raw.row(index),index+1,archive);
    ++index;
  }
  while (index<raw.entries) {
    const auto row=raw.row(index++);
    expect_value(raw.byte(row)==95,"expected JGS move");
    const auto before=review.state;
    review.play(raw,row,index,archive);
    review.frames.str("");
    bool controlled=false;
    while (index<raw.entries && raw.byte(raw.row(index))==245) {
      const auto notice=raw.row(index++);
      controlled|=!independent_control(raw,notice);
      review.control(raw,notice,index,archive);
    }
    tally(before,review.last.move,review,archive,profiles,controlled,output);
  }
  expect_value(review.ended && review.captured_flag==kNone,"missing terminal or retirement notice");
  output<<"G "<<review.state.ply<<' '<<static_cast<int>(review.state.result)<<' '
        <<review.state.no_progress<<' '<<archive.ccw<<'\n';
  return output.str();
}

static void run(const std::string& key,const std::string& input,Rules rules,const Rules& training) {
  try {
    const ArchiveRows raw(input);
    for (const bool ccw:{false,true}) {
      rules.ccw=ccw;
      try {
        const auto data=review_game(raw,rules,training);
        std::cout<<"S "<<key<<'\n'<<data;
        return;
      } catch (const std::runtime_error&) {
        // The archived registration order may depend on the room.
      }
    }
  } catch (const std::runtime_error&) {
    // Invalid archive does not contribute to rule evidence.
  }
  std::cout<<"R "<<key<<'\n';
}

int main(int argc,char** argv) {
  if (argc!=3) return 2;
  Rules archive,training;
  std::string error;
  if (!parse_rules(archive_review_rules(read_text(argv[1]),false),archive,error) ||
      !parse_rules(read_text(argv[2]),training,error)) return 3;
  std::string key,tape;
  while (std::cin>>key>>tape) {
    auto parsed=std::string(tape.size()/2,'\0');
    if (tape.find_first_not_of("0123456789abcdef")!=std::string::npos || tape.size()%2!=0) return 4;
    for (size_t i=0;i<parsed.size();++i)
      parsed[i]=static_cast<char>(std::stoi(tape.substr(2*i,2),nullptr,16));
    run(key,parsed,archive,training);
  }
}
