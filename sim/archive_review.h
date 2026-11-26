#pragma once
#include <array>
#include <filesystem>
#include <optional>
#include "header.h"
#include "replay.h"

namespace jq {

struct ArchiveRows {
  std::string file;
  int entries=0;
  std::array<int,kSeats> seat{};
  int8_t layouts[kSeats][kPiecesPerSeat]{};

  int byte(size_t i) const { return static_cast<unsigned char>(file[i]); }
  size_t row(int i) const { return 412 + static_cast<size_t>(i)*10; }

  explicit ArchiveRows(std::string content) : file(std::move(content)) {
    expect_value(file.size()>=412 && file.size()<=65536 && file.compare(0,10,"QQGame JQS")==0,
                 "unsupported JGS signature or size");
    entries=byte(26)+256*byte(27);
    expect_value(entries>=2 && row(entries)<=file.size(),"incomplete JGS event table");
    unsigned seen=0;
    const auto& board=Board::get();
    for(int s=0;s<kSeats;++s) {
      const int block=60+88*((s+3)%kSeats);
      const int code=byte(block-28);
      expect_value(code<kSeats && !(seen&(1u<<code)),"invalid JGS seats");
      seen|=1u<<code;
      seat[code]=s;
      for(int slot=0;slot<kPiecesPerSeat;++slot) {
        const auto& node=board.node(layout_slot_node(s,slot));
        const int rank=byte(block+5*node.r+4-node.c);
        expect_value(rank>=2 && rank<=13,"incomplete JGS layout");
        layouts[s][slot]=static_cast<int8_t>(rank<5?rank-2:16-rank);
      }
      for(int r=0;r<6;++r) for(int c=0;c<5;++c)
        if(board.node(board.seat_node(s,r,c)).kind==NodeKind::Camp)
          expect_value(byte(block+5*r+4-c)==0,"JGS layout has an occupied camp");
      expect_value(valid_layout(layouts[s]),"invalid JGS deployment");
    }
  }

  int point(size_t i) const {
    const int x=byte(i), y=byte(i+1);
    return x<=16 && y<=16 ? Board::get().at_grid(16-y,x) : kNone;
  }

  std::string tape() const {
    // Preserve only structural fields. Original player fields never enter JSONL.
    std::string record(row(entries),'\0');
    record.replace(0,10,file,0,10);
    record.replace(26,2,file,26,2);
    for(int block:{60,148,236,324}) {
      record[block-28]=file[block-28];
      record.replace(block,30,file,block,30);
    }
    for(int i=0;i<entries;++i) {
      const auto start=row(i);
      const auto bytes=byte(start)==95?10:5;
      record.replace(start,bytes,file,start,bytes);
    }
    constexpr char digits[]="0123456789abcdef";
    std::string hex;
    for(const unsigned char value:record) {
      hex+=digits[value>>4];
      hex+=digits[value&15];
    }
    return hex;
  }
};

inline int archive_first_seat(const ArchiveRows& raw) {
  const auto row=raw.row(0);
  if(raw.byte(row)==245 && raw.byte(row+1)==2) {
    const int code=raw.byte(row+2), next=raw.byte(row+4);
    expect_value(code<kSeats && next<kSeats && code!=next,"invalid leading JGS skip");
    return raw.seat[code];
  }
  const int origin=raw.point(row+2);
  expect_value(raw.byte(row)==95 && origin>=0,"JGS must begin with a move or an explicit skip");
  const int first=Board::get().node(origin).seat;
  expect_value(first>=0,"JGS first seat cannot be identified from deployment");
  return first;
}

inline std::string archive_review_rules(const std::string& rules_config,bool ccw) {
  auto config=parse_json(rules_config);
  auto& options=config.object.at("experimental").object;
  options["turn.order"]=parse_json(ccw?"\"ccw\"":"\"cw\"");
  options["elim.no_legal_move"]=parse_json("\"eliminated_after_involved_move\"");
  options["draw.no_progress.limit"]=parse_json("71");
  options["marshal_death.reveal_flag"]=parse_json("true");
  config.object["id"]=parse_json("\"qq-four-archive-review-1\"");
  return dump_json(config);
}

struct ArchiveReview {
  State state{};
  StepInfo last{};
  int captured_flag=kNone;
  bool ended=false;
  std::vector<int> source_steps{0};
  std::vector<std::array<int,5>> controls;
  std::ostringstream frames;

  void emit() {
    expect_value(captured_flag==kNone,"JGS flag was removed without its retirement record");
    frames<<frame_json(state,state.ply>0?&last:nullptr,IdentityVisibility::OwnAndPublic,false,true)<<'\n';
  }

  void advance(const Rules& rules) {
    for(int i=0;i<kSeats;++i) {
      state.to_move=static_cast<int8_t>(next_seat(state.to_move,rules));
      if(!state.out[state.to_move]) return;
    }
  }

  void play(const ArchiveRows& raw,size_t row,int sequence,const Rules& rules) {
    expect_value(!ended,"JGS move after terminal");
    emit();
    const int from=raw.point(row+2), to=raw.point(row+4);
    expect_value(from>=0 && to>=0,"invalid JGS move coordinates");
    const Move move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    const State before=state;
    last=recorded_move(state,move,rules);
    expect_value(int(last.outcome)==(raw.byte(row+1)&3),"JGS collision differs from core");
    unsigned revealed=0;
    for(size_t i:{row+6,row+8}) {
      if(!raw.byte(i) && !raw.byte(i+1)) continue;
      const int point=raw.point(i);
      expect_value(point>=0 && before.rank_at[point]==Flag,"JGS reveal has no matching flag");
      revealed|=1u<<before.seat_at[point];
    }
    expect_value(revealed==last.revealed_mask,"JGS flag reveal differs from core");
    if(before.rank_at[to]==Flag && (last.outcome==Outcome::AttackerWins || last.outcome==Outcome::BothRemoved))
      captured_flag=before.seat_at[to];
    advance(rules);
    source_steps.push_back(sequence);
  }

  void retire(int seat,int kind,int cause) {
    ExitReason reason=ExitReason::None;
    if(kind==3 && cause==1) {
      expect_value(seat==captured_flag,"JGS flag retirement has no matching collision");
      captured_flag=kNone;
      reason=ExitReason::FlagRemoved;
    }
    if(kind==3 && cause==2) {
      std::vector<Move> moves;
      gen_moves(StateView{state},seat,moves);
      expect_value(moves.empty(),"JGS no-move retirement still has a legal move");
      reason=ExitReason::NoLegalMove;
    }
    if(kind==3 || !state.out[seat]) recorded_retirement(state,seat,last,reason);
  }

  void control(const ArchiveRows& raw,size_t row,int sequence,const Rules& rules) {
    expect_value(!ended,"JGS control outside a live record");
    const int kind=raw.byte(row+1), code=raw.byte(row+2), cause=raw.byte(row+3), next=raw.byte(row+4);
    if(state.ply==0)
      expect_value(kind==2 && code<kSeats && next<kSeats && code!=next &&
                   raw.seat[code]==state.to_move,"invalid leading JGS skip");
    int seat=kNone;
    if(kind>=2 && kind<=4 && code<kSeats) {
      seat=raw.seat[code];
      if(kind!=2) retire(seat,kind,cause);
      if(next<kSeats) {
        expect_value(!state.out[raw.seat[next]],"JGS names a retired next seat");
        state.to_move=static_cast<int8_t>(raw.seat[next]);
      } else {
        expect_value(next==255 && kind!=2,"invalid JGS next seat");
        if(state.out[state.to_move]) advance(rules);
      }
    } else if(kind==5) {
      const bool lost0=state.out[0]&&state.out[2], lost1=state.out[1]&&state.out[3];
      state.result=cause==2 ? Result::Draw : Result::Truncated;
      if(cause==4 && lost0!=lost1) state.result=lost0?Result::Team1Wins:Result::Team0Wins;
      ended=true;
    } else {
      throw std::runtime_error("unsupported JGS control kind="+std::to_string(kind)+
                               " cause="+std::to_string(cause));
    }
    controls.push_back({state.ply,sequence,kind,seat,cause});
  }
};

inline std::string archive_review_attempt(const ArchiveRows& raw,const std::string& config,
                                          const std::string& observation,bool ccw) {
  auto rt=archive_review_rules(config,ccw);
  Rules rules;
  std::string error;
  if(!parse_rules(rt,rules,error)) throw std::runtime_error(error);
  auto obs=parse_json(observation);
  obs.object["marshal_reveals_flag"]=parse_json("true");
  const auto ot=match_observation_config(dump_json(obs),rules,IdentityVisibility::OwnAndPublic);
  const int first=archive_first_seat(raw);
  ArchiveReview review;
  init_state(review.state,raw.layouts,first,rules);
  for(int i=0;i<raw.entries;++i) {
    const auto row=raw.row(i);
    try {
      switch(raw.byte(row)) {
        case 95: review.play(raw,row,i+1,rules); break;
        case 245: review.control(raw,row,i+1,rules); break;
        default: throw std::runtime_error("unsupported JGS record type");
      }
    } catch(const std::runtime_error& issue) {
      throw std::runtime_error("JGS record "+std::to_string(i+1)+": "+issue.what());
    }
  }
  expect_value(review.ended,"missing JGS terminal record");
  review.emit();
  std::ostringstream base;
  write_header(base,0,rules,rt,ot,first,raw.layouts);
  auto header=parse_json(base.str());
  std::ostringstream sequence,controls;
  array_json(sequence,review.source_steps.data(),static_cast<int>(review.source_steps.size()));
  controls<<'[';
  for(size_t i=0;i<review.controls.size();++i) {
    if(i) controls<<',';
    array_json(controls,review.controls[i].data(),5);
  }
  controls<<']';
  header.object["review_version"]=parse_json("1");
  header.object["source_steps"]=parse_json(sequence.str());
  header.object["source_controls"]=parse_json(controls.str());
  header.object["source_tape"]=parse_json(quote_json(raw.tape()));
  return dump_json(header)+'\n'+review.frames.str();
}

inline std::string decode_archive_tape(const std::string& hex) {
  expect_value(hex.size()%2==0 && hex.size()<=2*65536,"invalid JGS verification tape size");
  const auto digit=[](char c) {
    if(c>='0' && c<='9') return c-'0';
    if(c>='a' && c<='f') return c-'a'+10;
    throw std::runtime_error("invalid JGS verification tape digit");
  };
  std::string raw;
  for(size_t i=0;i<hex.size();i+=2) raw+=static_cast<char>(digit(hex[i])*16+digit(hex[i+1]));
  return raw;
}

inline bool verify_archive_review(const std::string& text) {
  const auto header=parse_json(text.substr(0,text.find('\n')));
  if(!header.object.count("source_tape")) return false;
  const auto& rules=header.at("rules");
  const bool ccw=rules.at("experimental").at("turn.order").str()=="ccw";
  const ArchiveRows raw(decode_archive_tape(header.at("source_tape").str()));
  expect_value(text==archive_review_attempt(raw,dump_json(rules),dump_json(header.at("observation")),ccw),
               "JGS review differs from its recorded events");
  std::cout<<"JGS REPLAY PASS records="<<raw.entries<<'\n';
  return true;
}

inline void import_archive_review(const std::string& input,const std::string& output,
                                  const std::string& rules,const std::string& observation) {
  expect_value(!std::filesystem::exists(output),"review output exists");
  const ArchiveRows raw(read_text(input));
  const auto config=read_text(rules), visibility=read_text(observation);
  std::optional<std::string> accepted;
  std::string rejected;
  for(bool ccw:{false,true}) {
    try {
      auto candidate=archive_review_attempt(raw,config,visibility,ccw);
      if(accepted) {
        accepted.reset();
        rejected="JGS turn direction is ambiguous";
        break;
      }
      accepted=std::move(candidate);
    } catch(const std::runtime_error& issue) {
      rejected+=(rejected.empty()?"":" | ")+std::string(issue.what());
    }
  }
  if(!accepted) throw std::runtime_error(rejected);
  expect_value(accepted->size()<=64000000,"JGS review output too large");
  std::ofstream result(output,std::ios::binary);
  result<<*accepted;
  result.flush();
  expect_value(bool(result),"could not write JGS review output");
  std::cout<<"JGS REVIEW PASS records="<<raw.entries<<'\n';
}

}
