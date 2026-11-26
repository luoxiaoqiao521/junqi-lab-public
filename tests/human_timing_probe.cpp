#include "../sim/archive_review.h"
#include <algorithm>
#include <bitset>
#include <iostream>

using namespace jq;

struct Frame {
  State board;
  StepInfo action;
  std::array<int,kNodes> moved{};
  int sequence;
  size_t controls;
};

static std::array<int,kNodes> moved_nodes(const std::array<int,kNodes>& earlier,
                                          const ArchiveReview& review) {
  auto marks=earlier;
  if(review.state.ply>0) {
    const auto& event=review.last;
    marks[event.move.from]=0;
    if(event.outcome==Outcome::Move || event.outcome==Outcome::AttackerWins)
      marks[event.move.to]=review.state.ply;
  }
  for(int n=0;n<kNodes;++n) if(review.state.seat_at[n]==kNone) marks[n]=0;
  return marks;
}

static void frame(std::vector<Frame>& result,const ArchiveReview& game) {
  const auto marks=moved_nodes(result.empty()?std::array<int,kNodes>{}:result.back().moved,game);
  result.push_back({game.state,game.last,marks,game.source_steps.back(),game.controls.size()});
}

static std::vector<Frame> replay_trace(const ArchiveRows& raw,Rules rules) {
  ArchiveReview game;
  const int origin=raw.point(raw.row(0)+2);
  expect_value(raw.byte(raw.row(0))==95 && origin>=0,"first_move");
  const int first=Board::get().node(origin).seat;
  expect_value(first>=0,"first_seat");
  init_state(game.state,raw.layouts,first,rules);
  std::vector<Frame> trace;
  for(int i=0;i<raw.entries;++i) {
    const auto row=raw.row(i);
    if(raw.byte(row)==95) {
      frame(trace,game);
      game.play(raw,row,i+1,rules);
    } else if(raw.byte(row)==245) game.control(raw,row,i+1,rules);
    else throw std::runtime_error("record_type");
  }
  expect_value(game.ended,"terminal_record");
  expect_value(game.captured_flag==kNone,"unsettled_flag");
  frame(trace,game);
  return trace;
}

// The origin and flag must be visible from the declared perspective. A visibly
// moved dark piece can use any ordinary route, including when it is an engineer.
static std::bitset<kNodes> flag_routes(const Frame& frame,int actor,int viewer,int from) {
  std::bitset<kNodes> result;
  if(frame.board.result!=Result::Ongoing || frame.board.seat_at[from]!=actor) return result;
  auto view=observe(frame.board,viewer);
  const int rank=view.rank_at(from);
  if(rank==Flag || rank==Mine || (rank==Unknown && !frame.moved[from])) return result;
  if(rank==Unknown) view.ranks[from]=Platoon;
  std::bitset<kNodes> flags;
  for(int to=0;to<kNodes;++to) if(view.rank_at(to)==Flag && view.seat_at(to)>=0 &&
                                   view.team(view.seat_at(to))!=view.team(actor)) flags.set(to);
  if(flags.none()) return result;
  std::vector<Move> choices;
  gen_piece_moves(view,actor,from,choices);
  for(const auto move:choices) if(flags[move.to]) result.set(move.to);
  return result;
}

struct Route {
  int receiver, origin, flag, setup;
  bool opener_can_see;
};

static bool can_reach(const Frame& frame,int receiver,int flag) {
  for(int origin=0;origin<kNodes;++origin)
    if(frame.board.seat_at[origin]==receiver && flag_routes(frame,receiver,receiver,origin)[flag])
      return true;
  return false;
}

struct Reply {
  const char* outcome="no_turn";
  int next=0, cutoff=0, foes=0, closed_at=0;
};

static void write_window(const std::string& key,const char* kind,const Frame& opened,
                         const Route& route,const Reply& seen) {
  const auto& action=opened.action;
  std::cout<<"W "<<key<<' '<<kind<<' '<<seen.outcome<<' '<<opened.board.ply<<' '<<opened.sequence<<' '
           <<int(action.move.from)<<' '<<int(action.move.to)<<' '<<int(action.seat)<<' '
           <<route.receiver<<' '<<route.origin<<' '<<route.flag<<' '<<route.setup<<' '
           <<route.opener_can_see<<' '<<seen.next<<' '<<seen.cutoff<<' '<<seen.foes<<' '<<seen.closed_at<<'\n';
}

static Reply observe_replies(const std::vector<Frame>& trace,int ply,const Route& route) {
  Reply result;
  for(int j=ply+1;j<=ply+8 && static_cast<size_t>(j)<trace.size();++j) {
    const auto& now=trace[j];
    const auto mover=now.action.seat;
    if(mover==route.receiver) {
      result.next=j;
      const auto step=now.action.move;
      if(step.to==route.flag && (now.action.outcome==Outcome::AttackerWins ||
                                 now.action.outcome==Outcome::BothRemoved)) result.outcome="captured";
      else if(can_reach(trace[j-1],route.receiver,route.flag))
        result.outcome="available_unused";
      else result.outcome="closed";
      return result;
    }
    result.foes+=team_of(mover)!=team_of(route.receiver);
    if(now.action.move.to==route.flag && now.action.outcome!=Outcome::Move &&
       now.action.outcome!=Outcome::DefenderWins) {
      result.cutoff=j;
      result.outcome=team_of(mover)==team_of(route.receiver)?"teammate_captured":"target_removed";
      return result;
    }
    if(!result.closed_at && !can_reach(now,route.receiver,route.flag))
      result.closed_at=j;
    if(now.board.result!=Result::Ongoing || now.board.out[route.receiver]) {
      result.cutoff=j;
      result.outcome="terminated";
      return result;
    }
  }
  return result;
}

static bool strictly_between(int from,int via,int to) {
  return std::min(from,to)<via && via<std::max(from,to);
}

static bool rail_between(int from,int via,int to) {
  const auto& board=Board::get();
  const auto& origin=board.node(from);
  const auto& middle=board.node(via);
  const auto& target=board.node(to);
  if(!origin.rail || !middle.rail || !target.rail) return false;
  return (origin.gr==middle.gr && middle.gr==target.gr && strictly_between(origin.gc,middle.gc,target.gc)) ||
         (origin.gc==middle.gc && middle.gc==target.gc && strictly_between(origin.gr,middle.gr,target.gr));
}

static void contact_routes(const Frame& frame,int receiver,int viewer,int origin,bool public_only,
                           std::vector<Move>& moves) {
  moves.clear();
  if(frame.board.seat_at[origin]!=receiver || frame.board.result!=Result::Ongoing) return;
  auto view=observe(frame.board,viewer);
  if(public_only && view.rank_at(origin)==Unknown) {
    const auto& pos=Board::get().node(origin);
    if(!frame.moved[origin] && (pos.seat!=receiver || deployment_rank_allowed(origin,Mine) ||
                               deployment_rank_allowed(origin,Flag))) return;
    view.ranks[origin]=Platoon;
  }
  gen_piece_moves(view,receiver,origin,moves);
}

static Reply relay_reply(const std::vector<Frame>& trace,int start,const Route& relay) {
  const auto& opened=trace[start];
  const int owner=opened.board.seat_at[relay.flag], stamp=opened.moved[relay.flag];
  const Move contact{static_cast<uint8_t>(relay.origin),static_cast<uint8_t>(relay.flag)};
  const auto still_target=[&](const Frame& f) {
    return f.board.seat_at[relay.flag]==owner && f.moved[relay.flag]==stamp;
  };
  std::vector<Move> options;
  const auto here=[&](const Frame& f) {
    if(!still_target(f)) return false;
    contact_routes(f,relay.receiver,relay.receiver,relay.origin,false,options);
    return std::find(options.begin(),options.end(),contact)!=options.end();
  };
  Reply reply;
  for(int j=start+1;j<=start+8 && static_cast<size_t>(j)<trace.size();++j) {
    const auto& now=trace[j];
    if(now.action.seat==relay.receiver) {
      reply.next=j;
      if(!here(trace[j-1])) reply.outcome="closed";
      else if(!(now.action.move==contact)) reply.outcome="available_unused";
      else if(now.action.outcome==Outcome::AttackerWins) reply.outcome="attacker_won";
      else if(now.action.outcome==Outcome::DefenderWins) reply.outcome="attacker_lost";
      else reply.outcome="mutual";
      return reply;
    }
    reply.foes+=team_of(now.action.seat)!=team_of(relay.receiver);
    if(!reply.closed_at && !here(now)) reply.closed_at=j;
    if(now.board.result!=Result::Ongoing || now.board.out[relay.receiver]) {
      reply.cutoff=j;
      reply.outcome="terminated";
      return reply;
    }
  }
  return reply;
}

static void inspect_relays(const std::string& key,const std::vector<Frame>& trace,int i) {
  const auto& before=trace[i-1];
  const auto& opened=trace[i];
  const int giver=opened.action.seat, receiver=(giver+2)%kSeats;
  const int vacated=opened.action.move.from;
  if(!Board::get().node(vacated).rail || opened.board.out[receiver]) return;
  std::vector<Move> visible, earlier;
  for(int origin=0;origin<kNodes;++origin) {
    if(opened.board.seat_at[origin]!=receiver) continue;
    contact_routes(opened,receiver,giver,origin,true,visible);
    if(visible.empty()) continue;
    contact_routes(before,receiver,receiver,origin,false,earlier);
    for(const auto contact:visible) {
      const int target=opened.board.seat_at[contact.to];
      if(target<0 || team_of(target)==team_of(receiver) || !rail_between(origin,vacated,contact.to) ||
         std::find(earlier.begin(),earlier.end(),contact)!=earlier.end()) continue;
      const Route relay{receiver,origin,contact.to,opened.moved[origin],true};
      write_window(key,"relay",opened,relay,relay_reply(trace,i,relay));
    }
  }
}

static void inspect_game(const std::string& key,const std::vector<Frame>& trace) {
  for(int i=1;static_cast<size_t>(i)<trace.size();++i) {
    const auto& after=trace[i];
    const auto& before=trace[i-1];
    if(after.action.outcome!=Outcome::Move || after.controls!=before.controls) continue;
    inspect_relays(key,trace,i);
    if(std::none_of(std::begin(after.board.flag_revealed),std::end(after.board.flag_revealed),
                    [](bool revealed) {return revealed;})) continue;
    const int actor=after.action.seat;
    for(int receiver=0;receiver<kSeats;++receiver) {
      std::bitset<kNodes> previous, opened;
      std::array<Route,kNodes> witnesses{};
      for(int old=0;old<kNodes;++old)
        if(before.board.seat_at[old]==receiver) previous|=flag_routes(before,receiver,receiver,old);
      for(int origin=0;origin<kNodes;++origin) {
        if(after.board.seat_at[origin]!=receiver) continue;
        const auto new_window=flag_routes(after,receiver,receiver,origin) & ~previous;
        if(new_window.none()) continue;
        const auto open_view=flag_routes(after,receiver,actor,origin);
        for(int flag=0;flag<kNodes;++flag) if(new_window[flag]) {
          if(!opened[flag] || (!witnesses[flag].opener_can_see && open_view[flag]))
            witnesses[flag]={receiver,origin,flag,after.moved[origin],open_view[flag]};
          opened.set(flag);
        }
      }
      const char* kind=receiver==actor ? "self" : team_of(receiver)==team_of(actor) ? "partner" : "enemy";
      for(int flag=0;flag<kNodes;++flag) if(opened[flag]) {
        const auto& route=witnesses[flag];
        write_window(key,kind,after,route,observe_replies(trace,i,route));
      }
    }
  }
}

static std::vector<Frame> validated_trace(const ArchiveRows& raw,const std::string& config) {
  std::vector<Frame> accepted;
  int matches=0;
  for(bool ccw:{false,true}) {
    Rules rules;
    std::string error;
    expect_value(parse_rules(archive_review_rules(config,ccw),rules,error),"rules");
    try {
      auto trial=replay_trace(raw,rules);
      accepted=std::move(trial);
      ++matches;
    } catch(const std::runtime_error&) {}
  }
  expect_value(matches==1,"no_unique_legal_trace");
  return accepted;
}

int main(int argc,char** argv) {
  if(argc!=2) return 2;
  const auto config=read_text(argv[1]);
  std::string key,hex;
  while(std::cin>>key>>hex) {
    try {
      const ArchiveRows rows(decode_archive_tape(hex));
      const auto trace=validated_trace(rows,config);
      std::cout<<"G "<<key<<' '<<trace.size()-1<<'\n';
      inspect_game(key,trace);
    } catch(const std::runtime_error&) {
      std::cout<<"R "<<key<<'\n';
    }
  }
}
