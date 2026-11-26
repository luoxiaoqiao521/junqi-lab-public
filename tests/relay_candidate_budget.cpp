#include "../sim/archive_review.h"
#include "../sim/public_relay_screen.h"
#include "../sim/public_threat_evidence.h"
#include "../sim/relay_continuity.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <tuple>

using namespace jq;
using Clock=std::chrono::steady_clock;

struct Nomination {
  size_t index=0;
  std::optional<PublicRelayContact> contact;
  size_t scanned=0;
};

static Nomination nominate(const SeatDecision& decision,const OpeningMarks& marks,
                           const std::vector<size_t>& order) {
  Nomination nominated;
  for (const auto index : order) {
    const auto action=decision.actions[index];
    if (decision.observation.seat_at(action.to)>=0 ||
        !Board::get().node(action.from).rail) continue;
    ++nominated.scanned;
    const auto routes=public_relay_contacts(decision,marks,action);
    const auto ready=std::find_if(routes.begin(),routes.end(),[](const auto& route) {
      return route.evidence==PublicRelayEvidence::NewContact;
    });
    if (ready==routes.end()) continue;
    nominated.index=index;
    nominated.contact=*ready;
    break;
  }
  return nominated;
}

static auto evaluate(const SeatDecision& decision,const OpeningMarks& marks,
                     const Rules& rules,Move move) {
  const auto& view=decision.observation;
  auto after=view;
  if (view.seat_at(move.to)==kNone) after=project_quiet(view,move);
  const auto threat=visible_flag_threat(after,marks,rules);
  const bool wins=view.rank_at(move.to)==Flag && replay_flag_deadline(view,marks.moved,move,rules).team_win;
  return std::make_tuple(wins,-threat.forced,score_move(view,move).position);
}

static size_t select_move(const SeatDecision& decision,const OpeningMarks& marks,const Rules& rules,
                          const std::vector<size_t>& shortlist,size_t& calls) {
  size_t selected=shortlist.front();
  std::tuple<bool,int,int> best;
  calls=0;
  for (const auto index : shortlist) {
    const auto value=evaluate(decision,marks,rules,decision.actions[index]);
    if (calls==0 || value>best) {
      selected=index;
      best=value;
    }
    ++calls;
  }
  return selected;
}

struct Response { const char* label; size_t plies; };
static Response test_reply(const State& root, Move start,const Nomination& nomination,Rules rules,
                           const ShortWeights& weights,uint64_t seed) {
  if (!nomination.contact) return {"unqualified",0};
  const auto& opportunity=*nomination.contact;
  const int team=team_of(root.to_move,root.match);
  const RelayPolicy policy=[&](const SeatDecision& d,Rng& rng) {
    const auto& view=d.observation;
    if (view.team(view.viewer)==team) return short_move(view,d.actions,weights,rng);
    return d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
  };
  rules.truncation_cap=std::min(rules.truncation_cap,root.ply+12);
  const auto checkpoint=relay_checkpoint(root,start,opportunity.receiver,rules,seed,policy);
  const size_t plies=checkpoint.trace.events.size();
  const auto& state=checkpoint.trace.state;
  if (!checkpoint.can_decide()) {
    if (state.result==Result::Truncated) return {"cutoff",plies};
    if (state.result!=Result::Ongoing) return {"known_result",plies};
    return {"retired",plies};
  }
  bool target_touched=false;
  for (const auto& event : checkpoint.trace.events) target_touched|=relay_target_touched(opportunity,event.step);
  const auto reply=public_relay_reply(opportunity,seat_decision(state),target_touched);
  if (reply==PublicRelayReply::Available) return {"available",plies};
  if (reply==PublicRelayReply::Closed) return {"closed",plies};
  if (reply==PublicRelayReply::TargetChanged) return {"target_changed",plies};
  throw std::runtime_error("relay response check lost the receiver's turn");
}

static void inspect(std::string key,const State& state,const OpeningMarks& marks,
                    const Rules& rules,const ShortWeights& weights) {
  const auto d=seat_decision(state);
  if (d.actions.size()<3) {
    std::cout<<"M "<<key<<' '<<state.ply<<" fewer_than_three\n";
    return;
  }
  const auto masses=review_learned(d,weights);
  std::vector<size_t> order(d.actions.size());
  std::iota(order.begin(),order.end(),size_t{0});
  std::stable_sort(order.begin(),order.end(),[&](size_t left,size_t right){
    return masses[left]>masses[right];
  });
  const auto started=Clock::now();
  const auto proposal=nominate(d,marks,order);
  const auto scan_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-started).count();
  const std::vector<size_t> standard(order.begin(),order.begin()+3);
  auto adjusted=standard;
  const bool present=proposal.contact &&
      std::find(standard.begin(),standard.end(),proposal.index)!=standard.end();
  const bool edit=proposal.contact && !present;
  if (edit) adjusted.back()=proposal.index;
  size_t calls_a=0,calls_b=0;
  const auto start_a=Clock::now();
  const auto a=select_move(d,marks,rules,standard,calls_a);
  const auto cost_a=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start_a).count();
  const auto start_b=Clock::now();
  const auto b=select_move(d,marks,rules,adjusted,calls_b);
  const auto cost_b=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start_b).count();
  const uint64_t seed=mix_seed(std::stoull(key.substr(0,16),nullptr,16),uint64_t(state.ply));
  const auto a_reply=test_reply(state,d.actions[a],proposal,rules,weights,seed);
  const auto b_reply=test_reply(state,d.actions[b],proposal,rules,weights,seed);
  std::cout<<"R "<<key<<' '<<state.ply<<" ok "<<present<<' '<<edit<<' '<<(a!=b)<<' '
           <<a_reply.label<<' '<<b_reply.label<<' '<<a_reply.plies<<' '<<b_reply.plies<<' '
           <<calls_a<<' '<<calls_b<<' '<<proposal.scanned<<' '<<scan_ns<<' '<<cost_a<<' '<<cost_b<<'\n';
}

static int hexchar(char x) {
  if (x>='0' && x<='9') return x-'0';
  if (x>='a' && x<='f') return x-'a'+10;
  throw std::runtime_error("invalid tape hex");
}

static std::string tape_bytes(const std::string& tape) {
  expect_value(tape.size()%2==0,"invalid tape length");
  std::string input;
  for (size_t i=0;i<tape.size();i+=2)
    input+=static_cast<char>(16*hexchar(tape[i])+hexchar(tape[i+1]));
  return input;
}

static void play(ArchiveReview& game,const ArchiveRows& raw,size_t row,int sequence,const Rules& rules,
                 std::vector<OpeningMarks>& marks) {
  if (raw.byte(row)==245) {
    game.control(raw,row,sequence,rules);
    for (auto& viewer : marks)
      for (int node=0;node<kNodes;++node) if (game.state.seat_at[node]<0) {
        viewer.ranks[node]=0;
        viewer.moved[node]=false;
      }
    return;
  }
  expect_value(raw.byte(row)==95,"JGS record_type");
  const auto prior=game.state;
  game.play(raw,row,sequence,rules);
  game.frames.str("");
  for (int seat=0;seat<kSeats;++seat)
    advance_contact_ranks(marks[seat],observe(prior,seat),observe(game.state,seat),
                          game.last.move,game.last.outcome,rules);
}

static void source(const std::string& key,const std::string& tape,const ShortWeights& weights,const Rules& rules) {
  constexpr int windows[]={48,96,144};
  size_t next=0;
  const auto count=sizeof(windows)/sizeof(windows[0]);
  const char* missing="short";
  try {
    ArchiveRows raw(tape_bytes(tape));
    const int origin=raw.point(raw.row(0)+2);
    expect_value(raw.byte(raw.row(0))==95 && origin>=0,"first_move");
    const int first=Board::get().node(origin).seat;
    expect_value(first>=0,"first_seat");
    ArchiveReview game;
    init_state(game.state,raw.layouts,first,rules);
    std::vector<OpeningMarks> marks;
    for (int seat=0;seat<kSeats;++seat) marks.emplace_back(observe(game.state,seat),true);
    for (int i=0;next<count && i<raw.entries && !game.ended;++i) {
      const auto row=raw.row(i);
      if (raw.byte(row)==95 && game.state.ply==windows[next]) {
        inspect(key,game.state,marks[game.state.to_move],rules,weights);
        if (++next==count) return;
      }
      play(game,raw,row,i+1,rules,marks);
    }
    if (next==2 && game.state.ply==windows[next] && !game.ended) {
      inspect(key,game.state,marks[game.state.to_move],rules,weights);
      ++next;
    }
  } catch (const std::exception& e) {
    std::cerr<<key<<' '<<e.what()<<'\n';
    missing="prefix_or_evaluation_error";
  }
  for (;next<count;++next) std::cout<<"M "<<key<<' '<<windows[next]<<' '<<missing<<'\n';
}

int main(int argc,char** argv) {
  try {
    if (argc!=2) return 2;
    Rules rules;
    std::string error;
    if (!parse_rules(archive_review_rules(read_text(argv[1]),false),rules,error))
      throw std::runtime_error(error);
    ShortWeights weights{};
    for (size_t i=0;i<5;++i)
      if (!(std::cin>>weights[i]) || !std::isfinite(weights[i])) return 2;
    std::string key,tape;
    while (std::cin>>key>>tape) source(key,tape,weights,rules);
    return 0;
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
  }
}
