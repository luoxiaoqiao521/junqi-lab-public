#pragma once
#include <iomanip>

// This experiment applies a learned state-action interaction at the first
// eligible decision. The original five coefficients and continuations stay frozen.
static double clock_action(const jq::Observation& view,jq::Move action,const jq::Rules& rules) {
  return view.seat_at(action.to)>=0 ? double(view.quiet_clock)/rules.no_progress_limit : 0.;
}

static void clock_learning_fixtures(const jq::Rules& rules) {
  using namespace jq;
  for (int seat=0;seat<kSeats;++seat) for (int col:{1,3}) {
    auto near=selfplay_fixture();
    const auto& board=Board::get();
    const int enemy=next_seat(seat,rules);
    near.out[0]=near.out[1]=near.out[2]=near.out[3]=true;
    std::fill(std::begin(near.alive),std::end(near.alive),false);
    std::fill(std::begin(near.id_at),std::end(near.id_at),int8_t(kNone));
    std::fill(std::begin(near.seat_at),std::end(near.seat_at),int8_t(kNone));
    std::fill(std::begin(near.rank_at),std::end(near.rank_at),int8_t(Unknown));
    near.out[seat]=near.out[enemy]=false;
    near.to_move=static_cast<int8_t>(seat);
    near.ply=80;
    near.no_progress=rules.no_progress_limit-1;
    const auto from=board.seat_node(enemy,4,col);
    const auto flag=board.seat_node(enemy,5,col);
    const auto safe=board.seat_node(enemy,4,2);
    place_single_case_piece(near,seat*kPiecesPerSeat,board.seat_node(seat,5,1),Flag);
    place_single_case_piece(near,seat*kPiecesPerSeat+1,from,Division);
    place_single_case_piece(near,enemy*kPiecesPerSeat,flag,Flag);
    place_single_case_piece(near,enemy*kPiecesPerSeat+1,board.seat_node(enemy,0,0),Marshal);
    const Move take{static_cast<uint8_t>(from),static_cast<uint8_t>(flag)};
    const Move quiet{static_cast<uint8_t>(from),static_cast<uint8_t>(safe)};
    auto early=near;
    early.no_progress-=4;
    const auto a=seat_decision(early),b=seat_decision(near);
    single_case_check(a.actions==b.actions && b.observation.rank_at(flag)==Unknown &&
                      b.observation.quiet_clock==a.observation.quiet_clock+4 &&
                      short_features(a.observation,take)==short_features(b.observation,take) &&
                      clock_action(a.observation,take,rules)<clock_action(b.observation,take,rules) &&
                      clock_action(a.observation,quiet,rules)==clock_action(b.observation,quiet,rules),
                      "clock must interact with the action without revealing the flag");
    single_case_check(is_legal(early,quiet) && is_legal(near,take),"clock pair needs legal moves");
    step(early,quiet,rules);
    auto draw=near;
    step(draw,quiet,rules);
    const auto ending=step(near,take,rules);
    single_case_check(early.result==Result::Ongoing && draw.result==Result::Draw &&
                      review_outcome(near.result,seat)==2 && ending.exit_reason[enemy]==ExitReason::FlagRemoved,
                      "quiet draw, quiet continuation, and hidden capture must remain separate");
  }
}

struct ClockResult { int score, plies, reason; uint64_t trace; };

static jq::Move clock_baseline(const jq::SeatDecision& d,int learner,int table,const jq::ShortWeights& parent,
                              const jq::ShortWeights& history,jq::Rng& rng) {
  using namespace jq;
  const int seat=d.observation.viewer;
  if (seat==learner) return short_move(d.observation,d.actions,parent,rng);
  if ((seat+2)%kSeats==learner || table==1) return short_move(d.observation,d.actions,history,rng);
  return d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
}

static ClockResult clock_finish(jq::State state,std::array<jq::Rng,jq::kSeats> streams,jq::Move first,
    int learner,int table,const jq::Rules& rules,const jq::ShortWeights& parent,const jq::ShortWeights& history) {
  using namespace jq;
  auto event=step(state,first,rules);
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    event=step(state,clock_baseline(d,learner,table,parent,history,streams[state.to_move]),rules);
  }
  return {review_outcome(state.result,learner),state.ply,static_cast<int>(event.end_reason),state_hash(state)};
}

static void clock_result(ClockResult result) {
  std::cout<<' '<<result.score<<' '<<result.plies<<' '<<result.reason<<' '<<result.trace;
}

static void clock_trial(uint64_t seed,int learner,int table,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& history,std::optional<double> fitted,bool contact_control) {
  using namespace jq;
  auto state=opening(seed,rules);
  auto streams=review_streams(seed);
  EndReason end=EndReason::None;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto& view=d.observation;
    const auto base=clock_baseline(d,learner,table,parent,history,streams[view.viewer]);
    if (view.viewer==learner && view.quiet_clock>=rules.no_progress_limit-10) {
      const bool contact=view.seat_at(base.to)>=0;
      const auto values=short_action_masses(view,d.actions,parent);
      size_t best=d.actions.size();
      for (size_t i=0;i<d.actions.size();++i) {
        if ((view.seat_at(d.actions[i].to)>=0)==contact) continue;
        if (best==d.actions.size() || values[i]>values[best]) best=i;
      }
      if (best!=d.actions.size()) {
        const auto delta=std::log(values[best]/values[std::find(d.actions.begin(),d.actions.end(),base)-d.actions.begin()]);
        const auto action=d.actions[best];
        const auto interaction=clock_action(view,action,rules)-clock_action(view,base,rules);
        std::cout<<seed<<' '<<learner<<' '<<table<<' '<<view.ply<<' '<<view.quiet_clock<<' '
                 <<std::setprecision(17)<<delta<<' '<<interaction;
        if (fitted) {
          const double feature=contact_control?std::copysign(1.,interaction):interaction;
          const bool changed=delta+*fitted*feature>0;
          std::cout<<' '<<changed;
          clock_result(clock_finish(state,streams,changed?action:base,learner,table,rules,parent,history));
        } else {
          clock_result(clock_finish(state,streams,base,learner,table,rules,parent,history));
          clock_result(clock_finish(state,streams,action,learner,table,rules,parent,history));
        }
        std::cout<<'\n';
        return;
      }
    }
    end=step(state,base,rules).end_reason;
  }
  const ClockResult outcome{review_outcome(state.result,learner),state.ply,static_cast<int>(end),state_hash(state)};
  std::cout<<seed<<' '<<learner<<' '<<table<<" -1 0 0 0";
  if (fitted) std::cout<<" 0";
  clock_result(outcome);
  if (!fitted) clock_result(outcome);
  std::cout<<'\n';
}
