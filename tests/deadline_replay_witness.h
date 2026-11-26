#pragma once
#include <string_view>

struct PrefixReplayWork {
  int moves=0, failed_ply=-1;
  bool collision_mismatch=false;
  const char* reason="ok";
};

static std::optional<State> replay_prefix_world(const ArchiveRows& raw,const int8_t (&layouts)[kSeats][25],
    const Rules& rules,const std::vector<Observation>& history,PrefixReplayWork* work=nullptr) {
  ArchiveReview replay;
  const int viewer=history.front().viewer;
  init_state(replay.state,layouts,archive_first_seat(raw),rules);
  if (!same_observation(history.front(),observe(replay.state,viewer))) {
    if (work) { work->failed_ply=0; work->reason="opening-view"; }
    return std::nullopt;
  }
  try {
    for (int i=0;replay.state.ply+1<int(history.size()) && i<raw.entries;) {
      const auto move=raw.row(i++);
      if (work) ++work->moves;
      replay.play(raw,move,i,rules);
      while (i<raw.entries && raw.byte(raw.row(i))==245) {
        const auto control=raw.row(i++);
        replay.control(raw,control,i,rules);
      }
      replay.frames.str("");
      if (!same_observation(history.at(replay.state.ply),observe(replay.state,viewer))) {
        if (work) { work->failed_ply=replay.state.ply; work->reason="observer-event"; }
        return std::nullopt;
      }
    }
  } catch (const std::runtime_error& error) {
    if (work) {
      work->failed_ply=work->moves;
      work->collision_mismatch=std::string_view(error.what())=="JGS collision differs from core";
      const std::string_view message=error.what();
      work->reason=work->collision_mismatch?"collision":message.find("flag")!=message.npos?"flag-event":
                   message.find("legal")!=message.npos?"route":message.find("seat")!=message.npos?"seat-control":"archive-event";
    }
    return std::nullopt;
  }
  if (replay.state.ply+1!=int(history.size())) {
    if (work) { work->failed_ply=replay.state.ply; work->reason="prefix-length"; }
    return std::nullopt;
  }
  return replay.state;
}

static std::vector<State> replay_prefix_witnesses(const ArchiveRows& raw,const Rules& rules,
    const std::vector<Observation>& history,const SeatDecision& d,const std::string& key,int phase,uint64_t seed) {
  single_case_check(replay_prefix_world(raw,raw.layouts,rules,history).has_value(),"original QQ prefix did not replay");
  struct Swap { int seat,a,b; };
  std::vector<Swap> pairs;
  for (int seat=0;seat<kSeats;++seat) if (seat!=d.observation.viewer)
    for (int a=0;a<25;++a) for (int b=a+1;b<25;++b)
      if (raw.layouts[seat][a]!=raw.layouts[seat][b]) pairs.push_back({seat,a,b});
  Rng rng(seed);
  shuffle_slots(pairs,rng);
  std::vector<State> worlds;
  int attempts=0,legal_deployment=0,valid_prefix=0;
  while (attempts<96 && attempts<int(pairs.size()) && worlds.size()<6) {
    int8_t layouts[kSeats][25];
    for (int seat=0;seat<kSeats;++seat) std::copy_n(raw.layouts[seat],25,layouts[seat]);
    const int swaps=attempts%3+1;
    for (int i=0;i<swaps;++i) {
      const auto [seat,a,b]=pairs[(attempts+i*97)%pairs.size()];
      std::swap(layouts[seat][a],layouts[seat][b]);
    }
    ++attempts;
    bool legal=true;
    for (const auto& layout:layouts) legal&=valid_layout(layout);
    if (!legal) continue;
    ++legal_deployment;
    const auto world=replay_prefix_world(raw,layouts,rules,history);
    if (!world) continue;
    ++valid_prefix;
    single_case_check(seat_decision(*world).actions==d.actions,"witness changed legal actions");
    if (std::none_of(worlds.begin(),worlds.end(),[&](const State& other) {
          return std::equal(std::begin(other.rank_at),std::end(other.rank_at),std::begin(world->rank_at));
        })) worlds.push_back(*world);
  }
  std::cout<<"W "<<key<<' '<<phase<<' '<<attempts<<' '<<legal_deployment<<' '<<valid_prefix<<' '<<worlds.size()<<'\n';
  return worlds;
}
