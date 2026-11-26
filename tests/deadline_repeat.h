#pragma once

struct QuietRevisit {
  std::array<uint8_t,jq::kNodes> tokens{};
  struct Visit {
    uint64_t position;
    jq::Move move;
  };
  std::array<std::vector<Visit>,jq::kSeats> seen;
  std::array<jq::Move,jq::kSeats> last{};
  int quiet_clock=0;
  int turns=0, quiet=0, reversals=0, repeat=0, persistent=0, near_draw=0;

  void begin(const jq::Observation& view) {
    for (int n=0;n<jq::kNodes;++n)
      tokens[n]=view.seat_at(n)<0?0:static_cast<uint8_t>(n+1);
    for (auto& visits:seen) visits.clear();
    last={};
  }

  uint64_t key(const jq::Observation& view,const jq::OpeningMarks& marks) const {
    auto hash=jq::fnv1a(view.seats,sizeof(view.seats));
    hash=jq::fnv1a(view.ranks,sizeof(view.ranks),hash);
    hash=jq::fnv1a(tokens.data(),tokens.size(),hash);
    hash=jq::fnv1a(marks.ranks.data(),sizeof(marks.ranks),hash);
    hash=jq::fnv1a(marks.moved.data(),sizeof(marks.moved),hash);
    const int phase=(view.ply>=64)+(view.ply>=100)+(view.ply>=140);
    return jq::fnv1a(&phase,sizeof(phase),hash);
  }

  int sightings(const jq::Observation& view,const jq::OpeningMarks& marks,jq::Move action) const {
    const auto position=key(view,marks);
    return static_cast<int>(std::count_if(seen[view.viewer].begin(),seen[view.viewer].end(),[&](const Visit& v) {
      return v.position==position && v.move==action;
    }));
  }

  void note(const jq::Observation& view,const jq::OpeningMarks& marks,jq::Move action,
            const jq::Rules& rules) {
    ++turns;
    if (view.seat_at(action.to)>=0) return;
    ++quiet;
    const auto previous=last[view.viewer];
    reversals+=action.from==previous.to && action.to==previous.from;
    const int repetitions=sightings(view,marks,action);
    repeat+=repetitions>0;
    persistent+=repetitions>=2;
    near_draw+=repetitions>=2 && draw_soon(rules);
    seen[view.viewer].push_back({key(view,marks),action});
    last[view.viewer]=action;
  }

  bool draw_soon(const jq::Rules& rules) const {
    return rules.no_progress_limit>0 && quiet_clock+2*jq::kSeats>=rules.no_progress_limit;
  }

  void advance(jq::Move action,const jq::StepInfo& event,const jq::Observation& after) {
    const bool contact=event.outcome!=jq::Outcome::Move;
    if (contact || event.eliminated_mask || event.revealed_mask || event.skipped_mask) begin(after);
    else {
      tokens[action.to]=tokens[action.from];
      tokens[action.from]=0;
    }
    quiet_clock=contact?0:quiet_clock+1;
  }
};
