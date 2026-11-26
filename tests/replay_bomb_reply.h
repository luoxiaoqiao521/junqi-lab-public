#pragma once
#include <functional>

struct BombWindow { int ply, attacker, responder, ally; };
using BombVisit=std::function<void(const BombWindow&,const SeatDecision&,int)>;

struct BombReplies {
  std::vector<BombWindow> pending;
  std::ostringstream rows;

  void record(const BombWindow& win,int status,int now,int available=0,int offers=0,
              int moved_elsewhere=0,int parent_mass=0,int child_mass=0) {
    rows<<win.ply<<' '<<win.ally<<' '<<status<<' '<<now-win.ply<<' '<<available<<' '<<offers
        <<' '<<moved_elsewhere<<' '<<parent_mass<<' '<<child_mass<<'\n';
  }

  void take_turn(const State& state,Move actual,const Policies& models,const BombVisit& visit={}) {
    const auto d=seat_decision(state);
    auto w=pending.begin();
    while (w!=pending.end()) {
      if (!state.alive[w->attacker]) record(*w,0,state.ply);
      else if (state.out[w->responder]) record(*w,1,state.ply);
      else if (w->responder==state.to_move) {
        const auto target=state.node_of[w->attacker];
        if (turn(*w,d,target,actual,models) && visit) visit(*w,d,target);
      }
      else { ++w; continue; }
      w=pending.erase(w);
    }
  }

  bool turn(const BombWindow& w,const SeatDecision& d,int target,Move actual,const Policies& models) {
    const auto& view=d.observation;
    const auto parent=review_learned(d,models.parent), child=review_learned(d,models.trained);
    int available=0,offers=0;
    double parent_mass=0,child_mass=0;
    for (int n=0;n<kNodes;++n)
      available+=view.seat_at(n)==view.viewer && view.rank_at(n)==Bomb && can_leave_node(Bomb,n);
    for (size_t i=0;i<d.actions.size();++i) {
      const auto a=d.actions[i];
      if (view.rank_at(a.from)!=Bomb || a.to!=target) continue;
      ++offers;
      parent_mass+=parent[i]; child_mass+=child[i];
    }
    const bool bomb=view.rank_at(actual.from)==Bomb;
    const bool taken=bomb && actual.to==target;
    expect_value(!taken || offers,"direct bomb reply absent from legal moves");
    record(w,taken?5:offers?4:3,view.ply,available,offers,bomb && actual.to!=target,
           static_cast<int>(std::llround(parent_mass*1000000)),static_cast<int>(std::llround(child_mass*1000000)));
    return offers>0;
  }

  void after(const State& before,Move move,Outcome outcome) {
    if (outcome!=Outcome::AttackerWins) return;
    const int victim=before.seat_at[move.to],attacker=before.id_at[move.from];
    if (victim<0 || team_of(victim)==team_of(before.to_move)) return;
    pending.push_back({before.ply+1,attacker,victim,0});
    pending.push_back({before.ply+1,attacker,(victim+2)%kSeats,1});
  }

  void finish(const State& state,const std::string& key,int source) {
    for (const auto& w:pending) {
      const int status=!state.alive[w.attacker]?0:state.out[w.responder]?1:2;
      record(w,status,state.ply);
    }
    std::istringstream input(rows.str());
    std::string line;
    while (std::getline(input,line)) std::cout<<"B "<<key<<' '<<source<<' '<<line<<'\n';
    std::cout<<"G "<<key<<' '<<source<<' '<<state.ply<<' '<<static_cast<int>(state.result)<<'\n';
  }
};

static void simulated_bomb_replies(const ArchiveRows& raw,const std::string& key,const Rules& rules,
                                   const Policies& models,int source) {
  State state;
  init_state(state,raw.layouts,archive_first_seat(raw),rules);
  auto streams=review_streams(std::stoull(key.substr(0,16),nullptr,16));
  BombReplies replies;
  const auto& weights=source==1 ? models.parent : models.trained;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    const auto move=short_move(d.observation,d.actions,weights,streams[state.to_move]);
    replies.take_turn(state,move,models);
    const auto before=state;
    const auto event=step(state,move,rules);
    replies.after(before,move,event.outcome);
  }
  replies.finish(state,key,source);
}
