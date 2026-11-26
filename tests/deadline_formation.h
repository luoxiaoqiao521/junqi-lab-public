#pragma once
#include "../sim/formation_pair.h"
#include <set>

static int formation_flag_slot(const int8_t (&layout)[kPiecesPerSeat]) {
  return static_cast<int>(std::find(std::begin(layout),std::end(layout),Flag)-std::begin(layout));
}

static void formation_rule_cases(const Rules& rules) {
  for (int rotation=0;rotation<kSeats;++rotation) {
    const auto node=[&](int n){return static_cast<uint8_t>(rotate_panel_node(n,rotation));};
    auto base=selfplay_fixture();
    place_single_case_piece(base,1,50,Engineer);
    place_single_case_piece(base,26,51,Mine);
    place_single_case_piece(base,27,55,Mine);
    place_single_case_piece(base,28,57,Mine);
    place_single_case_piece(base,29,30,Platoon);
    base=rotate_panel(base,rotation);
    const Move breach{node(50),node(51)}, wait{node(30),node(31)}, capture{node(51),node(56)};
    auto marshal=base;
    marshal.rank_of[rotation*25+1]=marshal.rank_at[node(50)]=Marshal;
    const auto dark=compare_visibility(marshal,IdentityVisibility::OwnAndPublic);
    const auto full=compare_visibility(marshal,IdentityVisibility::AllIdentities);
    single_case_check(dark.actions==full.actions && is_legal(marshal,breach) && !is_legal(marshal,{node(50),node(56)}),
                      "deployment changed the move contract or allowed bypass into HQ");
    const int legacy=DecisionScorer(full.observation,rules,DecisionKind::Heuristic).score(breach).objective;
    const int aware=DecisionScorer(full.observation,rules,DecisionKind::FlagRoute).score(breach).objective;
    const auto loss=step(marshal,breach,rules);
    single_case_check(loss.outcome==Outcome::DefenderWins && legacy>0 && aware<0,
                      "the visible marshal-on-mine regression no longer discriminates scoring");

    auto bomb=base;
    bomb.rank_of[rotation*25+1]=bomb.rank_at[node(50)]=Bomb;
    const auto blast=step(bomb,breach,rules);
    single_case_check(blast.outcome==Outcome::BothRemoved && bomb.id_at[node(51)]==kNone,
                      "bomb/mine did not use the registered training profile");
    single_case_check(is_legal(base,breach) && step(base,breach,rules).outcome==Outcome::AttackerWins &&
                      is_legal(base,wait),"engineer breach or actual reply is not legal");
    step(base,wait,rules);
    single_case_check(base.to_move==rotation && is_legal(base,capture) &&
                      base.rank_at[node(55)]==Mine && base.rank_at[node(57)]==Mine,
                      "the remaining mines should not prohibit a capture through the opened entrance");
    auto bomb_flag=base;
    bomb_flag.rank_of[rotation*25+1]=bomb_flag.rank_at[node(51)]=Bomb;
    const auto exploded_flag=step(bomb_flag,capture,rules);
    const auto flag_event=step(base,capture,rules);
    const int defender=(rotation+1)%kSeats;
    single_case_check(flag_event.exit_reason[defender]==ExitReason::FlagRemoved &&
                      exploded_flag.exit_reason[defender]==ExitReason::FlagRemoved &&
                      exploded_flag.outcome==Outcome::BothRemoved && bomb_flag.id_at[node(56)]==kNone &&
                      base.id_at[node(55)]==kNone && base.id_at[node(57)]==kNone &&
                      base.rank_at[node(56)]==Engineer && !can_leave_node(Engineer,node(56)) &&
                      team_score(base.result,rotation)==1.0 && team_score(bomb_flag.result,rotation)==1.0,
                      "flag capture, cleanup, HQ lock or terminal priority failed");
    std::cout<<"R "<<rotation<<' '<<legacy<<' '<<aware<<'\n';
  }
}

static void formation_training_seeds(uint64_t seed,int count) {
  std::set<std::array<int8_t,kPiecesPerSeat>> unique;
  std::set<std::array<int8_t,kPieces>> tables;
  std::array<int,4> cover{};
  for (int i=0;i<count;++i) {
    Rng layouts_rng(seed+i);
    std::array<int8_t,kPieces> table{};
    for (int seat=0;seat<kSeats;++seat) {
      int8_t layout[kPiecesPerSeat];
      generate_layout(layout,layouts_rng);
      single_case_check(valid_layout(layout),"training produced illegal layout");
      std::copy(std::begin(layout),std::end(layout),table.begin()+seat*25);
      std::array<int8_t,kPiecesPerSeat> local{};
      std::copy(std::begin(layout),std::end(layout),local.begin());
      unique.insert(local);
      const auto& entries=Board::get().road(layout_slot_node(seat,formation_flag_slot(layout)));
      int mines=0;
      for (int slot=0;slot<kPiecesPerSeat;++slot)
        if (layout[slot]==Mine && std::find(entries.begin(),entries.end(),layout_slot_node(seat,slot))!=entries.end()) ++mines;
      ++cover.at(mines);
    }
    tables.insert(table);
  }
  std::cout<<"S "<<count<<' '<<tables.size()<<' '<<unique.size();
  for (int v:cover) std::cout<<' '<<v;
  std::cout<<'\n';
}

// 0 uses the old heuristic. 1 changes known mine/bomb collisions only. 2 is the current route policy.
static Move formation_move(const State& state,const Rules& rules,int policy,Rng& rng) {
  const auto d=compare_visibility(state,IdentityVisibility::AllIdentities);
  const auto kind=policy==2?DecisionKind::FlagRoute:policy==3?DecisionKind::CollisionHeuristic:DecisionKind::Heuristic;
  const DecisionScorer scorer(d.observation,rules,kind);
  Move move;
  const auto rank_aware=[&](Move m) {
    auto score=scorer.score(m);
    const int attacker=d.observation.rank_at(m.from), target=d.observation.rank_at(m.to);
    if (target!=Unknown && target!=Flag && (attacker==Bomb || target==Bomb || target==Mine)) {
      const auto outcome=resolve(attacker,target,rules);
      score.objective=outcome==Outcome::AttackerWins?60:outcome==Outcome::DefenderWins?-60:0;
    }
    return score.total();
  };
  const bool picked=policy==1 ? preferred_move(d.actions,rng,move,rank_aware) : scorer.choose(rng,move);
  single_case_check(picked && is_legal(state,move),"formation policy returned an illegal move");
  return move;
}

static void formation_episode(State state,const Rules& rules,uint64_t seed,int learner,int shape,int own,int foe) {
  auto streams=review_streams(seed);
  int own_flag=0, own_retired=0, first_mine=0, enemy_flag=0;
  while (state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const int policy=actor==learner ? own : team_of(actor)==team_of(learner) ? 2 : foe;
    const auto move=formation_move(state,rules,policy,streams[actor]);
    const bool mine=state.seat_at[move.to]==learner && state.rank_at[move.to]==Mine;
    const auto event=step(state,move,rules);
    if (!first_mine && mine && (event.outcome==Outcome::AttackerWins || event.outcome==Outcome::BothRemoved))
      first_mine=state.ply;
    if (event.exit_reason[learner]!=ExitReason::None) own_retired=state.ply;
    if (event.exit_reason[learner]==ExitReason::FlagRemoved) own_flag=state.ply;
    if (!enemy_flag && (event.exit_reason[(learner+1)%4]==ExitReason::FlagRemoved ||
                        event.exit_reason[(learner+3)%4]==ExitReason::FlagRemoved)) enemy_flag=state.ply;
  }
  const auto score=team_score(state.result,learner);
  std::cout<<"G "<<seed<<' '<<learner<<' '<<shape<<' '<<own<<' '<<foe<<' '
           <<(score?int(*score*2):-1)<<' '<<state.ply<<' '<<own_flag<<' '<<own_retired<<' '
           <<first_mine<<' '<<enemy_flag<<'\n';
}

static std::array<int,4> formation_swap_slots(const int8_t (&layout)[kPiecesPerSeat],int seat) {
  const auto& board=Board::get();
  const int column=board.node(layout_slot_node(seat,formation_flag_slot(layout))).c;
  const std::array<int,3> entries{board.seat_node(seat,4,column),board.seat_node(seat,5,2),
                                board.seat_node(seat,5,column==1?0:4)};
  std::array<int,4> slots{kNone,kNone,kNone,kNone};
  for (int slot=0;slot<kPiecesPerSeat;++slot) {
    const int point=layout_slot_node(seat,slot);
    const auto index=std::find(entries.begin(),entries.end(),point);
    if (index!=entries.end()) {
      single_case_check(layout[slot]==Mine,"missing mine at a flag entrance");
      slots[index-entries.begin()]=slot;
    } else if (slots[3]==kNone && is_army(layout[slot]) && deployment_rank_allowed(point,Mine) &&
               board.node(point).kind!=NodeKind::HQ) slots[3]=slot;
  }
  single_case_check(std::all_of(slots.begin(),slots.end(),[](int s){return s!=kNone;}),"incomplete matched formation");
  return slots;
}

static void run_formation_matrix(const Rules& rules,uint64_t seed,int count,bool collision_impact=false) {
  for (int index=0;index<count;++index) {
    Rng rng(seed+index);
    int8_t layouts[kSeats][kPiecesPerSeat];
    for (auto& layout:layouts) generate_layout(layout,rng);
    const int learner=index%kSeats, first=(learner+(index/kSeats)%kSeats)%kSeats;
    cover_flag_entrances(layouts[learner],learner);
    const auto slots=formation_swap_slots(layouts[learner],learner);
    State reference;
    init_state(reference,layouts,first,rules);
    const int flag=formation_flag_slot(layouts[learner]);
    const auto flag_col=Board::get().node(layout_slot_node(learner,flag)).c;
    for (int shape=0;shape<4;++shape) {
      if (collision_impact && shape!=0 && shape!=2) continue;
      if (shape) std::swap(layouts[learner][slots[shape-1]],layouts[learner][slots[3]]);
      State state;
      init_state(state,layouts,first,rules);
      single_case_check(valid_layout(layouts[learner]) && opening_mine_neighbors(state)[learner]==(shape?2:3) &&
                        state.to_move==reference.to_move && formation_flag_slot(layouts[learner])==flag,
                        "formation changed more than the controlled mine swap");
      for (int seat=0;seat<kSeats;++seat) if (seat!=learner)
        single_case_check(same_seat_view(reference,state,seat),"mine swap exposed hidden ranks in four-dark");
      std::cout<<"N "<<seed+index<<' '<<learner<<' '<<first<<' '<<int(flag_col)<<' '<<shape<<' '<<state_hash(state)<<'\n';
      if (collision_impact) {
        for (int own:{0,3}) formation_episode(state,rules,seed+index,learner,shape,own,2);
      } else {
        for (int own:{0,1,2}) for (int foe:{0,2}) formation_episode(state,rules,seed+index,learner,shape,own,foe);
      }
      if (shape) std::swap(layouts[learner][slots[shape-1]],layouts[learner][slots[3]]);
    }
  }
}
