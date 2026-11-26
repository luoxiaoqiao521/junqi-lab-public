#include "state.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "layout.h"
#include "rng.h"

namespace jq {

namespace {

void remove_piece(State& st, int id) {
  int n = st.node_of[id];
  st.alive[id] = false;
  st.seat_at[n] = kNone;
  st.rank_at[n] = Unknown;
  st.id_at[n] = kNone;
}

void place_piece(State& st, int id, int n) {
  st.alive[id] = true;
  st.node_of[id] = static_cast<uint8_t>(n);
  st.seat_at[n] = static_cast<int8_t>(id / kPiecesPerSeat);
  st.rank_at[n] = st.rank_of[id];
  st.id_at[n] = static_cast<int8_t>(id);
}

void eliminate(State& st, int seat, StepInfo& info, ExitReason reason) {
  st.out[seat] = true;
  info.eliminated_mask |= static_cast<uint8_t>(1u << seat);
  info.exit_reason[seat] = reason;
  for (int k = 0; k < kPiecesPerSeat; ++k) {
    int id = seat * kPiecesPerSeat + k;
    if (st.alive[id]) remove_piece(st, id);
  }
}

bool team_lost(const State& st, int team) {
  for (int seat = 0; seat < kSeats; ++seat)
    if (!st.out[seat] && team_of(seat, st.match) == team) return false;
  return true;
}

void check_involved_no_move(State& st, int defender, StepInfo& info) {
  std::vector<Move> moves;
  const int seats[] = {defender, info.seat};
  for (int seat : seats) {
    if (seat == kNone || st.out[seat]) continue;
    gen_moves(StateView{st}, seat, moves);
    if (!moves.empty()) continue;
    eliminate(st, seat, info, ExitReason::NoLegalMove);
    if (team_lost(st, team_of(seat, st.match))) return;
  }
}

bool has_movable_piece(const State& state,int seat) {
  const int first=seat*kPiecesPerSeat;
  for(int id=first;id<first+kPiecesPerSeat;++id)
    if(state.alive[id] && can_leave_node(state.rank_of[id],state.node_of[id])) return true;
  return false;
}

void retire_exhausted(State& state,int defender,StepInfo& info) {
  for(int seat:{defender,int(info.seat)})
    if(seat!=kNone && !state.out[seat] && !has_movable_piece(state,seat))
      eliminate(state,seat,info,ExitReason::MovablePiecesExhausted);
}

void on_removed(State& st, int id, const Rules& rules, StepInfo& info) {
  int seat = id / kPiecesPerSeat;
  if (st.rank_of[id] == Marshal && rules.marshal_reveals_flag && !st.flag_revealed[seat]) {
    st.flag_revealed[seat] = true;
    info.revealed_mask |= static_cast<uint8_t>(1u << seat);
  }
}

// Applies the collision at m and returns whose flag (if any) was removed.
int collide(State& st, Move m, const Rules& rules, StepInfo& info) {
  int a = st.id_at[m.from], d = st.id_at[m.to];
  info.outcome = resolve(st.rank_of[a], st.rank_of[d], rules);
  bool att_dies = info.outcome == Outcome::DefenderWins || info.outcome == Outcome::BothRemoved;
  bool def_dies = info.outcome == Outcome::AttackerWins || info.outcome == Outcome::BothRemoved;
  int flag_owner = (def_dies && st.rank_of[d] == Flag) ? d / kPiecesPerSeat : kNone;
  if (def_dies) { remove_piece(st, d); on_removed(st, d, rules, info); }
  remove_piece(st, a);
  if (att_dies) on_removed(st, a, rules, info);
  else place_piece(st, a, m.to);
  bool reset = !rules.reset_on_capture_only || def_dies;
  st.no_progress = reset ? 0 : st.no_progress + 1;
  return flag_owner;
}

// Returns whether the next decision has a legal move.
bool advance_turn(State& st, const Rules& rules, StepInfo& info) {
  std::vector<Move> buf;
  int s = st.to_move;
  const int limit = rules.eliminate_exhausted_movers ? kSeats : 2 * kSeats;
  for (int guard = 0; guard < limit; ++guard) {
    s = next_seat(s, rules);
    if (st.out[s]) continue;
    gen_moves(StateView{st}, s, buf);
    if (!buf.empty()) { st.to_move = static_cast<int8_t>(s); return true; }
    if (rules.eliminate_exhausted_movers) {
      info.skipped_mask |= static_cast<uint8_t>(1u << s);
      continue;
    }
    eliminate(st, s, info, ExitReason::NoLegalMove);
    if (team_lost(st, team_of(s, st.match))) return false;
  }
  return false;
}

int move_pieces(State& st, Move move, const Rules& rules, StepInfo& info) {
  ++st.ply;
  if (st.id_at[move.to] != kNone) return collide(st,move,rules,info);
  const int id=st.id_at[move.from];
  remove_piece(st,id);
  place_piece(st,id,move.to);
  ++st.no_progress;
  return kNone;
}

}  // namespace

void init_state(State& st, const int8_t layouts[kSeats][kPiecesPerSeat], int first_seat,
                const Rules&, MatchKind match) {
  if (!participates(match, first_seat)) throw std::invalid_argument("inactive first seat");
  std::memset(&st, 0, sizeof(st));
  st.match = match;
  std::fill(std::begin(st.seat_at), std::end(st.seat_at), int8_t(kNone));
  std::fill(std::begin(st.rank_at), std::end(st.rank_at), int8_t(Unknown));
  std::fill(std::begin(st.id_at), std::end(st.id_at), int8_t(kNone));
  std::fill(std::begin(st.rank_of), std::end(st.rank_of), int8_t(Unknown));
  for (int s = 0; s < kSeats; ++s) {
    st.out[s] = !participates(match, s);
    if (st.out[s]) continue;
    for (int k = 0; k < kPiecesPerSeat; ++k) {
      int id = s * kPiecesPerSeat + k;
      st.rank_of[id] = layouts[s][k];
      st.alive[id] = true;
      place_piece(st, id, layout_slot_node(s, k));
    }
  }
  st.to_move = static_cast<int8_t>(first_seat);
  st.result = Result::Ongoing;
}

void legal_moves(const State& st, std::vector<Move>& out) {
  out.clear();
  if (st.result == Result::Ongoing) gen_moves(StateView{st}, st.to_move, out);
}

bool is_legal(const State& st, Move m) {
  std::vector<Move> ms;
  legal_moves(st, ms);
  return std::find(ms.begin(), ms.end(), m) != ms.end();
}

StepInfo step(State& st, Move m, const Rules& rules) {
  StepInfo info{m, st.to_move, Outcome::Move, 0, 0};
  const int defender = st.seat_at[m.to];
  const int flag_owner = move_pieces(st,m,rules,info);
  if (flag_owner != kNone && rules.flag_eliminates) eliminate(st, flag_owner, info, ExitReason::FlagRemoved);
  if (!team_lost(st, 0) && !team_lost(st, 1)) {
    if (rules.eliminate_exhausted_movers) retire_exhausted(st, defender, info);
    else if (rules.check_involved_no_move) check_involved_no_move(st, defender, info);
  }
  const bool next_ready=!team_lost(st, 0) && !team_lost(st, 1) && advance_turn(st, rules, info);
  const bool team0_lost=team_lost(st, 0),team1_lost=team_lost(st, 1);
  const bool win = team0_lost || team1_lost;
  const bool draw = rules.no_progress_limit > 0 && st.no_progress >= rules.no_progress_limit;
  if (rules.eliminate_exhausted_movers && team0_lost && team1_lost) {
    st.result = Result::Truncated;
    info.end_reason = EndReason::SimultaneousExhaustion;
  } else if (win && !(draw && rules.draw_before_win)) {
    st.result = team0_lost ? Result::Team1Wins : Result::Team0Wins;
    info.end_reason = EndReason::TeamEliminated;
  } else if (draw) {
    st.result = Result::Draw;
    info.end_reason = EndReason::NoProgressDraw;
  } else if (st.ply >= rules.truncation_cap || (rules.eliminate_exhausted_movers && !next_ready)) {
    st.result = Result::Truncated;
    info.end_reason = rules.eliminate_exhausted_movers && !next_ready ?
        EndReason::NoLegalActions : EndReason::PlyCap;
  }
  return info;
}

StepInfo recorded_move(State& st, Move move, const Rules& rules) {
  StepInfo info{move,st.to_move,Outcome::Move,0,0};
  if (!is_legal(st,move)) throw std::runtime_error("recorded move is illegal or on the wrong turn");
  move_pieces(st,move,rules,info);
  return info;
}

void recorded_retirement(State& st,int seat,StepInfo& info,ExitReason reason) {
  if (seat<0 || seat>=kSeats || st.out[seat]) throw std::runtime_error("invalid recorded retirement");
  eliminate(st,seat,info,reason);
}

bool apply_checked(State& state, Move move, const Rules& rules, State& undo, StepInfo& event) {
  if (!is_legal(state, move)) return false;
  undo = state; event = step(state, move, rules); return true;
}
void restore(State& state, const State& undo) { state = undo; }

uint64_t state_hash(const State& st) {
  uint64_t h = fnv1a(st.id_at, sizeof(st.id_at));
  h = fnv1a(st.rank_of, sizeof(st.rank_of), h);
  h = fnv1a(st.out, sizeof(st.out), h);
  h = fnv1a(st.flag_revealed, sizeof(st.flag_revealed), h);
  const int tail[] = {st.to_move, st.ply, st.no_progress, static_cast<int>(st.result)};
  h = fnv1a(tail, sizeof(tail), h);
  return st.match == MatchKind::FourSeats ? h : fnv1a(&st.match, sizeof(st.match), h);
}

}  // namespace jq
