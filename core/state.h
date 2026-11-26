// Full game state (referee-side truth) and the rule step function.
// Move generation is a template over a "view" so the same code serves the
// referee (true state) and players (Observation): a view only needs
// seat_at(n), rank_at(n) and team(seat); ranks of other pieces may be unknown.
#pragma once
#include <cstdint>
#include <vector>
#include "board.h"
#include "moves.h"
#include "rules.h"
#include "types.h"

namespace jq {

enum class Result : uint8_t { Ongoing, Team0Wins, Team1Wins, Draw, Truncated };
enum class ExitReason : uint8_t { None, FlagRemoved, MovablePiecesExhausted, NoLegalMove, Count };
enum class EndReason : uint8_t {
  None, TeamEliminated, NoProgressDraw, SimultaneousExhaustion, NoLegalActions, PlyCap, Count
};

struct State {
  int8_t seat_at[kNodes];
  int8_t rank_at[kNodes];
  int8_t id_at[kNodes];          // piece id (seat * 25 + k) or kNone
  uint8_t node_of[kPieces];
  int8_t rank_of[kPieces];
  bool alive[kPieces];
  bool out[kSeats];
  bool flag_revealed[kSeats];
  int8_t to_move;
  int ply;
  int no_progress;
  Result result;
  MatchKind match;

  int seat_at_n(int n) const { return seat_at[n]; }
};

// Read-only adaptor so templates can take State directly.
struct StateView {
  const State& s;
  int seat_at(int n) const { return s.seat_at[n]; }
  int rank_at(int n) const { return s.rank_at[n]; }
  int team(int seat) const { return team_of(seat, s.match); }
};

// What happened in one step, for logs and observations.
struct StepInfo {
  Move move;
  int8_t seat;
  Outcome outcome;
  uint8_t eliminated_mask;   // seats eliminated during this step
  uint8_t revealed_mask;     // seats whose flag became public this step
  ExitReason exit_reason[kSeats]{};
  uint8_t skipped_mask = 0;  // temporarily blocked live seats passed over by the referee
  EndReason end_reason = EndReason::None;
};

// layouts[seat][k] = rank placed on the k-th non-camp node of the seat,
// in order of increasing (r, c). See layout.h.
void init_state(State& st, const int8_t layouts[kSeats][kPiecesPerSeat], int first_seat,
                const Rules& rules, MatchKind match = MatchKind::FourSeats);

// Legal moves for the side to move. Empty when the game is over.
void legal_moves(const State& st, std::vector<Move>& out);

// Applies a legal move (caller guarantees legality; use is_legal first when
// the move comes from outside). Advances the turn and evaluates termination.
StepInfo step(State& st, Move m, const Rules& rules);

// Archival replay takes turn order and settlement from separately recorded events.
StepInfo recorded_move(State& st, Move move, const Rules& rules);
void recorded_retirement(State& st,int seat,StepInfo& info,ExitReason reason);

bool apply_checked(State& state, Move move, const Rules& rules, State& undo, StepInfo& event);
void restore(State& state, const State& undo);

bool is_legal(const State& st, Move m);
uint64_t state_hash(const State& st);

}  // namespace jq
