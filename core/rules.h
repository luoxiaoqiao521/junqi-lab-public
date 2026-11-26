// Experimental rule configuration. Values come from config/ruleset.json
// ("experimental" section); none of them is a confirmed QQ rule.
#pragma once
#include <cstdint>
#include <string>
#include "types.h"

namespace jq {

struct Rules {
  bool ccw = true;                    // turn.order
  bool random_first_seat = true;      // turn.first_seat
  bool bomb_captures_flag = false;    // combat.bomb_vs_flag
  bool bomb_mine_both_removed = true; // combat.bomb_vs_mine
  bool mine_survives = true;          // combat.mine_survives_non_engineer
  bool flag_eliminates = true;        // elim.flag_removed_eliminates
  bool marshal_reveals_flag = false;  // marshal_death.reveal_flag
  int no_progress_limit = 70;         // draw.no_progress.limit, 0 = off
  bool reset_on_capture_only = false; // draw.no_progress.reset_on
  bool draw_before_win = false;       // terminal.priority
  int truncation_cap = 3000;          // sim.truncation_cap (guard, not a rule)
  bool eliminate_exhausted_movers = false; // selfplay: only intrinsically immobile seats retire
  bool check_involved_no_move = false; // elim.no_legal_move archive variant
};

inline int next_seat(int seat, const Rules& rules) {
  return (seat + (rules.ccw ? 1 : kSeats - 1)) % kSeats;
}

// Parses the "experimental" object of a ruleset JSON text. Returns false and
// fills err when a required key is missing or a value is not implemented;
// the caller must then refuse to start a game.
bool parse_rules(const std::string& json, Rules& out, std::string& err);

// Collision between two known ranks, independent of board or hidden state.
Outcome resolve(int att, int def, const Rules& rules);

// Stable hash of the effective rule values (not of the file bytes).
uint64_t rules_hash(const Rules& r, MatchKind match = MatchKind::FourSeats);

}  // namespace jq
