#include "rules.h"
#include <cctype>
#include <charconv>
#include "json.h"
#include <map>
#include "rng.h"

namespace jq {

namespace {

using KV = std::map<std::string, std::string>;

// Reads the flat object that follows "experimental". Values are kept as raw
// text with string quotes stripped.
bool read_section(const std::string& j, KV& kv, std::string& err) {
  try {
    auto root = parse_json(j);
    if (root.at("schema_version").integer() != 2 || !root.at("runnable").boolean())
      throw std::runtime_error("unsupported/disabled ruleset");
    const auto& values = root.at("experimental");
    const char* keys[] = {"board.points", "turn.order", "turn.first_seat", "combat.bomb_vs_flag", "combat.bomb_vs_mine", "combat.mine_survives_non_engineer", "elim.flag_removed_eliminates", "elim.no_legal_move", "elim.piece_handling", "marshal_death.reveal_flag", "ally.attack", "draw.no_progress.limit", "draw.no_progress.unit", "draw.no_progress.reset_on", "terminal.priority", "reveal.on_collision", "sim.truncation_cap"};
    const char types[] = {'i', 's', 's', 's', 's', 'b', 'b', 's', 's', 'b', 'b', 'i', 's', 's', 's', 's', 'i'};
    if (values.type != 'o' || values.object.size() != sizeof(types)) throw std::runtime_error("unknown rule key");
    for (size_t i = 0; i < sizeof(types); ++i) {
      const auto& v = values.at(keys[i]);
      if (v.type != types[i]) throw std::runtime_error("rule value type");
      kv[keys[i]] = v.text;
    }
    return true;
  } catch (const std::exception& e) { err = e.what(); return false; }
}

// Maps key to one of two allowed values; anything else is an error.
bool pick(const KV& kv, const char* key, const char* yes, const char* no, bool& out, std::string& err) {
  auto it = kv.find(key);
  if (it == kv.end()) { err = std::string("missing ") + key; return false; }
  if (it->second == yes) { out = true; return true; }
  if (it->second == no) { out = false; return true; }
  err = std::string("unsupported value for ") + key + ": " + it->second;
  return false;
}

bool require(const KV& kv, const char* key, const char* only, std::string& err) {
  auto it = kv.find(key);
  if (it != kv.end() && it->second == only) return true;
  err = std::string("unsupported/missing ") + key; return false;
}

bool retirement_policy(const KV& kv, Rules& rules, std::string& err) {
  if (kv.at("elim.no_legal_move") != "eliminated_no_movable_piece")
    return pick(kv, "elim.no_legal_move", "eliminated_after_involved_move", "eliminated",
                rules.check_involved_no_move, err);
  rules.eliminate_exhausted_movers = true;
  return require(kv, "elim.flag_removed_eliminates", "true", err);
}

bool number(const KV& kv, const char* key, int lo, int& out, std::string& err) {
  auto it = kv.find(key);
  if (it == kv.end()) { err = std::string("missing ") + key; return false; }
  auto parsed = std::from_chars(it->second.data(), it->second.data() + it->second.size(), out);
  if (parsed.ec != std::errc{} || parsed.ptr != it->second.data() + it->second.size()) { err = "invalid integer"; return false; }
  if (out < lo) { err = std::string("out of range: ") + key; return false; }
  return true;
}

}  // namespace

static bool parse_values(const std::string& json, Rules& r, std::string& err) {
  KV kv;
  if (!read_section(json, kv, err)) return false;
  return require(kv, "board.points", "129", err) &&
         pick(kv, "turn.order", "ccw", "cw", r.ccw, err) &&
         pick(kv, "turn.first_seat", "seeded_random", "fixed_seat_0", r.random_first_seat, err) &&
         pick(kv, "combat.bomb_vs_flag", "bomb_captures", "both_removed", r.bomb_captures_flag, err) &&
         pick(kv, "combat.bomb_vs_mine", "both_removed", "bomb_removed_mine_stays", r.bomb_mine_both_removed, err) &&
         pick(kv, "combat.mine_survives_non_engineer", "true", "false", r.mine_survives, err) &&
         pick(kv, "elim.flag_removed_eliminates", "true", "false", r.flag_eliminates, err) &&
         retirement_policy(kv, r, err) &&
         require(kv, "elim.piece_handling", "removed", err) &&
         pick(kv, "marshal_death.reveal_flag", "true", "false", r.marshal_reveals_flag, err) &&
         require(kv, "ally.attack", "false", err) &&
         number(kv, "draw.no_progress.limit", 0, r.no_progress_limit, err) &&
         require(kv, "draw.no_progress.unit", "ply", err) &&
         pick(kv, "draw.no_progress.reset_on", "capture", "collision", r.reset_on_capture_only, err) &&
         pick(kv, "terminal.priority", "draw_before_win", "win_before_draw", r.draw_before_win, err) &&
         require(kv, "reveal.on_collision", "result_only", err) &&
         number(kv, "sim.truncation_cap", 1, r.truncation_cap, err);
}

bool parse_rules(const std::string& json, Rules& r, std::string& err) {
  Rules candidate;
  if (!parse_values(json, candidate, err)) return false;
  r = candidate; err.clear(); return true;
}

Outcome resolve(int att, int def, const Rules& rules) {
  if (att == Bomb || def == Bomb) {
    int other = (att == Bomb) ? def : att;
    if (other == Flag && att == Bomb && rules.bomb_captures_flag) return Outcome::AttackerWins;
    if (other == Mine && def == Mine && !rules.bomb_mine_both_removed) return Outcome::DefenderWins;
    return Outcome::BothRemoved;
  }
  if (def == Mine) {
    if (att == Engineer) return Outcome::AttackerWins;
    return rules.mine_survives ? Outcome::DefenderWins : Outcome::BothRemoved;
  }
  if (def == Flag) return Outcome::AttackerWins;
  if (att == def) return Outcome::BothRemoved;
  return att > def ? Outcome::AttackerWins : Outcome::DefenderWins;
}

uint64_t rules_hash(const Rules& r, MatchKind match) {
  const int v[] = {r.ccw, r.random_first_seat, r.bomb_captures_flag, r.bomb_mine_both_removed,
                   r.mine_survives, r.flag_eliminates, r.marshal_reveals_flag, r.no_progress_limit,
                   r.reset_on_capture_only, r.draw_before_win, r.truncation_cap};
  auto hash = fnv1a(v, sizeof(v));
  if (r.eliminate_exhausted_movers) {
    constexpr uint8_t exhausted = 2;
    hash = fnv1a(&exhausted, sizeof(exhausted), hash);
  } else if (r.check_involved_no_move) {
    constexpr uint8_t involved = 1;
    hash = fnv1a(&involved, sizeof(involved), hash);
  }
  return match == MatchKind::FourSeats ? hash : fnv1a(&match, sizeof(match), hash);
}

}  // namespace jq
