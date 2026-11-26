#pragma once
#include "config.h"
#include "log.h"
#include "decision.h"
#include "../core/layout.h"
#include <iostream>
#include <functional>
#include <array>
#include <optional>
namespace jq {
inline void expect_value(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
inline State replay_initial(const Json& h, Rules& rules, IdentityVisibility* recorded_visibility=nullptr) {
  expect_value(h.at("type").str() == "header" && h.at("schema").integer() == 1, "log schema");
  expect_value(h.at("engine").str() == "junqi-mvp-1" && h.at("topology").str() == "experimental-129-v1", "engine/topology version mismatch");
  expect_value(h.at("history_complete").boolean() && h.at("info_level").str() == "referee_private", "incomplete/non-private log");
  std::string err, rt = dump_json(h.at("rules")), ot = dump_json(h.at("observation"));
  if (!parse_rules(rt, rules, err)) throw std::runtime_error(err);
  const auto visibility=validate_observation(ot,rules);
  const auto match = h.object.count("match") ? MatchKind::SouthNorth : MatchKind::FourSeats;
  if (match == MatchKind::SouthNorth) {
    expect_value(h.at("match").str()=="single-dark", "unsupported match");
    const auto& config=h.at("rules");
    expect_value(config.at("players").integer()==2 && dump_json(config.at("teams"))=="[[0],[2]]",
                 "single match metadata mismatch");
    expect_value(visibility==IdentityVisibility::OwnAndPublic,"single-dark requires hidden opponent ranks");
  }
  if (recorded_visibility) *recorded_visibility=visibility;
  else expect_value(visibility==IdentityVisibility::OwnAndPublic,"diagnostic requires four-dark visibility");
  expect_value(h.at("rules_hash").str() == std::to_string(rules_hash(rules,match)), "rules hash mismatch");
  expect_value(h.at("observation_hash").str() == std::to_string(fnv1a(ot.data(), ot.size())), "observation hash mismatch");
  const auto& ls = h.at("layouts"); int8_t layouts[kSeats][kPiecesPerSeat]{};
  expect_value(ls.type == 'a' && ls.array.size() == 4, "layout count");
  for (int s = 0; s < 4; ++s) {
    const bool active=participates(match,s);
    expect_value(ls.array[s].type == 'a' && ls.array[s].array.size() == (active ? size_t(kPiecesPerSeat) : 0),
                 "layout slots for match");
    if (!active) continue;
    for (int k = 0; k < 25; ++k) {
      int rank = ls.array[s].array[k].integer(); expect_value(rank >= Flag && rank < RankCount, "rank bounds");
      layouts[s][k] = static_cast<int8_t>(rank);
    }
    expect_value(valid_layout(layouts[s]), "invalid deployment");
  }
  int first = h.at("first").integer(); expect_value(participates(match,first), "invalid first seat for match");
  std::ostringstream board; board << '{'; board_json(board); board << '}'; auto expected = parse_json(board.str());
  for (const auto& kv : expected.object) expect_value(dump_json(h.at(kv.first)) == dump_json(kv.second), "board data mismatch");
  State s; init_state(s, layouts, first, rules, match); return s;
}
// Diagnostic only: inspect reachable enemy squares, not their hidden ranks.
inline void report_candidates(const DecisionScorer& scorer, Move chosen) {
  const auto& observation=scorer.view;
  const bool scored=scorer.kind!=DecisionKind::Random;
  std::vector<Move> moves;
  gen_moves(observation, observation.viewer, moves);
  const int chosen_score = scorer.score(chosen).total();
  int attack_count = 0, best_attack = -100000, same_score_alternatives = 0;
  for (const auto move : moves) {
    const int score = scorer.score(move).total();
    if ((move.from != chosen.from || move.to != chosen.to) && score == chosen_score)
      ++same_score_alternatives;
    if (observation.seat_at(move.to) >= 0 &&
        observation.team(observation.seat_at(move.to)) != observation.team(observation.viewer)) {
      ++attack_count;
      best_attack = std::max(best_attack, score);
    }
  }
  std::cout << "CAND " << observation.ply + 1 << ' ' << moves.size() << ' ' << attack_count
            << ' ' << (attack_count && scored ? chosen_score - best_attack : 0)
            << ' ' << (scored ? same_score_alternatives : 0) << '\n';
}
constexpr unsigned all_evidence_ranks = (1u << RankCount) - 1;
inline unsigned public_route_ranks(const Observation& before, Move move) {
  if (move.from >= kNodes || move.to >= kNodes || before.seat_at(move.from) < 0)
    return all_evidence_ranks;
  Observation hypothesis = before;
  unsigned compatible = 0;
  std::vector<Move> moves;
  for (int rank = Flag; rank < RankCount; ++rank) {
    hypothesis.ranks[move.from] = static_cast<int8_t>(rank);
    moves.clear();
    gen_piece_moves(hypothesis, before.seat_at(move.from), move.from, moves);
    if (std::find(moves.begin(), moves.end(), move) != moves.end()) compatible |= 1u << rank;
  }
  return compatible;
}
inline unsigned public_defender_ranks(int known_attacker, Outcome outcome, const Rules& rules) {
  if (known_attacker < Bomb || known_attacker >= RankCount || outcome == Outcome::Move)
    return all_evidence_ranks;
  unsigned compatible = 0;
  for (int rank = Flag; rank < RankCount; ++rank)
    if (resolve(known_attacker, rank, rules) == outcome) compatible |= 1u << rank;
  return compatible;
}
struct RemovedRanks {
  std::array<uint16_t,kPiecesPerSeat> masks{};
  std::array<uint8_t,RankCount> confirmed{};
  unsigned exhausted=0;
  uint8_t size=0;

  void add(unsigned mask) {
    expect_value(mask && size<kPiecesPerSeat,"invalid collision loss evidence");
    masks[size++]=static_cast<uint16_t>(mask);
    if ((mask&(mask-1))!=0) return;
    for (int rank=Flag;rank<RankCount;++rank) if (mask==(1u<<rank)) {
      expect_value(++confirmed[rank]<=kRankCount[rank],"confirmed losses exceed rank inventory");
      if (confirmed[rank]==kRankCount[rank]) exhausted|=mask;
    }
  }
};

struct OpeningMarks {
  std::array<unsigned, kNodes> ranks{};
  std::array<bool, kNodes> moved{};
  std::array<RemovedRanks,kSeats> removed{};
  OpeningMarks(const Observation& initial, bool complete_opening) {
    for (int node = 0; node < kNodes; ++node) {
      if (initial.seat_at(node) < 0) continue;
      ranks[node] = all_evidence_ranks;
      if (!complete_opening || initial.ply != 0 || initial.seat_at(node) != Board::get().node(node).seat)
        continue;
      ranks[node] = 0;
      for (int rank = Flag; rank < RankCount; ++rank)
        if (deployment_rank_allowed(node, rank)) ranks[node] |= 1u << rank;
    }
  }
  void advance(Move move, Outcome outcome, const Observation& after, unsigned route=all_evidence_ranks) {
    if (outcome == Outcome::Move || outcome == Outcome::AttackerWins) {
      constexpr unsigned immobile=(1u << Flag) | (1u << Mine);
      const unsigned possible=ranks[move.from] & route & ~immobile;
      expect_value(possible != 0, "inconsistent public route history");
      ranks[move.to]=possible;
      moved[move.to]=true;
    }
    ranks[move.from] = 0;
    moved[move.from] = false;
    for (int node = 0; node < kNodes; ++node)
      if (after.seat_at(node) < 0) { ranks[node] = 0; moved[node] = false; }
  }
};
inline void advance_public_ranks(OpeningMarks& marks, const Observation& before,
                                 const Observation& after, Move move, Outcome outcome) {
  const bool surviving_dark_mover=before.rank_at(move.from) == Unknown &&
    (outcome == Outcome::Move || outcome == Outcome::AttackerWins);
  marks.advance(move,outcome,after,surviving_dark_mover ? public_route_ranks(before,move) : all_evidence_ranks);
}
inline StepInfo step_with_public_ranks(State& state, const Rules& rules, int viewer,
                                  Move action, OpeningMarks* marks) {
  if (!marks) return step(state,action,rules);
  const auto before=observe(state,viewer);
  const auto event=step(state,action,rules);
  advance_public_ranks(*marks,before,observe(state,viewer),action,event.outcome);
  return event;
}
inline unsigned collision_outcome_mask(int attacker, unsigned defenders, const Rules& rules) {
  unsigned outcomes = 0;
  for (int rank = Flag; rank < RankCount; ++rank)
    if (defenders & (1u << rank)) outcomes |= 1u << static_cast<unsigned>(resolve(attacker,rank,rules));
  return outcomes;
}
struct OpeningContacts {
  bool marked = false;
  bool bomb_changes_outcome = false;
};
inline OpeningContacts opening_contacts(const Observation& before, const OpeningMarks& marks,
                                         const Rules& rules) {
  OpeningContacts result;
  std::vector<Move> legal;
  gen_moves(before, before.viewer, legal);
  for (const auto move : legal) {
    const unsigned ranks = marks.ranks[move.to];
    if (before.rank_at(move.to) != Unknown || !ranks || (ranks & (1u << Bomb))) continue;
    result.marked = true;
    const int attacker = before.rank_at(move.from);
    result.bomb_changes_outcome |= collision_outcome_mask(attacker,ranks,rules) !=
                                   collision_outcome_mask(attacker,ranks | (1u << Bomb),rules);
  }
  return result;
}
struct SurvivorEvidence {
  std::array<unsigned, kNodes> collision{};
  std::array<unsigned, kNodes> after_move{};

  int encounters = 0, first_moves = 0;
  void advance(const Observation& before, const Observation& after, Move move, Outcome outcome,
               const OpeningMarks& opening, unsigned route, const Rules& rules) {
    constexpr unsigned mine = 1u << Mine;
    if (outcome == Outcome::DefenderWins) {
      if (before.turn == before.viewer && before.rank_at(move.from) == Platoon &&
          before.rank_at(move.to) == Unknown && !opening.moved[move.to] && (opening.ranks[move.to] & mine)) {
        const auto candidates = public_defender_ranks(Platoon,outcome,rules) & opening.ranks[move.to];
        if ((candidates & mine) && !collision[move.to]) {
          collision[move.to] = candidates;
          ++encounters;
        }
      }
    } else {
      const bool attacker_lives = outcome == Outcome::Move || outcome == Outcome::AttackerWins;
      if (attacker_lives && collision[move.from] && !after_move[move.from]) ++first_moves;
      collision[move.to] = attacker_lives ? collision[move.from] : 0;
      after_move[move.to] = attacker_lives ? (after_move[move.from] ? after_move[move.from] :
                                             collision[move.from]) & route : 0;
      if (collision[move.to] && !after_move[move.to])
        throw std::runtime_error("survivor evidence contradicts observed route");
    }
    collision[move.from] = after_move[move.from] = 0;
    for (int node = 0; node < kNodes; ++node)
      if (after.seat_at(node) < 0) collision[node] = after_move[node] = 0;
  }
};

inline bool survivor_changes_attack(const Observation& before, const SurvivorEvidence& evidence,
                                    const Rules& rules) {
  std::vector<Move> legal;
  gen_moves(before,before.viewer,legal);
  for (const auto move : legal) {
    const int target = move.to, owner = before.seat_at(target);
    if (!evidence.after_move[target] || before.rank_at(move.from) != Engineer ||
        before.rank_at(target) != Unknown || owner != Board::get().node(target).seat ||
        !deployment_rank_allowed(target,Mine)) continue;
    // Toggle only the mine possibility. Other observed route exclusions stay fixed.
    const unsigned narrowed = evidence.after_move[target];
    const unsigned counterfactual = narrowed | (evidence.collision[target] & (1u << Mine));
    if (collision_outcome_mask(Engineer,narrowed,rules) !=
        collision_outcome_mask(Engineer,counterfactual,rules)) return true;
  }
  return false;
}
inline StepInfo replay_frame(const Json& frame, const Rules& rules, State& s, State& undo,
                             IdentityVisibility visibility, SimulationPolicy policy, bool candidates, bool settlement,
                             bool review = false) {
  const auto& ev=frame.at("event");
  expect_value(ev.type=='a' && ev.array.size()==6,"event shape");
  const int from=ev.array[1].integer(), to=ev.array[2].integer();
  expect_value(from>=0 && from<kNodes && to>=0 && to<kNodes,"move bounds");
  const Move move{uint8_t(from),uint8_t(to)};
  const auto observation=observe(s,s.to_move,visibility);
  const DecisionScorer scorer(observation,rules,policy.at(s.to_move));
  if (candidates) report_candidates(scorer,move);
  StepInfo info;
  expect_value(apply_checked(s,move,rules,undo,info),"illegal replay move");
  const auto expected=policy.records_decision() ? decision_frame(s,info,decision_record(scorer,move),visibility,settlement,review)
                           : frame_json(s,&info,visibility,settlement,review);
  expect_value(dump_json(frame)==dump_json(parse_json(expected)),"frame/event mismatch");
  return info;
}

using ReplayObserver = std::function<void(const State&, const StepInfo&, const State&)>;
inline void replay(const std::string& file, bool candidates = false,
                   const ReplayObserver& observer = {}, int evidence_seat = -1,
                   std::ostream* review_output = nullptr) {
  std::string text = read_text(file); expect_value(!text.empty() && text.back() == '\n', "truncated log tail");
  std::istringstream input(text); std::string line; std::getline(input, line);
  auto header = parse_json(line);
  expect_value(!header.object.count("source_tape"),"recorded JGS review requires the archival replay entry");
  const auto policy=header.object.count("policy")?simulation_policy(header.at("policy").str()):SimulationPolicy{};
  Rules rules; IdentityVisibility visibility;
  State s = replay_initial(header,rules,&visibility);
  const bool settlement=header.object.count("settlement_detail")!=0;
  const bool review=header.object.count("review_version")!=0;
  if (review) expect_value(header.at("review_version").integer()==1,"unsupported review version");
  if (settlement) expect_value(header.at("settlement_detail").boolean() && rules.eliminate_exhausted_movers,
                               "invalid settlement detail profile");
  if (s.match==MatchKind::SouthNorth || visibility!=IdentityVisibility::OwnAndPublic)
    expect_value(!observer && evidence_seat<0, "diagnostic requires a four-dark match");
  expect_value(bool(std::getline(input, line)), "missing initial frame");
  expect_value(dump_json(parse_json(line)) == dump_json(parse_json(frame_json(s,nullptr,visibility,false,review))), "initial frame mismatch");
  if (review_output) {
    header.object["review_version"]=parse_json("1");
    *review_output << dump_json(header) << '\n' << frame_json(s,nullptr,visibility,false,true) << '\n';
  }
  int route_checks = 0, engineer_only = 0, collision_checks = 0, narrowed = 0;
  int marked_contacts = 0, bomb_outcome_changes = 0, evidence_turns = 0;
  int survivor_attack_turns = 0;
  SurvivorEvidence survivor;
  std::optional<OpeningMarks> opening_marks;
  if (evidence_seat >= 0) opening_marks.emplace(observe(s,evidence_seat), true);
  while (std::getline(input, line)) {
    expect_value(!line.empty() && s.result == Result::Ongoing, "extra/empty event");
    State undo;
    const auto frame=parse_json(line);
    const auto info=replay_frame(frame,rules,s,undo,visibility,policy,candidates,settlement,review);
    if (review_output) {
      *review_output << (policy.records_decision() ? decision_frame(s,info,frame.at("decision"),visibility,settlement,true)
                             : frame_json(s,&info,visibility,settlement,true)) << '\n';
    }
    if (evidence_seat >= 0) {
      const auto move=info.move;
      const auto visible = observe(undo, evidence_seat);
      unsigned route = all_evidence_ranks;
      if (info.seat == evidence_seat) {
        const auto contacts = opening_contacts(visible,*opening_marks,rules);
        ++evidence_turns;
        survivor_attack_turns += survivor_changes_attack(visible,survivor,rules);
        marked_contacts += contacts.marked;
        bomb_outcome_changes += contacts.bomb_changes_outcome;
      }
      if (visible.rank_at(move.from) == Unknown) {
        const auto ranks = public_route_ranks(visible, move);
        route = ranks;
        expect_value(ranks != 0, "public route inconsistent with rules");
        ++route_checks;
        engineer_only += ranks == (1u << Engineer);
      } else if (info.seat == evidence_seat && info.outcome != Outcome::Move &&
                 visible.rank_at(move.to) == Unknown) {
        const auto ranks = public_defender_ranks(visible.rank_at(move.from), info.outcome, rules);
        expect_value(ranks != 0, "public collision inconsistent with rules");
        ++collision_checks;
        narrowed += ranks != all_evidence_ranks;
      }
      const auto after = observe(s,evidence_seat);
      survivor.advance(visible,after,move,info.outcome,*opening_marks,route,rules);
      opening_marks->advance(move,info.outcome,after,route);
    }
    if (observer) observer(undo,info,s);
  }
  expect_value(s.result != Result::Ongoing, "unfinished log");
  if (evidence_seat >= 0)
    std::cout << "PUBLIC EVIDENCE routes=" << route_checks << " engineer_only=" << engineer_only
              << " collisions=" << collision_checks << " narrowed=" << narrowed
              << " opening_no_bomb_contact_turns=" << marked_contacts
              << " bomb_outcome_change_turns=" << bomb_outcome_changes
              << " evidence_turns=" << evidence_turns
              << " survivor_attack_turns=" << survivor_attack_turns
              << " survivor_encounters=" << survivor.encounters << " survivor_moves=" << survivor.first_moves << '\n';
  std::cout << "REPLAY PASS plies=" << s.ply << " result=" << int(s.result) << '\n';
}
}
