// Basic identifiers shared by every module. Single source for ranks and counts.
#pragma once
#include <cstdint>

namespace jq {

constexpr int kSeats = 4;
constexpr int kNodes = 129;
constexpr int kPiecesPerSeat = 25;
constexpr int kPieces = kSeats * kPiecesPerSeat;
constexpr int kNone = -1;

// Order matters: values >= Engineer compare by strength.
enum Rank : int8_t {
  Unknown = -1,
  Flag = 0, Mine, Bomb,
  Engineer, Platoon, Company, Battalion, Regiment, Brigade, Division, Corps, Marshal,
  RankCount
};

// Pieces per seat for each rank, indexed by Rank.
constexpr int kRankCount[RankCount] = {1, 3, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1};

inline bool is_army(int r) { return r >= Engineer; }
inline int team_of(int seat) { return seat & 1; }

enum class MatchKind : uint8_t { FourSeats, SouthNorth };

inline bool participates(MatchKind match, int seat) {
  return seat >= 0 && seat < kSeats && (match == MatchKind::FourSeats || seat == 0 || seat == 2);
}

inline int team_of(int seat, MatchKind match) {
  if (!participates(match, seat)) return kNone;
  return match == MatchKind::SouthNorth ? seat / 2 : team_of(seat);
}

struct Move {
  uint8_t from = 0, to = 0;
  bool operator==(const Move& o) const { return from == o.from && to == o.to; }
};

// Collision outcome as seen publicly (reveal.on_collision = result_only).
enum class Outcome : uint8_t { Move, AttackerWins, DefenderWins, BothRemoved };

}  // namespace jq
