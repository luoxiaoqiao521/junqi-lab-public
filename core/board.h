// Static board topology: 4 x 30 seat points + 9 center points = 129 nodes.
// Seat-local coordinates: row r = 0 (front, nearest center) .. 5 (back row),
// column c = 0 (the player's left) .. 4. Seats are 0 south, 1 east, 2 north,
// 3 west; seat s+1 is counter-clockwise from seat s.
#pragma once
#include <cstdint>
#include <vector>
#include "types.h"

namespace jq {

enum class NodeKind : uint8_t { Post, Camp, HQ, Center };

struct Node {
  int8_t seat;   // owning seat, kNone for center
  int8_t r, c;   // seat-local coordinates (center: -1)
  int8_t gr, gc; // global 17x17 grid coordinates, for display and tests
  NodeKind kind;
  bool rail;
};

struct RailRef { uint8_t line, index; };

class Board {
 public:
  static const Board& get();
  const Node& node(int n) const { return nodes_[n]; }
  int seat_node(int seat, int r, int c) const { return seat_index_[seat][r][c]; }
  int at_grid(int gr, int gc) const { return grid_[gr][gc]; }
  const std::vector<uint8_t>& road(int n) const { return road_[n]; }
  const std::vector<uint8_t>& rail_adj(int n) const { return rail_adj_[n]; }
  const std::vector<RailRef>& rail_refs(int n) const { return rail_refs_[n]; }
  const std::vector<uint8_t>& line(int i) const { return lines_[i]; }
  int line_count() const { return static_cast<int>(lines_.size()); }

 private:
  Board();
  int add_node(int seat, int r, int c, int gr, int gc, NodeKind k, bool rail);
  void link_road(int a, int b);
  void add_line(const std::vector<uint8_t>& pts);
  void build_seats();
  void build_lines();

  Node nodes_[kNodes];
  int count_ = 0;
  int seat_index_[kSeats][6][5];
  int grid_[17][17];
  std::vector<uint8_t> road_[kNodes];
  std::vector<uint8_t> rail_adj_[kNodes];
  std::vector<RailRef> rail_refs_[kNodes];
  std::vector<std::vector<uint8_t>> lines_;
};

}  // namespace jq
