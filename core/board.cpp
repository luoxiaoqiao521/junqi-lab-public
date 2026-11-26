#include "board.h"
#include <algorithm>

namespace jq {

namespace {

// Seat-local (r, c) -> global grid. r may be negative for center points.
void to_grid(int seat, int r, int c, int& gr, int& gc) {
  switch (seat) {
    case 0: gr = 11 + r; gc = 6 + c; break;   // south
    case 1: gr = 10 - c; gc = 11 + r; break;  // east
    case 2: gr = 5 - r; gc = 10 - c; break;   // north
    default: gr = 6 + c; gc = 5 - r; break;   // west
  }
}

bool is_camp(int r, int c) {
  return (r == 1 || r == 3) ? (c == 1 || c == 3) : (r == 2 && c == 2);
}

void push_unique(std::vector<uint8_t>& v, int x) {
  if (std::find(v.begin(), v.end(), x) == v.end()) v.push_back(static_cast<uint8_t>(x));
}

}  // namespace

const Board& Board::get() {
  static const Board b;
  return b;
}

Board::Board() {
  for (auto& row : grid_) for (int& g : row) g = kNone;
  build_seats();
  for (int gr = 6; gr <= 10; gr += 2)
    for (int gc = 6; gc <= 10; gc += 2) add_node(kNone, -1, -1, gr, gc, NodeKind::Center, true);
  build_lines();
}

int Board::add_node(int seat, int r, int c, int gr, int gc, NodeKind k, bool rail) {
  int n = count_++;
  nodes_[n] = Node{static_cast<int8_t>(seat), static_cast<int8_t>(r), static_cast<int8_t>(c),
                   static_cast<int8_t>(gr), static_cast<int8_t>(gc), k, rail};
  grid_[gr][gc] = n;
  return n;
}

void Board::link_road(int a, int b) {
  push_unique(road_[a], b);
  push_unique(road_[b], a);
}

void Board::build_seats() {
  for (int s = 0; s < kSeats; ++s) {
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 5; ++c) {
        int gr, gc;
        to_grid(s, r, c, gr, gc);
        NodeKind k = is_camp(r, c) ? NodeKind::Camp
                   : (r == 5 && (c == 1 || c == 3)) ? NodeKind::HQ : NodeKind::Post;
        bool rail = r <= 4 && (r == 0 || r == 4 || c == 0 || c == 4);
        seat_index_[s][r][c] = add_node(s, r, c, gr, gc, k, rail);
      }
    }
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 5; ++c) {
        int n = seat_index_[s][r][c];
        if (r + 1 < 6) link_road(n, seat_index_[s][r + 1][c]);
        if (c + 1 < 5) link_road(n, seat_index_[s][r][c + 1]);
        if (!is_camp(r, c)) continue;
        for (int dr : {-1, 1})
          for (int dc : {-1, 1}) link_road(n, seat_index_[s][r + dr][c + dc]);
      }
    }
  }
}

void Board::add_line(const std::vector<uint8_t>& pts) {
  int id = static_cast<int>(lines_.size());
  lines_.push_back(pts);
  for (size_t i = 0; i < pts.size(); ++i) {
    rail_refs_[pts[i]].push_back(RailRef{static_cast<uint8_t>(id), static_cast<uint8_t>(i)});
    if (i == 0) continue;
    link_road(pts[i - 1], pts[i]);
    push_unique(rail_adj_[pts[i - 1]], pts[i]);
    push_unique(rail_adj_[pts[i]], pts[i - 1]);
  }
}

void Board::build_lines() {
  // Lines crossing the center: seat s column c, three center points, then the
  // opposite seat's mirrored column. Only the front point of column 2 is rail.
  for (int s = 0; s < 2; ++s) {
    for (int c = 0; c <= 4; c += 2) {
      std::vector<uint8_t> pts;
      int deep = (c == 2) ? 0 : 4;
      for (int r = deep; r >= 0; --r) pts.push_back(static_cast<uint8_t>(seat_index_[s][r][c]));
      for (int r = -1; r >= -5; r -= 2) {
        int gr, gc;
        to_grid(s, r, c, gr, gc);
        pts.push_back(static_cast<uint8_t>(grid_[gr][gc]));
      }
      for (int r = 0; r <= deep; ++r) pts.push_back(static_cast<uint8_t>(seat_index_[s + 2][r][4 - c]));
      add_line(pts);
    }
  }
  for (int s = 0; s < kSeats; ++s) {
    for (int r : {0, 4}) {
      std::vector<uint8_t> pts;
      for (int c = 0; c < 5; ++c) pts.push_back(static_cast<uint8_t>(seat_index_[s][r][c]));
      add_line(pts);
    }
    // Corner arc: this seat's right column joins the next seat's left column.
    std::vector<uint8_t> arc;
    int nx = (s + 1) % kSeats;
    for (int r = 4; r >= 0; --r) arc.push_back(static_cast<uint8_t>(seat_index_[s][r][4]));
    for (int r = 0; r <= 4; ++r) arc.push_back(static_cast<uint8_t>(seat_index_[nx][r][0]));
    add_line(arc);
  }
}

}  // namespace jq
