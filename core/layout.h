#pragma once
#include "types.h"
#include "rng.h"

namespace jq {
// Row-major non-camp slots; invalid arguments return kNone.
int layout_slot_node(int seat, int slot);
// Experimental deployment: flag in HQ, mines in last two rows,
// bombs outside first row. This does not assert QQ compatibility.
bool deployment_rank_allowed(int node, int rank);
bool valid_layout(const int8_t (&layout)[kPiecesPerSeat]);
void generate_layout(int8_t (&layout)[kPiecesPerSeat], Rng& rng);
}
