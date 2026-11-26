#pragma once
#include <ostream>
#include "../core/state.h"

namespace jq {

inline constexpr const char* exit_names[] = {
  "none", "flag_removed", "movable_pieces_exhausted", "no_legal_move"
};
inline constexpr const char* end_names[] = {
  "none", "team_eliminated", "no_progress_draw", "simultaneous_exhaustion", "no_legal_actions", "ply_cap"
};
static_assert(sizeof(exit_names)/sizeof(*exit_names)==int(ExitReason::Count));
static_assert(sizeof(end_names)/sizeof(*end_names)==int(EndReason::Count));

struct TrainingSettlement {
  struct Exit {
    int count=0, first_ply=0, first_seat=-1;
    uint64_t seed=0;
    int learner=-1;
  };
  Exit exits[int(ExitReason::Count)]{};
  int endings[int(EndReason::Count)]{};
  int automatic_skips=0;

  void record(const StepInfo& event, int ply) {
    for (int seat=0;seat<kSeats;++seat) {
      const auto reason=event.exit_reason[seat];
      if (reason!=ExitReason::None) {
        auto& exit=exits[int(reason)];
        if (exit.count++==0) { exit.first_ply=ply; exit.first_seat=seat; }
      }
      automatic_skips += (event.skipped_mask & (1u<<seat))!=0;
    }
    if (event.end_reason!=EndReason::None) ++endings[int(event.end_reason)];
  }

  void merge(const TrainingSettlement& episode, uint64_t seed, int learner) {
    for (int i=1;i<int(ExitReason::Count);++i) {
      const auto& exit=episode.exits[i];
      if (!exits[i].count && exit.count) {
        exits[i]=exit;
        exits[i].seed=seed;
        exits[i].learner=learner;
      } else exits[i].count+=exit.count;
    }
    for (int i=1;i<int(EndReason::Count);++i) endings[i]+=episode.endings[i];
    automatic_skips+=episode.automatic_skips;
  }

  void print(std::ostream& out, int scored_with_decisions) const {
    out << ",\"settlement\":{\"terminal_reasons\":{";
    for (int i=1;i<int(EndReason::Count);++i)
      out << (i==1 ? "" : ",") << '"' << end_names[i] << "\":" << endings[i];
    out << "},\"exit_reasons\":{";
    for (int i=1;i<int(ExitReason::Count);++i) {
      const auto& exit=exits[i];
      out << (i==1 ? "" : ",") << '"' << exit_names[i] << "\":{\"count\":" << exit.count;
      if (exit.count) out << ",\"example\":{\"ply\":" << exit.first_ply << ",\"seat\":" << exit.first_seat
                          << ",\"seed\":\"" << exit.seed << "\",\"learner\":" << exit.learner << '}';
      out << '}';
    }
    out << "},\"auto_skipped_turns\":" << automatic_skips
        << ",\"scored_with_decisions\":" << scored_with_decisions << '}';
  }
};

}
