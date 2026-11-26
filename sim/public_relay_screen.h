#pragma once
#include "regional_arrival.h"
#include "seat_episode.h"

namespace jq {

enum class PublicRelayEvidence { NewContact, EngineerMayAlreadyReach, EngineerOnly };
enum class PublicRelayReply { NotReceiverTurn, TargetChanged, Closed, Available };

struct PublicRelayContact {
  Move attack;
  int receiver;
  int target_seat;
  PublicRelayEvidence evidence;
};

inline unsigned relay_source_ranks(const Observation& view, const OpeningMarks& marks, int from) {
  if (view.rank_at(from)!=Unknown) return 1u << view.rank_at(from);
  return marks.ranks[from];
}

inline std::vector<Move> relay_contact_moves(Observation view, int receiver, int from, int rank) {
  view.ranks[from]=static_cast<int8_t>(rank);
  std::vector<Move> moves;
  gen_piece_moves(view,receiver,from,moves);
  return moves;
}

inline bool relay_has_move(const std::vector<Move>& moves, Move contact) {
  return std::find(moves.begin(),moves.end(),contact)!=moves.end();
}

// Project only the quiet opener. Nothing about enemy replies or the partner's hidden rank
// is needed to establish what changed in the opener's information.
inline std::vector<PublicRelayContact> public_relay_contacts(
    const SeatDecision& decision, const OpeningMarks& marks, Move opening) {
  const auto& before=decision.observation;
  const auto& board=Board::get();
  const int opener=before.viewer;
  if (before.match!=MatchKind::FourSeats || before.turn!=opener ||
      !relay_has_move(decision.actions,opening) || before.seat_at(opening.to)!=kNone ||
      !board.node(opening.from).rail) return {};
  const int receiver=(opener+2)%kSeats;
  const auto after=project_quiet(before,opening);
  constexpr unsigned immobile=(1u << Flag) | (1u << Mine);
  constexpr unsigned engineer=1u << Engineer;
  std::vector<PublicRelayContact> contacts;
  for (int from=0;from<kNodes;++from) {
    if (before.seat_at(from)!=receiver || !board.node(from).rail) continue;
    const unsigned ranks=relay_source_ranks(before,marks,from);
    if (!ranks || (ranks & immobile)) continue;
    const bool ordinary_possible=(ranks & ~engineer)!=0;
    const bool engineer_possible=(ranks & engineer)!=0;
    const auto ordinary_before=ordinary_possible?relay_contact_moves(before,receiver,from,Platoon):std::vector<Move>{};
    const auto ordinary_after=ordinary_possible?relay_contact_moves(after,receiver,from,Platoon):std::vector<Move>{};
    const auto engineer_before=engineer_possible?relay_contact_moves(before,receiver,from,Engineer):std::vector<Move>{};
    const auto engineer_after=engineer_possible?relay_contact_moves(after,receiver,from,Engineer):std::vector<Move>{};
    for (int to=0;to<kNodes;++to) {
      const int owner=after.seat_at(to);
      if (owner<0 || after.team(owner)==after.team(receiver) || !board.node(to).rail) continue;
      const Move contact{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
      if (relay_has_move(ordinary_before,contact)) continue;
      const bool opens_ordinary=relay_has_move(ordinary_after,contact);
      const bool opens_engineer=relay_has_move(engineer_after,contact);
      const bool engineer_was_open=relay_has_move(engineer_before,contact);
      if (opens_ordinary) {
        contacts.push_back({contact,receiver,owner,engineer_was_open?
                            PublicRelayEvidence::EngineerMayAlreadyReach:PublicRelayEvidence::NewContact});
      } else if (opens_engineer && !engineer_was_open) {
        contacts.push_back({contact,receiver,owner,ordinary_possible?
                            PublicRelayEvidence::EngineerOnly:PublicRelayEvidence::NewContact});
      }
    }
  }
  return contacts;
}

inline bool relay_target_touched(const PublicRelayContact& contact, const StepInfo& event) {
  return event.move.from==contact.attack.to ||
         (event.move.to==contact.attack.to && event.outcome!=Outcome::DefenderWins) ||
         (event.eliminated_mask & (1u << contact.target_seat));
}

inline PublicRelayReply public_relay_reply(
    const PublicRelayContact& contact, const SeatDecision& decision, bool target_touched) {
  const auto& view=decision.observation;
  if (view.viewer!=contact.receiver || view.turn!=contact.receiver) return PublicRelayReply::NotReceiverTurn;
  if (target_touched || view.seat_at(contact.attack.to)!=contact.target_seat) return PublicRelayReply::TargetChanged;
  if (!relay_has_move(decision.actions,contact.attack)) return PublicRelayReply::Closed;
  return PublicRelayReply::Available;
}

}
