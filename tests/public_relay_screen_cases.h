#pragma once
#include "selfplay_elimination_cases.h"
#include "../sim/public_relay_screen.h"

namespace jq {

inline uint8_t relay_grid_node(int row, int col) {
  return static_cast<uint8_t>(Board::get().at_grid(row,col));
}

inline State public_relay_board(int hidden_rank, uint8_t target, bool retain_alternate_track) {
  auto state=selfplay_fixture();
  state.out[2]=false;
  const auto& board=Board::get();
  place_single_case_piece(state,50,board.seat_node(2,5,1),Flag);
  place_single_case_piece(state,51,relay_grid_node(8,5),hidden_rank);
  place_single_case_piece(state,1,relay_grid_node(8,8),Platoon);
  place_single_case_piece(state,26,target,Platoon);
  if (!retain_alternate_track) {
    place_single_case_piece(state,2,relay_grid_node(7,5),Platoon);
    place_single_case_piece(state,3,relay_grid_node(9,5),Platoon);
    place_single_case_piece(state,4,relay_grid_node(6,6),Platoon);
    place_single_case_piece(state,5,relay_grid_node(10,6),Platoon);
  }
  return state;
}

inline OpeningMarks relay_test_public_marks(const Observation& view) {
  OpeningMarks marks(view,false);
  const auto origin=relay_grid_node(8,5);
  marks.moved[origin]=true;
  marks.ranks[origin]&=~((1u << Flag) | (1u << Mine));
  return marks;
}

inline void verify_relay_screen_paths(Move opener) {
  const auto source=relay_grid_node(8,5);
  const auto target=relay_grid_node(8,10);
  for (int hidden_rank : {Platoon,Engineer}) {
    auto state=public_relay_board(hidden_rank,target,false);
    const auto decision=seat_decision(state);
    single_case_check(is_legal(state,opener) && decision.observation.rank_at(source)==Unknown,
                      "relay screen needs a legal opener and a hidden receiver");
    const auto ranks=relay_test_public_marks(decision.observation);
    const auto contact=public_relay_contacts(decision,ranks,opener);
    single_case_check(contact.size()==1 && contact[0].attack==Move{source,target} &&
                      contact[0].receiver==2 && contact[0].target_seat==1 &&
                      contact[0].evidence==PublicRelayEvidence::NewContact,
                      "known mobility should open the same contact in both hidden worlds");

    OpeningMarks unqualified(decision.observation,false);
    single_case_check(public_relay_contacts(decision,unqualified,opener).empty(),
                      "unproven receiver mobility should be excluded");
    place_single_case_piece(state,6,relay_grid_node(8,6),Platoon);
    const auto blocked=seat_decision(state);
    single_case_check(public_relay_contacts(blocked,ranks,opener).empty(),
                      "another blocker still prevents the contact");

    state=public_relay_board(hidden_rank,target,true);
    const auto alternate=seat_decision(state);
    const auto ambiguous=public_relay_contacts(alternate,relay_test_public_marks(alternate.observation),opener);
    single_case_check(ambiguous.size()==1 && ambiguous[0].attack==Move{source,target} &&
                      ambiguous[0].evidence==PublicRelayEvidence::EngineerMayAlreadyReach,
                      "ordinary access must distinguish the possible pre-existing engineer detour");

    const auto corner=relay_grid_node(6,10);
    state=public_relay_board(hidden_rank,corner,false);
    const auto turn=seat_decision(state);
    auto turn_marks=relay_test_public_marks(turn.observation);
    const auto conditional=public_relay_contacts(turn,turn_marks,opener);
    single_case_check(conditional.size()==1 && conditional[0].attack==Move{source,corner} &&
                      conditional[0].evidence==PublicRelayEvidence::EngineerOnly,
                      "a route requiring an unconfirmed engineer must be labelled conditional");
    if (hidden_rank==Engineer) {
      turn_marks.ranks[source]=1u << Engineer;
      const auto confirmed=public_relay_contacts(turn,turn_marks,opener);
      single_case_check(confirmed.size()==1 && confirmed[0].attack==Move{source,corner} &&
                        confirmed[0].evidence==PublicRelayEvidence::NewContact,
                        "known engineer evidence should validate the turn");
    }
  }
  std::cout << "PASS public relay screen: two hidden worlds, new, blocked, earlier detour, engineer only, unknown mobility\n";
}

inline void verify_relay_screen_replies(Move opener) {
  const auto target=relay_grid_node(8,10);
  const auto enemy_support=relay_grid_node(6,8);
  auto root=public_relay_board(Platoon,target,false);
  place_single_case_piece(root,27,enemy_support,Engineer);
  const auto decision=seat_decision(root);
  const auto choices=public_relay_contacts(decision,relay_test_public_marks(decision.observation),opener);
  const auto contact=std::find_if(choices.begin(),choices.end(),[=](const auto& c) { return c.attack.to==target; });
  single_case_check(contact!=choices.end() && contact->evidence==PublicRelayEvidence::NewContact,
                    "reply fixture requires a public starting opportunity");
  const auto rules=relay_test_rules();
  single_case_check(is_legal(root,opener),"reply fixture opener must be legal");
  step(root,opener,rules);
  single_case_check(root.to_move==1 &&
                    public_relay_reply(*contact,seat_decision(root),false)==PublicRelayReply::NotReceiverTurn,
                    "response must wait for the actual receiver");
  struct ReplyCase { Move response; PublicRelayReply expected; };
  const ReplyCase cases[]={
      {{enemy_support,relay_grid_node(6,10)},PublicRelayReply::Available},
      {{enemy_support,opener.from},PublicRelayReply::Closed},
      {{target,relay_grid_node(8,11)},PublicRelayReply::TargetChanged},
  };
  for (const auto& trial : cases) {
    auto after=root;
    single_case_check(is_legal(after,trial.response),"response fixture contains an illegal enemy move");
    const auto response=step(after,trial.response,rules);
    single_case_check(after.to_move==2 && after.result==Result::Ongoing &&
                      public_relay_reply(*contact,seat_decision(after),relay_target_touched(*contact,response))==
                        trial.expected,
                      "receiver did not re-evaluate the shared opener after the actual enemy reply");
  }
  std::cout << "PASS public relay replies: identical opener, one still available, one blocked, one target withdrew\n";
}

inline void public_relay_screen_cases() {
  const Move opener{relay_grid_node(8,8),relay_grid_node(10,8)};
  verify_relay_screen_paths(opener);
  verify_relay_screen_replies(opener);
}
}
