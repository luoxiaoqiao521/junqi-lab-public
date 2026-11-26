#include "../sim/regional_arrival.h"
#include "../sim/public_threat_evidence.h"
#include "selfplay_elimination_cases.h"
#include <iostream>

using namespace jq;
constexpr unsigned immobile = (1u << Flag) | (1u << Mine);
constexpr unsigned attacker_wins = 1u << static_cast<unsigned>(Outcome::AttackerWins);

// Use only the player's own flag or an ally flag already visible to this player.
static bool exposed_flag(const Observation& view,const OpeningMarks& marks) {
  const auto& board=Board::get();
  for (int flag=0;flag<kNodes;++flag) {
    const int owner=view.seat_at(flag);
    if (owner==kNone || view.team(owner)!=view.team(view.viewer) || view.rank_at(flag)!=Flag) continue;
    for (const int origin : board.road(flag)) {
      const int entrant=view.seat_at(origin);
      if (entrant!=kNone && view.team(entrant)!=view.team(view.viewer) &&
          can_leave_node(Engineer,origin) && (compatible_rank_mask(view,marks,origin)&~immobile)) return true;
    }
  }
  return false;
}

static int guaranteed_victim(const SeatDecision& d,const OpeningMarks& marks,Move action,const Rules& rules) {
  const auto& view=d.observation;
  if (view.seat_at(action.to)==kNone || !can_leave_node(Engineer,action.to)) return -1;
  const auto rivals=compatible_rank_mask(view,marks,action.to);
  if (!rivals || (rivals&immobile) ||
      collision_outcome_mask(view.rank_at(action.from),rivals,rules)!=attacker_wins) return -1;
  if (exposed_flag(project_quiet(view,action),marks)) return -1;
  for (int rank=Bomb;rank<RankCount;++rank) if (rivals&(1u<<rank)) return rank;
  return -1;
}

static Move advance_if_safe(const SeatDecision& decision,const OpeningMarks& marks,Move fallback,
                            const Rules& rules,int quiet) {
  if (quiet<12 || exposed_flag(decision.observation,marks) ||
      guaranteed_victim(decision,marks,fallback,rules)>=0) return fallback;
  int best_rank=-1;
  Move result=fallback;
  for (const auto move : decision.actions) {
    const int victim=guaranteed_victim(decision,marks,move,rules);
    if (victim>best_rank) {
      best_rank=victim;
      result=move;
    }
  }
  return result;
}

static void fixtures(const Rules& rules) {
  auto state=selfplay_fixture();
  place_single_case_piece(state,1,4,Marshal);
  place_single_case_piece(state,26,30,Platoon);
  const Move quiet{4,3},cap{4,30};
  const auto decision=seat_decision(state);
  single_case_check(is_legal(state,quiet) && is_legal(state,cap),"tempo fixtures must be legal");
  OpeningMarks marks(decision.observation,false);
  marks.ranks[30]=1u<<Platoon;
  single_case_check(advance_if_safe(decision,marks,quiet,rules,11)==quiet &&
                    advance_if_safe(decision,marks,quiet,rules,12)==cap,
                    "guaranteed contact did not respect the quiet window");
  marks.ranks[30]|=1u<<Bomb;
  single_case_check(advance_if_safe(decision,marks,quiet,rules,12)==quiet,
                    "possible bomb did not reject the contact");
  marks.ranks[30]=1u<<Platoon;
  place_single_case_piece(state,27,21,Platoon);
  marks.ranks[21]=1u<<Platoon;
  single_case_check(advance_if_safe(seat_decision(state),marks,quiet,rules,12)==quiet,
                    "flag threat did not retain the baseline decision");
  std::cerr<<"PASS quiet window, guaranteed collision, bomb uncertainty, own flag exposure\n";
}

static State setup(uint64_t seed,const Rules& rules) {
  Rng rng(seed);
  int8_t layout[kSeats][kPiecesPerSeat];
  for (auto& seat : layout) generate_layout(seat,rng);
  State state;
  init_state(state,layout,rules.random_first_seat?int(rng.below(kSeats)):0,rules);
  return state;
}

static void game(uint64_t seed,int learner,int table,bool experiment,const Rules& rules,
                 const ShortWeights& own,const ShortWeights& peer) {
  auto state=setup(seed,rules);
  const int first=state.to_move;
  auto streams=review_streams(seed);
  OpeningMarks marks(observe(state,learner),true);
  int overrides=0, enemy_retired=0, self_retired=0;
  EndReason terminal=EndReason::None;
  while (state.result==Result::Ongoing) {
    const auto decision=seat_decision(state);
    const int seat=state.to_move;
    auto& rng=streams[seat];
    Move move;
    if (seat==learner) {
      const auto fallback=short_move(decision.observation,decision.actions,own,rng);
      move=experiment?advance_if_safe(decision,marks,fallback,rules,state.no_progress):fallback;
      overrides+=!(move==fallback);
    } else {
      move=table?short_move(decision.observation,decision.actions,peer,rng):
                 decision.actions.at(rng.below(static_cast<uint32_t>(decision.actions.size())));
    }
    const bool record=experiment && !state.out[learner];
    const auto before=record?observe(state,learner):Observation{};
    const auto event=step(state,move,rules);
    if (record) advance_contact_ranks(marks,before,observe(state,learner),move,event.outcome,rules);
    for (int exited=0;exited<kSeats;++exited) if (event.eliminated_mask & (1u<<exited)) {
      int& earliest=team_of(learner)==team_of(exited)?self_retired:enemy_retired;
      if (!earliest) earliest=state.ply;
    }
    terminal=event.end_reason;
  }
  std::cout<<seed<<' '<<learner<<' '<<first<<' '<<table<<' '<<experiment<<' '<<state.ply<<' '
           <<review_outcome(state.result,learner)<<' '<<int(terminal)<<' '<<overrides<<' '
           <<enemy_retired<<' '<<self_retired<<'\n';
}

int main(int argc,char** argv) {
  try {
    if (argc!=2) return 2;
    Rules rules;
    std::string issue;
    if (!parse_rules(read_text(argv[1]),rules,issue) || !rules.eliminate_exhausted_movers)
      throw std::runtime_error("invalid selfplay rules");
    fixtures(rules);
    uint64_t start,count;
    ShortWeights own{},peer{};
    if (!(std::cin>>start>>count) || !count || count>64 || start>UINT64_MAX-count) return 2;
    for (auto target : {&own,&peer})
      for (size_t i=0;i<5;++i) if (!(std::cin>>(*target)[i]) || !std::isfinite((*target)[i])) return 2;
    for (int table : {0,1}) for (bool variant : {false,true})
      for (uint64_t seed=start;seed<start+count;++seed) for (int learner=0;learner<kSeats;++learner)
        game(seed,learner,table,variant,rules,own,peer);
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
