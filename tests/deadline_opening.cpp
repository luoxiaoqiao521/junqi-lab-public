#include "../sim/regional_arrival.h"
#include "../sim/public_threat_evidence.h"
#include "selfplay_elimination_cases.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include "deadline_repeat.h"

using namespace jq;
constexpr unsigned immobile_ranks=(1u<<Flag)|(1u<<Mine);
constexpr unsigned outcome_bit(Outcome outcome) { return 1u<<static_cast<unsigned>(outcome); }

struct PublicTurn {
  unsigned occupied=0;
  int teams[kSeats]{};
};

static PublicTurn occupied_seats(const Observation& view) {
  PublicTurn result;
  for (int n=0;n<kNodes;++n) {
    const int seat=view.seat_at(n);
    if (seat>=0) result.occupied|=1u<<seat;
  }
  for (int seat=0;seat<kSeats;++seat)
    if (result.occupied&(1u<<seat)) ++result.teams[view.team(seat)];
  return result;
}

static bool flag_exposure(const Observation& view,const OpeningMarks& marks,int flag) {
  if (flag<0) return false;
  for (const int n : Board::get().road(flag)) {
    const int owner=view.seat_at(n);
    if (owner>=0 && view.team(owner)!=view.team(view.viewer) && can_leave_node(Engineer,n) &&
        (compatible_rank_mask(view,marks,n)&~immobile_ranks)) return true;
  }
  return false;
}

static std::vector<int> flag_targets(const Observation& view,const OpeningMarks& marks) {
  std::vector<int> result;
  const auto& board=Board::get();
  for (int n=0;n<kNodes;++n) {
    const int owner=view.seat_at(n);
    if (owner>=0 && view.team(owner)!=view.team(view.viewer) && board.node(n).kind==NodeKind::HQ &&
        (compatible_rank_mask(view,marks,n)&(1u<<Flag))) result.push_back(n);
  }
  return result;
}

static int flag_distance(int node,const std::vector<int>& targets) {
  int value=40;
  for (const int target : targets) value=std::min(value,distance(node,target));
  return value;
}

static int own_flag(const Observation& view) {
  for (int n=0;n<kNodes;++n) if (view.seat_at(n)==view.viewer && view.rank_at(n)==Flag) return n;
  return kNone;
}

static Observation contact_view(const Observation& view,Move action,Outcome outcome) {
  auto after=view;
  after.seats[action.from]=kNone;
  after.ranks[action.from]=Unknown;
  if (outcome==Outcome::BothRemoved) {
    after.seats[action.to]=kNone;
    after.ranks[action.to]=Unknown;
  } else if (outcome==Outcome::AttackerWins || outcome==Outcome::Move) {
    after.seats[action.to]=view.seats[action.from];
    after.ranks[action.to]=view.ranks[action.from];
  }
  return after;
}

struct Effort { int advance, risk, retreat; };

static Effort urgency(int ply,int friendly,int foes) {
  if (friendly==1 && foes==2) return {1,14,6};
  if (ply<100 && (ply>=64 || foes==1)) return {4,5,3};
  if (ply<140) return {3,8,4};
  return {2,15,6};
}

static int action_value(Move action,unsigned outcomes,unsigned defender,
                        const std::vector<int>& targets,const Effort& density,Move last) {
  if (outcomes==outcome_bit(Outcome::DefenderWins)) return -1000;
  const int remaining=flag_distance(action.to,targets);
  const int forward=flag_distance(action.from,targets)-remaining;
  int score=density.advance*forward;
  if (action.to==last.from && action.from==last.to) score-=density.retreat;
  if (Board::get().node(action.to).kind==NodeKind::Camp) score+=2;
  if (remaining==1) score+=6;
  if (outcomes==outcome_bit(Outcome::Move)) return score;
  if (defender==(1u<<Flag)) return 1000;
  if (outcomes&outcome_bit(Outcome::AttackerWins)) score+=7;
  if (outcomes==(outcome_bit(Outcome::AttackerWins))) score+=11;
  if (defender&(1u<<Flag)) score+=13;
  if (outcomes&(outcome_bit(Outcome::BothRemoved)|outcome_bit(Outcome::DefenderWins))) score-=density.risk;
  return score;
}

static Move plan(const SeatDecision& d,const OpeningMarks& marks,Move baseline,Move last,
                 const Rules& rules,std::vector<std::pair<Move,int>>* scores=nullptr) {
  const auto& view=d.observation;
  const auto targets=flag_targets(view,marks);
  if (targets.empty()) return baseline;
  const auto active=occupied_seats(view);
  const int team=view.team(view.viewer), flag=own_flag(view);
  const auto effort=urgency(view.ply,active.teams[team],active.teams[1-team]);
  int movers=0;
  for (int n=0;n<kNodes;++n)
    if (view.seat_at(n)==view.viewer && can_leave_node(view.rank_at(n),n)) ++movers;
  int best=-1001;
  Move selected=baseline;
  for (const auto action : d.actions) {
    const bool quiet=view.seat_at(action.to)<0;
    const unsigned candidates=quiet?0:compatible_rank_mask(view,marks,action.to);
    if (!quiet && !candidates) continue;
    if (movers==1 && Board::get().node(action.to).kind==NodeKind::HQ) {
      const bool wins=candidates==(1u<<Flag) && active.teams[1-team]==1;
      if (!wins) continue;
    }
    const unsigned outcomes=quiet?outcome_bit(Outcome::Move):
        collision_outcome_mask(view.rank_at(action.from),candidates,rules);
    if (outcomes==outcome_bit(Outcome::DefenderWins)) continue;
    bool hazard=false;
    for (const auto result : {Outcome::Move,Outcome::AttackerWins,Outcome::DefenderWins,Outcome::BothRemoved}) {
      if ((outcomes&outcome_bit(result)) && flag_exposure(contact_view(view,action,result),marks,flag)) {
        hazard=true;
        break;
      }
    }
    const bool immediately_wins=rules.flag_eliminates && candidates==(1u<<Flag) &&
                                active.teams[1-team]==1;
    if (hazard && !immediately_wins) continue;
    const int value=action_value(action,outcomes,candidates,targets,effort,last);
    if (scores) scores->emplace_back(action,value);
    if (value>best || (value==best && action==baseline)) {selected=action; best=value;}
  }
  return selected;
}

#include "deadline_gate.h"
#include "deadline_rollout.h"
#include "deadline_public_reply.h"
#include "deadline_conditional_reply.h"
#include "deadline_bomb.h"
#include "deadline_tail.h"
#include "deadline_learning_root.h"

static void gate_fixtures(const Rules& rules) {
  auto state=selfplay_fixture();
  state.out[2]=false;
  place_single_case_piece(state,1,4,Marshal);
  place_single_case_piece(state,2,51,Brigade);
  place_single_case_piece(state,26,46,Platoon);
  place_single_case_piece(state,50,86,Flag);
  place_single_case_piece(state,51,50,Platoon);
  const auto d=seat_decision(state);
  OpeningMarks marks(d.observation,false);
  marks.ranks[56]=1u<<Flag;
  marks.ranks[50]=(1u<<Platoon)|(1u<<Engineer);
  const Move wait{4,3},handoff{50,51};
  single_case_check(is_legal(state,wait),"gate fixture baseline is illegal");
  const auto proposal=gate_plan(d,marks,wait);
  single_case_check(proposal && proposal->receive==handoff && proposal->prepare.from==51,
                    "vacating the gate didn't create a public relay");
  auto hidden=state;
  hidden.rank_at[50]=hidden.rank_of[51]=Engineer;
  const auto same=gate_plan(seat_decision(hidden),marks,wait);
  single_case_check(same && same->prepare==proposal->prepare && same->receive==proposal->receive,
                    "hidden partner rank changed public choice");
  auto alternate=state;
  step(alternate,proposal->prepare,rules);
  const auto reply=seat_decision(alternate);
  single_case_check(reply.observation.viewer==1,"expected intervening enemy in gate fixture");
  GateTrial result{proposal,0,1,1,false};
  for (const auto move:reply.actions) if (move.to==handoff.to) {
    auto closed=alternate;
    step(closed,move,rules);
    result.turn(seat_decision(closed),Move{});
    single_case_check(result.answer==2,"enemy reply did not close the projected open gate");
    return;
  }
  single_case_check(false,"gate fixture lacked a legal blocker");
}

static void fixtures(const Rules& rules) {
  auto state=selfplay_fixture();
  place_single_case_piece(state,1,4,Marshal);
  place_single_case_piece(state,26,30,Platoon);
  const auto d=seat_decision(state);
  OpeningMarks marks(d.observation,false);
  marks.ranks[56]=1u<<Flag;
  marks.ranks[30]=(1u<<Platoon)|(1u<<Engineer);
  const Move wait{4,3},capture{4,30};
  single_case_check(plan(d,marks,wait,Move{},rules)==capture,"known early capture fixture failed");
  const auto early=urgency(80,2,2), late=urgency(140,2,2), danger=urgency(80,1,2);
  single_case_check(early.advance>late.advance && early.risk<late.risk && danger.advance<early.advance,
                    "deadline and reserve conditions do not change willingness to risk");
  auto swapped=state;
  swapped.rank_at[30]=swapped.rank_of[26]=Engineer;
  single_case_check(plan(seat_decision(swapped),marks,wait,Move{},rules)==capture,
                    "swapping a hidden piece changed selection under identical observations");
  std::cerr<<"PASS early capture, phase risk, and hidden-identity isolation\n";
}

static State opening(uint64_t seed,const Rules& rules) {
  Rng rng(seed);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& row : layouts) generate_layout(row,rng);
  State state;
  init_state(state,layouts,rules.random_first_seat?int(rng.below(kSeats)):0,rules);
  return state;
}

#include "deadline_clock_learning.h"
#include "deadline_early_labels.h"
#include "deadline_discovery.h"
#include "deadline_evaluator_bounds.h"
#include "deadline_evaluator_panel.h"
#include "deadline_first_exit_coverage.h"
#include "deadline_first_exit_approach.h"
#include "deadline_student.h"
#include "deadline_exchange_reply.h"
#include "deadline_conversion_cross.h"
#include "deadline_conversion_history.h"
#include "deadline_blind_extension.h"
#include "deadline_winner_takeover.h"
#include "deadline_conversion_labels.h"
#include "deadline_takeover_delay.h"
#include "deadline_post_retirement.h"
#include "deadline_replay_phase.h"
#include "deadline_capture_replies.h"
#include "deadline_capture_coverage.h"
#include "deadline_directed_coverage.h"
#include "deadline_visible_learning.h"
#include "deadline_full_mode_baseline.h"
#include "deadline_dual_pressure.h"
#include "deadline_archive_pressure.h"
#include "deadline_formation.h"
#include "deadline_route_choice.h"
#include "deadline_natural_routes.h"
#include "deadline_team_relay.h"
#include "deadline_visible_terminal.h"
#include "deadline_relay_setup.h"
#include "deadline_reply_budget.h"
#include "deadline_relay_chain.h"
#include "deadline_counterfactual_reply.h"
#include "deadline_reply_window_learning.h"
#include "deadline_dynamic_reply.h"
#include "deadline_full_student_game.h"
#include "deadline_engineer_formation.h"
#include "deadline_quiet_options.h"
#include "deadline_reply_value_audit.h"
#include "deadline_leaf_teacher.h"
#include "deadline_scoring_audit.h"
#include "deadline_dark_info_audit.h"
#include "deadline_dark_bomb_belief.h"
#include "deadline_dark_search_budget.h"
#include "deadline_dark_formation_library.h"
#include "deadline_dark_reply_leaf.h"
#include "deadline_dark_opening_reply.h"
#include "deadline_dark_witness_audit.h"
#include "deadline_dark_ordinary_learning.h"
#include "deadline_public_flag_reply.h"
#include "deadline_dark_course_tape.h"
#include "deadline_dark_first_loss.h"
#include "deadline_dark_partner_support.h"
#include "deadline_dark_replanning.h"
#include "deadline_dark_response_distribution.h"
#include "deadline_dark_seat_candidates.h"
#include "deadline_dark_stratified.h"
#include "deadline_dark_broad_pairs.h"
#include "deadline_dark_partner_evidence.h"
#include "deadline_dark_heritage_nominations.h"
#include "deadline_dark_gate_timing.h"
#include "deadline_reply_diversity.h"
#include "deadline_product_shadow.h"
#include "deadline_collision_semantics.h"

static bool controlled(int arm,int seat,int learner) {
  return arm==3 || (arm>=1 && seat==learner) ||
         ((arm==2 || arm>=4) && team_of(seat)==team_of(learner));
}

static void trial(uint64_t seed,int learner,int table,int arm,const Rules& rules,
                  const ShortWeights& parent,const ShortWeights& history,bool tail_audit=false,int partner=-1,
                  bool export_learning=false,bool bomb_test=false,bool book_test=false,bool hard_learning=false) {
  auto state=opening(seed,rules);
  const int first=state.to_move;
  auto streams=review_streams(seed);
  std::array<OpeningMarks,kSeats> marks{OpeningMarks(observe(state,0),true),OpeningMarks(observe(state,1),true),
                                       OpeningMarks(observe(state,2),true),OpeningMarks(observe(state,3),true)};
  std::array<Move,kSeats> last{};
  int override=0, first_enemy=0, first_friendly=0, first_contact=0, changes_100=0, changes_140=0;
  uint64_t pre_exit_trace=seed;
  GateTrial gate{std::nullopt,seed};
  ResponseWork work;
  BombWork bombs;
  int response_windows[2]{}, first_window=0, public_clock=0;
  uint64_t before_forecast=seed, event_trace=seed;
  const bool first_exit_run=arm>=11;
  const bool response_run=arm==9 || arm==10 || first_exit_run;
  const bool keep_reply_history=arm==10 || arm>=12;
  PublicReplyHistory reply_history(state,true);
  const auto routes=arm==18?RoutePriority::FirstThenContinue:
                    (arm==17 || arm==19 || arm==20 || arm==22)?RoutePriority::Joint:RoutePriority::Old;
  TailWindow tail;
  QuietRevisit revisit;
  if (arm==5) revisit.begin(observe(state,learner));
  const auto adapted=[&](int seat) { return controlled(arm,seat,learner) && (partner<=0 || seat==learner); };
  EndReason end=EndReason::None;
  while (state.result==Result::Ongoing) {
    const int seat=state.to_move;
    const auto d=seat_decision(state);
    const bool ally=seat==(learner+2)%kSeats;
    const auto base=seat==learner?short_move(d.observation,d.actions,parent,streams[seat]):
      ally && partner==2?review_move(d,learner,1,true,parent,streams[seat]):
      (table || (ally && partner==1))?short_move(d.observation,d.actions,history,streams[seat]):
      d.actions.at(streams[seat].below(static_cast<uint32_t>(d.actions.size())));
    const bool active=adapted(seat) && (arm!=4 || !first_enemy);
    const bool opposing_pressure=hard_learning && team_of(seat)!=team_of(learner);
    auto action=(active || opposing_pressure)?plan(d,marks[seat],base,last[seat],rules):base;
    if (arm==7 || arm==8) {
      gate.turn(d,action);
      if (!gate.offered && active && first_enemy) {
        gate.offered=gate_plan(d,marks[seat],action);
        if (gate.offered) {
          gate.offered_at=state.ply+1;
          gate.answer=1;
          if (arm==8) action=gate.offered->prepare;
        }
      }
    }
    const bool early_forecast=!first_enemy && first_exit_run && state.ply>=12 && state.ply<100;
    const bool late_forecast=first_enemy && (!first_exit_run || (arm>=13 && first_window)) && state.ply<280;
    if (response_run && active && (early_forecast || late_forecast) && d.actions.size()>1) {
      int& windows=response_windows[late_forecast];
      if (windows<26 && windows%4==0) {
        if (!first_window) first_window=state.ply+1;
        if ((arm==10 || arm>=12) && !((arm==14 || arm==21) && tail.at)) {
          const auto proposed=response_choice(d,marks[seat],reply_history,action,rules,history,public_clock,work,
                                              early_forecast,routes,arm==19 && early_forecast,
                                              arm==20 && early_forecast,arm==22);
          if (!(proposed==action)) {
            if (export_learning && !tail.at)
              print_learning_root(std::cout,d,reply_history.for_actor(d.observation,marks[seat]),public_clock,
                                  seed,learner,table,event_trace);
            const bool skip_first=arm>=15 && !tail.at;
            tail.offer(state.ply,seat,action,proposed,event_trace);
            if (!skip_first) {
              action=proposed;
              ++tail.applied;
              if (work.chose_book) {
                ++work.book_applied;
                if (!work.first_book) work.first_book=state.ply+1;
              }
            }
          }
        } else ++work.evaluations;
      }
      ++windows;
    }
    if (bomb_test && active && d.observation.rank_at(action.from)==Bomb && d.observation.seat_at(action.to)>=0) {
      ++bombs.offered;
      if (!bombs.first) { bombs.first=state.ply+1; bombs.prefix=event_trace; }
      if (arm==16) {
        const auto guarded=bomb_balance(d,marks[seat],reply_history,action,last[seat],rules,history,public_clock,bombs);
        bombs.changed+=!(guarded==action);
        action=guarded;
      }
    }
    if (arm==5 && first_enemy && active) revisit.note(d.observation,marks[seat],action,rules);
    if (!(action==base)) {
      ++override;
      changes_100+=state.ply<100;
      changes_140+=state.ply<140;
    }
    last[seat]=action;
    auto before=state;
    const auto event=step(state,action,rules);
    if (keep_reply_history) reply_history.advance(before,state,event,rules);
    if (bomb_test && active && before.rank_at[action.from]==Bomb && event.outcome!=Outcome::Move) {
      const int rank=before.rank_at[action.to];
      ++bombs.contacts;
      bombs.low+=rank>=Engineer && rank<Division;
      bombs.large+=rank>=Division;
      bombs.mine+=rank==Mine;
      bombs.flag+=rank==Flag;
    }
    const uint64_t event_key=uint64_t(action.from) | (uint64_t(action.to)<<8) | (uint64_t(seat)<<16) |
                             (uint64_t(event.outcome)<<18) | (uint64_t(event.eliminated_mask)<<22);
    if ((arm==7 || arm==8) && !gate.offered) gate.prefix=mix_seed(gate.prefix,event_key);
    if (response_run) {
      public_clock=event.outcome==Outcome::Move?public_clock+1:0;
      single_case_check(public_clock==state.no_progress,"rollout quiet clock drifted from the public actions");
      event_trace=mix_seed(event_trace,event_key);
      if (!first_window) before_forecast=event_trace;
      if (tail_audit) tail.observe(event,state.ply,public_clock);
    }
    if (arm==5) {
      revisit.advance(action,event,observe(state,learner));
      single_case_check(revisit.quiet_clock==state.no_progress,"public quiet clock drifted");
    }
    if (!first_enemy) pre_exit_trace=mix_seed(pre_exit_trace,event_key);
    for (int viewer=0;viewer<kSeats;++viewer)
      if (!state.out[viewer] && (adapted(viewer) || (hard_learning && team_of(viewer)!=team_of(learner))))
        advance_contact_ranks(marks[viewer],observe(before,viewer),observe(state,viewer),action,event.outcome,rules);
    for (int exited=0;exited<kSeats;++exited) if (event.eliminated_mask&(1u<<exited)) {
      int& at=team_of(exited)==team_of(learner)?first_friendly:first_enemy;
      if (!at) at=state.ply;
    }
    if (!first_contact && event.outcome!=Outcome::Move) first_contact=state.ply;
    end=event.end_reason;
  }
  std::cout<<seed<<' '<<learner<<' '<<first<<' '<<table<<' '<<arm<<' '<<state.ply<<' '
           <<review_outcome(state.result,learner)<<' '<<int(end)<<' '<<override<<' '<<changes_100
           <<' '<<changes_140<<' '<<first_contact<<' '<<first_enemy<<' '<<first_friendly
           <<' '<<pre_exit_trace;
  if (arm==5) std::cout<<' '<<revisit.turns<<' '<<revisit.quiet<<' '<<revisit.reversals
                       <<' '<<revisit.repeat<<' '<<revisit.persistent<<' '<<revisit.near_draw;
  if (arm==7 || arm==8) std::cout<<' '<<gate.offered_at<<' '<<gate.answer<<' '<<gate.prefix;
  if (response_run) std::cout<<' '<<first_window<<' '<<work.evaluations<<' '<<work.changed
    <<' '<<work.early_changes<<' '<<work.absent<<' '<<work.samples<<' '<<work.terminal_wins<<' '<<work.terminal_losses
    <<' '<<before_forecast<<' '<<event_trace;
  if (tail_audit) tail.write(std::cout);
  if (book_test) std::cout<<' '<<work.book_offered<<' '<<work.book_selected<<' '<<work.book_applied<<' '<<work.first_book;
  if (bomb_test) std::cout<<' '<<bombs.first<<' '<<bombs.prefix<<' '<<bombs.offered<<' '<<bombs.considered
                        <<' '<<bombs.kept_critical<<' '<<bombs.changed<<' '<<bombs.contacts<<' '<<bombs.low
                        <<' '<<bombs.large<<' '<<bombs.mine<<' '<<bombs.flag;
  if (partner>=0) std::cout<<' '<<partner;
  std::cout<<'\n';
}

int main(int argc,char** argv) {
  try {
    const std::string mode=argc==3?argv[2]:"";
    const bool active=active_full_research();
    if (active) {
      const std::string_view accepted[]{"full-leaf-teacher-data","visited-reply-value","visited-lesson-review",
        "full-scoring-audit",
        "dark-info-audit",
        "dark-bomb-belief",
        "dark-search-budget",
        "dark-formation-library",
        "dark-formation-sources",
        "dark-reply-leaf",
        "dark-reply-course",
        "dark-opening-reply",
        "dark-opening-reply-window",
        "dark-witness-audit",
        "dark-ordinary-learning",
        "dark-course-tape-replay",
        "dark-course-openings",
        "dark-first-loss",
        "dark-partner-support",
        "dark-replanning",
        "dark-replanning-matched",
        "dark-replanning-course",
        "dark-response-distribution",
        "dark-seat-candidates",
        "dark-stratified",
        "dark-broad-pairs",
        "dark-partner-evidence",
        "dark-heritage-nominations",
        "dark-heritage-action-nominations",
        "dark-heritage-forced-contact",
        "dark-gate-timing",
        "reply-window-examples","reply-window-judge-board","reply-window-judge-window","full-student-evolution-games"};
      if (std::find(std::begin(accepted),std::end(accepted),mode)==std::end(accepted)) {
        std::cerr<<"JUNQI_ACTIVE_DENIED "<<mode<<'\n';
        return 86;
      }
    }
    const bool mixed=mode=="partner-mix";
    const bool heldout_reply=mode=="confirm-critical-replies";
    const bool critical_reply=mode=="audit-critical-replies" || heldout_reply;
    const bool calibration=mode=="calibrate-reply-evaluator";
    const bool evaluator_bounds=mode=="prove-evaluator-controls";
    const bool visible_learning=mode=="audit-visible-learning";
    const bool student_capture_replies=mode=="student-capture-replies";
    const bool capture_coverage=mode=="capture-world-coverage";
    const bool directed_coverage=mode=="directed-capture-coverage";
    const bool dark_info_audit=mode=="dark-info-audit";
    const bool dark_bomb_belief=mode=="dark-bomb-belief";
    const bool dark_search_budget=mode=="dark-search-budget";
    const bool dark_formation_library=mode=="dark-formation-library" || mode=="dark-formation-sources";
    const bool dark_reply_leaf=mode=="dark-reply-leaf" || mode=="dark-reply-course";
    const bool dark_opening_reply=mode=="dark-opening-reply" || mode=="dark-opening-reply-window";
    const bool dark_witness=mode=="dark-witness-audit";
    const bool dark_ordinary=mode=="dark-ordinary-learning";
    const bool flag_reply=mode=="public-flag-reply-controls" || mode=="public-flag-reply-games" ||
                          mode=="public-flag-reply-archive" || mode=="public-flag-reply-attribute";
    const bool dark_course_openings=mode=="dark-course-openings";
    const bool dark_course_tape=mode=="dark-course-tape-replay" || dark_course_openings;
    const bool dark_first_loss=mode=="dark-first-loss";
    const bool dark_partner_support=mode=="dark-partner-support";
    const bool dark_replanning=mode=="dark-replanning" || mode=="dark-replanning-matched" || mode=="dark-replanning-course";
    const bool dark_response_distribution=mode=="dark-response-distribution";
    const bool dark_seat_candidates=mode=="dark-seat-candidates";
    const bool dark_stratified=mode=="dark-stratified";
    const bool dark_broad_pairs=mode=="dark-broad-pairs";
    const bool dark_partner_evidence=mode=="dark-partner-evidence";
    const bool force_heritage_contact=mode=="dark-heritage-forced-contact";
    const bool action_level_nomination=mode=="dark-heritage-action-nominations" || force_heritage_contact;
    const bool dark_heritage_nominations=mode=="dark-heritage-nominations" || action_level_nomination;
    const bool dark_gate_timing=mode=="dark-gate-timing";
    const bool full_mode_baseline=mode=="full-mode-baseline";
    const bool dual_pressure=mode=="dual-pressure";
    const bool archive_pressure=mode=="archive-pressure";
    const bool formation=mode=="formation-matrix" || mode=="formation-rules" || mode=="formation-collision-impact";
    const bool route_choice=mode=="route-panels" || mode=="route-replays" || mode=="route-resources";
    const bool natural_routes=mode=="natural-routes";
    const bool team_relay=mode=="team-relay";
    const bool visible_terminal=mode=="visible-terminal";
    const bool relay_setup=mode=="relay-setup";
    const bool collision_semantics=mode=="collision-semantics";
    const bool gate_trace=mode=="trace-evaluator-gate";
    const bool evaluator_panel=mode=="compare-evaluator-panel" || gate_trace;
    const bool reply_budget=mode=="reply-budget";
    const bool relay_chain=mode=="relay-chain";
    const bool counterreply=mode=="counterreply-audit";
    const bool reply_examples=mode=="reply-window-examples";
    const bool reply_judge=mode=="reply-window-judge-window" || mode=="reply-window-judge-board";
    const bool reply_window_games=mode=="reply-window-games";
    const bool frozen_reply=mode=="frozen-reply-audit";
    const bool scheduled_reply=mode=="scheduled-reply-games";
    const bool ordinary_classroom=mode=="ordinary-reply-classroom";
    const bool evolution_trace=mode=="full-student-evolution-trace";
    const bool engineer_trace=mode=="engineer-student-formation-trace";
    const bool evolution_games=mode=="full-student-evolution-games" || evolution_trace;
    const bool engineer_formation=mode=="engineer-student-formation" || engineer_trace;
    const bool quiet_options=mode=="quiet-options-audit";
    const bool reply_value_audit=mode=="visited-reply-value";
    const bool lesson_review=mode=="visited-lesson-review";
    const bool preference_games=mode=="full-student-preference-games";
    const bool leaf_recheck=mode=="full-leaf-teacher-recheck";
    const bool leaf_teacher=mode=="full-leaf-teacher-data" || leaf_recheck;
    const bool diversity_panel=mode=="reply-diversity-cross";
    const bool product_shadow=mode=="product-shadow-replay";
    const bool scoring_audit=mode=="full-scoring-audit";
    const bool first_exit_audit=mode=="first-exit-coverage";
    const bool first_exit_pair=mode=="first-exit-approach" || mode=="first-exit-approach-control";
    const bool conversion_cross=mode=="cross-conversion";
    const bool conversion_history=mode=="export-conversion-history" || mode=="restore-conversion-history";
    const bool blind_extension=mode=="blind-extension";
    const bool replay_phase=mode=="replay-phase-split" || mode=="replay-phase-witness";
    const bool student_labels=mode=="student-label-locked" || mode=="student-label-holdout";
    const bool confirm_labels=mode=="conversion-label-locked" || mode=="conversion-label-holdout" || student_labels;
    const bool conversion_labels=mode=="conversion-label-pilot" || mode=="conversion-label-fresh" || confirm_labels;
    const bool reply_counterfactual=mode=="discover-reply-counterfactuals" || calibration;
    const bool discovery=mode=="discover-public-actions" || critical_reply || reply_counterfactual;
    const bool bomb_test=mode=="bomb-balance";
    const bool staged_routes=mode=="staged-target-path";
    const bool book_cover=mode=="book-cover";
    const bool book_test=mode=="book-candidate" || book_cover;
    const bool joint_routes=mode=="two-target-path" || staged_routes;
    const bool export_learning=mode=="learning-roots";
    const bool hard_learning=mode=="learning-hard-outcomes";
    const bool public_reply=mode=="public-reply-proof";
    const bool conditional_reply=mode=="conditional-reply-proof";
    const bool challenger_race=mode=="challenger-race";
    const bool early_heuristic=mode=="early-teacher-heuristic";
    const bool recovery_only=mode=="exchange-natural-recovery";
    const bool winner_delay=mode=="winner-delay-audit";
    const bool winner_confirm=mode=="winner-finish-confirm";
    const bool post_evaluation=mode=="post-retirement-openings";
    const bool post_training=mode=="post-retirement-labels";
    const bool winner_takeover=mode=="winner-takeover" || winner_confirm;
    const bool follow_archive=mode=="exchange-archive", follow_natural=mode=="exchange-natural" || recovery_only;
    const bool early_labels=mode=="early-teacher-labels" || early_heuristic;
    const bool counterplay_test=mode=="active-counterplay-forecast";
    const bool exchange=mode=="student-apply-exchange" || mode=="student-features-exchange";
    const bool student_contact=mode=="student-apply-contact" || mode=="student-reference-contact" ||
                               mode=="student-apply-exchange";
    const bool student_eval=mode=="student-apply" || mode=="student-apply-contact" || mode=="student-apply-exchange";
    const bool student_control=mode=="student-reference" || mode=="student-reference-contact";
    const bool reseeded_visit=mode=="student-feedback-reseeded";
    const bool student_visits=mode=="student-feedback-visit" || reseeded_visit, student_frozen=mode=="student-feedback-frozen";
    const bool student_features_only=mode=="student-features" || mode=="student-features-exchange";
    const bool contact_control=mode=="clock-contact-control";
    const bool clock_fit=mode=="clock-action-apply" || contact_control;
    const bool clock_test=clock_fit || mode=="clock-action-pairs";
    const bool tail_audit=mode=="long-draw" || mixed || export_learning || hard_learning;
    const bool has_forecast=mode=="response" || mode=="first-exit" || mode=="combined" ||
                            tail_audit || bomb_test || joint_routes || book_test || challenger_race ||
                            early_labels || counterplay_test || student_visits || student_frozen || discovery;
    if (argc!=2 && mode!="conversion" && mode!="repeat" && mode!="gate" &&
        !has_forecast && !public_reply && !conditional_reply && !clock_test &&
        !student_eval && !student_control && !student_features_only && !follow_natural && !follow_archive &&
        !winner_takeover && !winner_delay && !post_evaluation && !post_training && !evaluator_bounds && !evaluator_panel &&
        !first_exit_audit && !first_exit_pair && !conversion_cross && !conversion_history && !blind_extension && !conversion_labels &&
        !replay_phase && !visible_learning && mode!="capture-replies" && !student_capture_replies &&
        !capture_coverage && !directed_coverage && !dark_info_audit && !dark_bomb_belief && !dark_search_budget &&
        !dark_formation_library && !dark_reply_leaf && !dark_opening_reply && !dark_witness && !dark_ordinary && !dark_course_tape && !dark_first_loss && !dark_partner_support && !dark_replanning &&
        !dark_response_distribution && !dark_seat_candidates && !dark_stratified && !dark_broad_pairs && !dark_partner_evidence && !dark_heritage_nominations && !dark_gate_timing &&
        !full_mode_baseline && !dual_pressure && !archive_pressure && !flag_reply &&
        !formation && !route_choice && !natural_routes && !team_relay && !collision_semantics && !visible_terminal &&
        !relay_setup && !reply_budget && !relay_chain && !counterreply && !reply_examples && !reply_judge &&
        !reply_window_games && !frozen_reply && !scheduled_reply && !ordinary_classroom && !evolution_games &&
        !engineer_formation && !quiet_options && !reply_value_audit && !lesson_review && !preference_games && !scoring_audit &&
        !leaf_teacher && !diversity_panel && !product_shadow) return 2;
    if (product_shadow) {
      run_product_shadow(argv[1]);
      return 0;
    }
    Rules rules;
    std::string error;
    if (!parse_rules(read_text(argv[1]),rules,error) || !rules.eliminate_exhausted_movers)
      throw std::runtime_error("invalid frozen selfplay rules");
    if (active) std::cerr<<"JUNQI_ACTIVE_PATH "<<mode<<' '<<rules_hash(rules)<<'\n';
    if (mode=="repeat" && rules.reset_on_capture_only) throw std::runtime_error("audit requires collision resets");
    fixtures(rules);
    if (flag_reply) {
      run_flag_reply_pressure(argv[1],rules,mode);
      return 0;
    }
    if (reply_examples) {
      run_reply_window_examples(rules);
      return 0;
    }
    if (reply_window_games) {
      run_window_games(rules);
      return 0;
    }
    if (frozen_reply) {
      run_frozen_reply_audit(argv[1],rules);
      return 0;
    }
    if (scheduled_reply) {
      run_scheduled_reply_games(rules);
      return 0;
    }
    if (ordinary_classroom) {
      run_ordinary_classroom(rules);
      return 0;
    }
    if (evolution_games) {
      run_evolution_games(rules,evolution_trace);
      return 0;
    }
    if (preference_games) {
      run_preference_games(rules);
      return 0;
    }
    if (engineer_formation) {
      run_engineer_formation(rules,engineer_trace);
      return 0;
    }
    if (quiet_options) {
      run_quiet_options(rules);
      return 0;
    }
    if (reply_value_audit || lesson_review) {
      run_visited_reply_value(rules,lesson_review);
      return 0;
    }
    if (leaf_teacher) {
      run_leaf_teacher_data(rules,leaf_recheck);
      return 0;
    }
    if (scoring_audit) {
      run_scoring_audit(rules);
      return 0;
    }
    if (diversity_panel) {
      run_reply_diversity_panel(rules);
      return 0;
    }
    if (reply_judge) {
      const bool window=mode=="reply-window-judge-window";
      StudentNetwork model;
      int count;
      if (!model.load(std::cin,true,window) || !(std::cin>>count) || count<1 || count>8192) return 2;
      run_scored_window(count,model,window?reply_window_inputs:exchange_inputs);
      return 0;
    }
    if (dual_pressure) {
      run_dual_pressure(rules);
      return 0;
    }
    if (archive_pressure) {
      run_archive_pressure(argv[1],rules);
      return 0;
    }
    if (route_choice) {
      if (mode=="route-panels") run_route_panels(rules);
      else if (mode=="route-resources") run_route_resources(rules);
      else run_route_replays(argv[1],rules);
      return 0;
    }
    if (natural_routes) {
      run_natural_routes(argv[1],rules);
      return 0;
    }
    if (team_relay) {
      run_team_relay(rules);
      return 0;
    }
    if (visible_terminal) {
      uint64_t seed;
      single_case_check(bool(std::cin>>seed),"missing visible terminal seed");
      run_visible_terminal(rules,seed);
      return 0;
    }
    if (relay_setup) {
      run_relay_setup(argv[1],rules);
      return 0;
    }
    if (collision_semantics) {
      run_collision_semantics(rules);
      return 0;
    }
    if (formation) {
      if (mode=="formation-rules") {
        formation_rule_cases(rules);
        formation_training_seeds(50000,512);
      } else {
        uint64_t seed;
        int count;
        single_case_check(bool(std::cin>>seed>>count) && count>0 && count<=64 && seed<=UINT64_MAX-count,
                          "invalid formation seed block");
        run_formation_matrix(rules,seed,count,mode=="formation-collision-impact");
      }
      return 0;
    }
    if (full_mode_baseline) {
      run_full_mode_baseline(argv[1],rules);
      return 0;
    }
    if (dark_info_audit) {
      run_dark_info_audit(argv[1]);
      return 0;
    }
    if (dark_bomb_belief) {
      run_dark_bomb_belief(argv[1]);
      return 0;
    }
    if (dark_search_budget) {
      run_dark_search_budget(argv[1]);
      return 0;
    }
    if (dark_formation_library) {
      run_dark_formation_library(argv[1],mode=="dark-formation-sources");
      return 0;
    }
    if (dark_reply_leaf) {
      run_dark_reply_leaf(argv[1],mode=="dark-reply-course");
      return 0;
    }
    if (dark_opening_reply) {
      run_dark_opening_reply(argv[1],mode=="dark-opening-reply-window");
      return 0;
    }
    if (dark_witness) {
      run_dark_witness_audit(argv[1]);
      return 0;
    }
    if (dark_course_tape) {
      if (dark_course_openings) run_dark_course_openings(rules);
      else run_dark_course_tape(rules);
      return 0;
    }
    if (dark_ordinary) {
      run_dark_ordinary_learning(argv[1]);
      return 0;
    }
    if (dark_first_loss) {
      run_dark_first_loss(argv[1]);
      return 0;
    }
    if (dark_partner_support) {
      run_dark_partner_support(argv[1]);
      return 0;
    }
    if (dark_replanning) {
      run_dark_replanning(argv[1],mode=="dark-replanning-matched"?ReplanMode::Matched:
                                  mode=="dark-replanning-course"?ReplanMode::Course:ReplanMode::Factorial);
      return 0;
    }
    if (dark_response_distribution) {
      run_dark_response_distribution(argv[1]);
      return 0;
    }
    if (dark_seat_candidates) {
      run_dark_seat_candidates(argv[1]);
      return 0;
    }
    if (dark_stratified) {
      run_dark_stratified(argv[1]);
      return 0;
    }
    if (dark_heritage_nominations) {
      run_dark_heritage_nominations(rules,action_level_nomination,force_heritage_contact);
      return 0;
    }
    if (dark_partner_evidence) {
      run_dark_partner_evidence(rules);
      return 0;
    }
    if (dark_broad_pairs) {
      run_dark_broad_pairs(argv[1]);
      return 0;
    }
    if (dark_gate_timing) {
      run_dark_gate_timing(argv[1]);
      return 0;
    }
    if (directed_coverage) {
      run_directed_coverage(argv[1]);
      return 0;
    }
    if (capture_coverage) {
      run_capture_coverage(argv[1]);
      return 0;
    }
    if (mode=="capture-replies" || student_capture_replies) {
      run_capture_replies(student_capture_replies);
      return 0;
    }
    if (visible_learning) {
      StudentNetwork student;
      if (!student.load(std::cin)) return 2;
      audit_visible_learning(rules,student);
      return 0;
    }
    if (evaluator_bounds) {
      prove_evaluator_controls(rules);
      return 0;
    }
    if (evaluator_panel || reply_budget || relay_chain || counterreply) {
      ShortWeights peer{};
      for (auto& value:peer) if (!(std::cin>>value) || !std::isfinite(value)) return 2;
      if (reply_budget) run_reply_budget(argv[1],rules,peer);
      else if (relay_chain) run_relay_chain(argv[1],rules,peer);
      else if (counterreply) run_counterreply_audit(argv[1],rules,peer);
      else compare_evaluator_panel(rules,peer,gate_trace);
      return 0;
    }
    if (mode=="gate") gate_fixtures(rules);
    if (has_forecast) {
      if (rules.reset_on_capture_only || rules.marshal_reveals_flag) throw std::runtime_error("unsupported public clock or reveal rule");
      response_fixtures(rules);
    }
    if (bomb_test) bomb_fixtures(rules);
    if (joint_routes) two_target_fixtures(rules);
    if (book_cover) book_cover_fixtures(rules);
    else if (book_test) book_preparation_fixtures(rules);
    if (conditional_reply) {
      conditional_public_test(rules);
      conditional_hidden_test(rules);
      return 0;
    }
    if (student_features_only) {
      if (exchange) exchange_feature_fixtures(rules);
      export_student_features(std::cin,exchange?&rules:nullptr);
      return 0;
    }
    uint64_t start,count;
    ShortWeights parent{},peer{};
    if (!(std::cin>>start>>count) || !count || count>64 || start>UINT64_MAX-count) return 2;
    for (auto target : {&parent,&peer})
      for (size_t i=0;i<5;++i) if (!(std::cin>>(*target)[i]) || !std::isfinite((*target)[i])) return 2;
    if (first_exit_audit) {
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int partner:{1,3}) for (int table:{0,1}) audit_first_exit(seed,seat,partner,table,rules,parent,peer);
      return 0;
    }
    if (first_exit_pair) {
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int partner:{1,3}) for (int table:{0,1})
          paired_first_exit(seed,seat,partner,table,rules,parent,peer,mode=="first-exit-approach-control");
      return 0;
    }
    if (conversion_cross) {
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int partner:{1,3}) for (int table:{0,1}) for (bool early:{false,true})
          cross_conversion(seed,ExchangePolicy{rules,parent,peer,seat,partner,table},early);
      return 0;
    }
    if (conversion_history) {
      const auto operation=mode=="export-conversion-history"?export_conversion_history:restore_conversion_history;
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int partner:{1,3}) for (int table:{0,1}) for (bool early:{false,true})
          operation(seed,ExchangePolicy{rules,parent,peer,seat,partner,table},early);
      if (mode=="restore-conversion-history") {
        std::string extra;
        single_case_check(!(std::cin>>extra),"unexpected extra conversion journal data");
      }
      return 0;
    }
    if (blind_extension) {
      blind_extension_controls(rules);
      std::cout<<std::setprecision(17);
      for (uint64_t seed=start;seed<start+count;++seed) for (int offset:{0,1})
        for (int table:{0,1}) audit_blind_extension(seed,ExchangePolicy{rules,parent,peer,int((seed+offset)%4),1,table});
      return 0;
    }
    if (replay_phase) {
      StudentNetwork student;
      if (!student.load(std::cin)) return 2;
      run_replay_phases(argv[1],rules,parent,peer,student,mode=="replay-phase-witness");
      return 0;
    }
    if (conversion_labels) {
      StudentNetwork student;
      if (!student.load(std::cin)) return 2;
      std::cout<<std::setprecision(17);
      for (uint64_t seed=start;seed<start+count;++seed) {
        if (mode=="conversion-label-pilot") {
          for (bool early:{false,true}) audit_conversion_label(seed,ExchangePolicy{rules,parent,peer,3,3,1},early,student);
        } else if (mode=="conversion-label-locked" || mode=="student-label-locked") {
          audit_conversion_label(seed,ExchangePolicy{rules,parent,peer,int(seed%4),3,0},false,student,true,student_labels);
        } else {
          for (int table:{0,1}) audit_conversion_label(seed,ExchangePolicy{rules,parent,peer,int(seed%4),3,table},false,student,confirm_labels,student_labels);
        }
      }
      return 0;
    }
    if (discovery) {
      std::cout<<std::setprecision(17);
      const auto probe=calibration?DiscoveryProbe::Calibration:
                       reply_counterfactual?DiscoveryProbe::Counterfactual:heldout_reply?DiscoveryProbe::Heldout:
                       critical_reply?DiscoveryProbe::Replan:DiscoveryProbe::Original;
      if (calibration) evaluator_controls(rules,parent);
      for (uint64_t seed=start;seed<start+count;++seed) discover_public_actions(seed,rules,parent,peer,probe);
      return 0;
    }
    if (post_training || post_evaluation) {
      StudentNetwork original,updated;
      if (!original.load(std::cin) || (post_evaluation && !updated.load(std::cin))) return 2;
      for (uint64_t seed=start;seed<start+count;++seed)
        for (int partner:{1,post_training?2:3}) for (int table:{0,1}) {
          const int learner=(int(seed%4)+2*table+partner-1)%kSeats;
          post_retirement(seed,{rules,parent,peer,learner,partner,table,&original},
                          post_evaluation?&updated:nullptr);
        }
      return 0;
    }
    if (winner_takeover || winner_delay) {
      StudentNetwork student;
      if (!student.load(std::cin)) return 2;
      Rules historical;
      if (!parse_rules(archive_review_rules(read_text("config/ruleset.json"),false),historical,error)) return 2;
      historical.truncation_cap=rules.truncation_cap;
      std::string key,tape;
      int group,reps;
      while (std::cin>>key>>group>>reps>>tape) {
        const ExchangePolicy frozen{historical,parent,peer,kNone,1,0,&student};
        if (winner_delay) {
          if (group<0 || group>3 || reps<0 || reps>3) return 2;
          delay_archive(key,ArchiveRows(decode_archive_tape(tape)),group,reps,frozen);
        } else {
          if (group<0 || group>1 || reps<2 || reps>4) return 2;
          takeover_archive(key,ArchiveRows(decode_archive_tape(tape)),group,reps,frozen,winner_confirm);
        }
      }
      return 0;
    }
    if (follow_natural) {
      StudentNetwork student;
      if (!student.load(std::cin)) return 2;
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int partner:{1,2}) for (int table:{0,1})
          natural_exchange(seed,{rules,parent,peer,seat,partner,table,&student,recovery_only});
      return 0;
    }
    if (follow_archive) {
      std::string key,tape;
      Rules historical;
      if (!parse_rules(archive_review_rules(read_text("config/ruleset.json"),false),historical,error)) return 2;
      while (std::cin>>key>>tape) archived_exchange(key,ArchiveRows(decode_archive_tape(tape)),historical,parent,peer);
      return 0;
    }
    if (student_eval || student_control) {
      if (rules.no_progress_limit!=70 || rules.reset_on_capture_only || rules.marshal_reveals_flag) return 2;
      StudentNetwork student;
      if (student_eval && !student.load(std::cin,exchange)) return 2;
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int partner:{1,2}) for (int table:{0,1})
          student_trial(seed,seat,partner,table,rules,parent,peer,student_eval?&student:nullptr,student_contact);
      return 0;
    }
    if (student_visits || student_frozen) {
      StudentNetwork student;
      uint64_t reply_salt=0;
      if (reseeded_visit && (!(std::cin>>reply_salt) || !reply_salt)) return 2;
      if (student_visits && !student.load(std::cin)) return 2;
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int table:{0,1}) student_feedback_trial(seed,seat,table,rules,parent,peer,
                                                    student_visits?&student:nullptr,reply_salt);
      return 0;
    }
    if (early_labels) {
      for (uint64_t seed=start;seed<start+count;++seed) for (int seat=0;seat<kSeats;++seat)
        for (int table:{0,1}) early_learning_trial(seed,seat,table,rules,parent,peer,early_heuristic?2:1);
      return 0;
    }
    if (clock_test) {
      if (rules.no_progress_limit!=70 || rules.reset_on_capture_only || rules.draw_before_win)
        throw std::runtime_error("clock experiment requires the frozen draw contract");
      clock_learning_fixtures(rules);
      double coefficient=0;
      if (clock_fit && (!(std::cin>>coefficient) || !std::isfinite(coefficient))) return 2;
      for (uint64_t seed=start;seed<start+count;++seed) for (int table:{0,1})
        for (int seat=0;seat<kSeats;++seat)
          clock_trial(seed,seat,table,rules,parent,peer,
                      clock_fit?std::optional<double>(coefficient):std::nullopt,contact_control);
      return 0;
    }
    if (public_reply) {
      public_reply_fixtures(rules,parent);
      return 0;
    }
    const auto arms=counterplay_test?std::vector<int>{17,22}:challenger_race?std::vector<int>{0,2,15,17}:
                    mode=="repeat"?std::vector<int>{5}:
                    mode=="gate"?std::vector<int>{7,8}:
                    mode=="response"?std::vector<int>{9,10}:
                    mode=="first-exit"?std::vector<int>{11,12}:
                    mode=="combined"?std::vector<int>{11,12,13}:
                    export_learning?std::vector<int>{13}:
                    hard_learning?std::vector<int>{14,21}:
                    bomb_test?std::vector<int>{15,16}:
                    book_test?std::vector<int>{17,book_cover?20:19}:
                    staged_routes?std::vector<int>{15,17,18}:
                    joint_routes?std::vector<int>{15,17}:
                    mixed?std::vector<int>{11,13,15}:
                    tail_audit?std::vector<int>{11,13,14,15}:
                    mode=="conversion"?std::vector<int>{2,4}:std::vector<int>{0,1,2,3};
    const auto partners=(challenger_race || counterplay_test)?std::vector<int>{1,2}:hard_learning?std::vector<int>{1}:
                        (bomb_test || joint_routes || book_test)?std::vector<int>{0,1}:
                        mixed?std::vector<int>{0,1,2}:std::vector<int>{-1};
    const auto tables=counterplay_test?std::vector<int>{2}:challenger_race?std::vector<int>{0,1,2}:std::vector<int>{0,1};
    for (int partner : partners) for (int table : tables) for (int arm : arms)
      for (uint64_t seed=start;seed<start+count;++seed) for (int learner=0;learner<kSeats;++learner)
        trial(seed,learner,table,arm,rules,parent,peer,tail_audit,partner,
              export_learning || (hard_learning && arm==14),bomb_test,book_test,hard_learning || table==2);
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
