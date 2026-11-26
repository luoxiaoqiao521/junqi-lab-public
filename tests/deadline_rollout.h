#pragma once

#include "../sim/public_reply_history.h"
#include "deadline_scenarios.h"
#include "deadline_two_targets.h"
#include "deadline_book_slot.h"
#include "deadline_book_cover.h"

struct ScoredMove { jq::Move move; int score=-10000; };

static std::vector<jq::Move> response_candidates(const jq::SeatDecision& d,
    const jq::OpeningMarks& marks,jq::Move baseline,const jq::Rules& rules,uint64_t seed,
    std::optional<jq::Move>* inserted=nullptr,bool cover_slot=false) {
  using namespace jq;
  const auto& view=d.observation;
  const auto targets=flag_targets(view,marks);
  std::vector<Move> moves{baseline};
  if (targets.empty()) return moves;
  const auto& board=Board::get();
  const std::optional<BookPreparation> preparation=inserted && !cover_slot?
      std::optional<BookPreparation>(std::in_place,d,marks,targets):std::nullopt;
  const std::optional<BookCover> cover=cover_slot?
      std::optional<BookCover>(std::in_place,view,marks,rules,targets):std::nullopt;
  std::vector<ScoredMove> prepared;
  ScoredMove contact{}, advance{}, defence{}, follow{};
  const auto rank_best=[](ScoredMove& slot,Move move,int score) { if (score>slot.score) slot={move,score}; };
  const int flag=own_flag(view);
  for (const auto action:d.actions) {
    if (action==baseline) continue;
    const int left=flag_distance(action.to,targets), gain=flag_distance(action.from,targets)-left;
    if (view.seat_at(action.to)>=0) {
      const unsigned ranks=compatible_rank_mask(view,marks,action.to);
      const auto outcome=collision_outcome_mask(view.rank_at(action.from),ranks,rules);
      if (outcome && outcome!=outcome_bit(Outcome::DefenderWins))
        rank_best(contact,action,10*((ranks&(1u<<Flag))!=0)+4*gain-left);
    } else if (board.node(action.to).kind!=NodeKind::HQ) {
      rank_best(advance,action,4*gain-left);
      if (flag>=0) rank_best(defence,action,3*(distance(action.from,flag)-distance(action.to,flag))-distance(action.to,flag));
      const auto after=project_quiet(view,action);
      std::vector<Move> next;
      gen_piece_moves(after,view.viewer,action.to,next);
      if (preparation) {
        const auto value=preparation->score(view,marks,rules,action,next);
        if (value) prepared.push_back({action,value});
      }
      if (cover) {
        const auto value=cover->score(after,marks);
        if (value) prepared.push_back({action,value});
      }
      int reach=left;
      for (const auto route:next) reach=std::min(reach,flag_distance(route.to,targets));
      rank_best(follow,action,-8*reach-left);
    }
  }
  for (const auto slot:{contact,advance,defence,follow})
    if (slot.score>-10000 && std::find(moves.begin(),moves.end(),slot.move)==moves.end()) moves.push_back(slot.move);
  const auto extra=d.actions[Rng(seed).below(static_cast<uint32_t>(d.actions.size()))];
  if (std::find(moves.begin(),moves.end(),extra)==moves.end()) {
    ScoredMove best{};
    for (const auto item:prepared)
      if (!(item.move==extra) && std::find(moves.begin(),moves.end(),item.move)==moves.end())
        rank_best(best,item.move,item.score);
    if (inserted && best.score>0) { *inserted=best.move; moves.push_back(best.move); }
    else moves.push_back(extra);
  }
  return moves;
}

static jq::OpeningMarks imagined_marks(const jq::Observation& view) {
  using namespace jq;
  OpeningMarks marks(view,false);
  for (int n=0;n<kNodes;++n) if (view.seat_at(n)>=0) {
    const auto& square=Board::get().node(n);
    if (square.kind!=NodeKind::HQ || square.seat!=view.seat_at(n)) marks.ranks[n]&=~(1u<<Flag);
  }
  return marks;
}

struct ResponseValue {
  int wins=0, losses=0, draws=0, unknown=0;
  double value=0;
};

struct ResponseAudit {
  int depth, plies=0, extended=0;
  bool conditional=false, all_identities=false;
  unsigned cutoff_reasons=0;
  int extension_at=9, gate_visits=0, early_settled=0, early_unresolved=0, extension_steps=0;
  std::array<int,8> gate_reasons{};
};

static unsigned response_extension_reasons(const jq::SeatDecision& d,const jq::OpeningMarks& marks,
                                           const jq::Rules& rules,bool contact) {
  if (d.actions.empty()) return 0;
  return unsigned(contact) | (public_flag_finish(d,d.actions.front(),rules)?2u:0u) |
         (visible_flag_threat(d.observation,marks,rules).forced?4u:0u);
}

struct LeafValueTerms {
  double movable, distance, adjacency;
  double total() const { return movable+distance+adjacency; }
};

static LeafValueTerms leaf_terms(const jq::State& sample,int team) {
  using namespace jq;
  int flags[kSeats]{kNone,kNone,kNone,kNone};
  double material=0;
  for (int n=0;n<kNodes;++n) if (sample.seat_at[n]>=0) {
    const int seat=sample.seat_at[n], rank=sample.rank_at[n];
    if (rank==Flag) flags[seat]=n;
    else if (can_leave_node(rank,n)) material+=(team==team_of(seat)?1:-1)*(rank==Engineer?3:2);
  }
  int nearest[2]{20,20};
  int threats[2]{};
  const auto& board=Board::get();
  for (int seat=0;seat<kSeats;++seat) if (flags[seat]>=0) {
    const int foe=1-team_of(seat);
    for (int n=0;n<kNodes;++n) if (sample.seat_at[n]>=0 && team_of(sample.seat_at[n])==foe &&
                                   can_leave_node(sample.rank_at[n],n)) {
      nearest[foe]=std::min(nearest[foe],distance(n,flags[seat]));
      const auto& roads=board.road(flags[seat]);
      if (std::find(roads.begin(),roads.end(),n)!=roads.end()) ++threats[foe];
    }
  }
  return {.008*material, .055*(nearest[1-team]-nearest[team]),
          .20*(std::min(2,threats[team])-std::min(2,threats[1-team]))};
}

static double leaf_value(const jq::State& sample,int team) {
  return leaf_terms(sample,team).total();
}

static void record_leaf(ResponseValue& score,const jq::State& sample,int viewer,int root_ply,bool first_exit,
                        const jq::Rules& rules,RoutePriority routes) {
  using namespace jq;
  const int result=review_outcome(sample.result,viewer);
  if (result==2) {
    ++score.wins;
    score.value+=5+(sample.ply<=100?1.5:sample.ply<=140?.8:0.)-.025*(sample.ply-root_ply);
  } else if (result==0) {
    ++score.losses;
    score.value-=6;
  } else if (result==1) ++score.draws;
  else {
    ++score.unknown;
    if (sample.result==Result::Ongoing) {
      score.value+=leaf_value(sample,team_of(viewer));
      if (routes!=RoutePriority::Old) score.value+=readiness_term(sample,rules,team_of(viewer),first_exit,routes);
      if (first_exit) for (int seat=0;seat<kSeats;++seat)
        if (sample.out[seat]) score.value+=team_of(seat)==team_of(viewer)?-.80:.80;
    }
  }
}

struct ResponseBranch {
  jq::State sample;
  int viewer, root_ply;
  std::array<jq::OpeningMarks,jq::kSeats> marks;
  std::array<jq::Move,jq::kSeats> last;
  std::array<jq::Rng,jq::kSeats> streams;
  bool collision=false;

  ResponseBranch(const jq::State& root,const jq::OpeningMarks& evidence,
                 const jq::PublicReplyHistory& prefix,uint64_t seed)
      : sample(root),viewer(root.to_move),root_ply(root.ply),
        marks{prefix.marks(),prefix.marks(),prefix.marks(),prefix.marks()},
        last(prefix.last_moves()),streams(review_streams(seed)) {
    marks[viewer]=evidence;
  }

  void play(jq::Move move,const jq::Rules& rules,bool active_counterplay) {
    using namespace jq;
    const int actor=sample.to_move;
    const auto before=sample;
    const auto event=step(sample,move,rules);
    collision=event.outcome!=Outcome::Move;
    last[actor]=move;
    for (int seat=0;seat<kSeats;++seat)
      if (!sample.out[seat] && (active_counterplay || team_of(seat)==team_of(viewer)))
        advance_contact_ranks(marks[seat],observe(before,seat),observe(sample,seat),move,event.outcome,rules);
  }

  jq::Move reply(const jq::SeatDecision& d,const jq::Rules& rules,const jq::ShortWeights& history,
                 int context,bool active_counterplay) {
    using namespace jq;
    const int seat=sample.to_move;
    const auto fallback=context==0?d.actions[streams[seat].below(static_cast<uint32_t>(d.actions.size()))]:
                                         short_move(d.observation,d.actions,history,streams[seat]);
    return active_counterplay || team_of(seat)==team_of(viewer)?plan(d,marks[seat],fallback,last[seat],rules):fallback;
  }
};

static void continue_response(ResponseBranch& branch,const jq::Rules& rules,const jq::ShortWeights& history,
                              int context,int depth,bool active_counterplay,ResponseAudit* audit) {
  using namespace jq;
  auto& sample=branch.sample;
  const int root_ply=branch.root_ply, viewer=branch.viewer;
  const int limit=root_ply+depth, start=sample.ply;
  while (sample.result==Result::Ongoing && sample.ply<limit) {
    auto d=seat_decision(sample);
    const int seat=sample.to_move;
    if (audit && audit->all_identities) d.observation=observe(sample,seat,IdentityVisibility::AllIdentities);
    if (audit && audit->conditional && sample.ply==root_ply+audit->extension_at) {
      const auto reasons=response_extension_reasons(d,branch.marks[seat],rules,branch.collision);
      ++audit->gate_visits;
      ++audit->gate_reasons[reasons];
      if (!reasons) break;
      audit->cutoff_reasons|=reasons;
      ++audit->extended;
    }
    branch.play(branch.reply(d,rules,history,context,active_counterplay),rules,active_counterplay);
  }
  if (audit) {
    const int played=sample.ply-root_ply;
    audit->plies+=sample.ply-start;
    if (audit->conditional) {
      if (played<audit->extension_at || (played==audit->extension_at && sample.result!=Result::Ongoing)) {
        if (team_score(sample.result,viewer,sample.match)) ++audit->early_settled;
        else ++audit->early_unresolved;
      }
      audit->extension_steps+=std::max(0,played-audit->extension_at);
    }
  }
}

static void response_trial(jq::State sample,jq::Move first,const jq::OpeningMarks& evidence,
    const jq::PublicReplyHistory& prefix,
    const jq::Rules& rules,const jq::ShortWeights& history,int context,uint64_t seed,ResponseValue& score,
    bool first_exit,RoutePriority routes=RoutePriority::Old,bool active_counterplay=false,ResponseAudit* audit=nullptr) {
  ResponseBranch branch(sample,evidence,prefix,seed);
  branch.play(first,rules,active_counterplay);
  if (audit) ++audit->plies;
  continue_response(branch,rules,history,context,audit?audit->depth:9,active_counterplay,audit);
  record_leaf(score,branch.sample,branch.viewer,branch.root_ply,first_exit,rules,routes);
}

struct ResponseWork {
  int evaluations=0, changed=0, early_changes=0, absent=0, samples=0, terminal_wins=0, terminal_losses=0;
  int book_offered=0, book_selected=0, book_applied=0, first_book=0;
  bool chose_book=false;
};

static uint64_t public_rollout_seed(const jq::Observation& view,const jq::OpeningMarks& marks,int clock) {
  auto hash=jq::fnv1a(view.seats,sizeof(view.seats));
  hash=jq::fnv1a(view.ranks,sizeof(view.ranks),hash);
  hash=jq::fnv1a(marks.ranks.data(),sizeof(marks.ranks),hash);
  return jq::mix_seed(hash,uint64_t(view.ply*4+view.viewer)*3001u+uint64_t(clock));
}

static jq::Move response_choice(const jq::SeatDecision& d,const jq::OpeningMarks& actor,
    const jq::PublicReplyHistory& prefix,
    jq::Move baseline,const jq::Rules& rules,const jq::ShortWeights& history,int clock,
    ResponseWork& work,bool first_exit=false,RoutePriority routes=RoutePriority::Old,
    bool book_slot=false,bool cover_slot=false,bool active_counterplay=false) {
  using namespace jq;
  ++work.evaluations;
  const uint64_t seed=public_rollout_seed(d.observation,actor,clock);
  const auto evidence=prefix.for_actor(d.observation,actor);
  std::optional<Move> inserted;
  work.chose_book=false;
  const auto actions=response_candidates(d,evidence,baseline,rules,seed,
                                        book_slot || cover_slot?&inserted:nullptr,cover_slot);
  work.book_offered+=inserted.has_value();
  if (actions.size()<2) {++work.absent; return baseline;}
  std::vector<State> worlds;
  const int ply=d.observation.ply;
  for (uint64_t i=0;i<12 && worlds.size()<4;++i) {
    const auto sample=public_sample(d.observation,evidence,mix_seed(seed,i));
    if (sample) worlds.push_back(*sample);
  }
  if (worlds.size()!=4) {++work.absent; return baseline;}
  std::vector<ResponseValue> values(actions.size());
  for (size_t a=0;a<actions.size();++a) for (size_t i=0;i<worlds.size();++i)
    response_trial(worlds[i],actions[a],evidence,prefix,rules,history,int(i%2),mix_seed(seed,i+71),
                   values[a],first_exit,routes,active_counterplay);
  size_t best=0;
  const double margin=(ply<100?.24:ply<140?.34:.48);
  for (size_t a=1;a<values.size();++a) {
    const auto& v=values[a];
    if (v.losses>values[0].losses || v.wins<values[0].wins || v.value<=values[0].value+margin) continue;
    if (v.value>values[best].value) best=a;
  }
  work.samples+=static_cast<int>(actions.size()*worlds.size());
  work.terminal_wins+=values[best].wins;
  work.terminal_losses+=values[best].losses;
  work.changed+=best!=0;
  work.early_changes+=best!=0 && ply<140;
  work.chose_book=inserted && actions[best]==*inserted;
  work.book_selected+=work.chose_book;
  return actions[best];
}

static void response_fixtures(const jq::Rules& rules) {
  using namespace jq;
  auto s=selfplay_fixture();
  place_single_case_piece(s,1,4,Marshal);
  place_single_case_piece(s,26,30,Platoon);
  const auto d=seat_decision(s);
  OpeningMarks marks(d.observation,false);
  const PublicReplyHistory prefix(s,false);
  marks.ranks[30]=(1u<<Platoon)|(1u<<Engineer);
  marks.ranks[56]=1u<<Flag;
  const auto world=public_sample(d.observation,marks,712);
  single_case_check(world && seat_decision(*world).actions==d.actions,"hypothetical legal moves diverged");
  single_case_check(world->rank_at[30]==Platoon || world->rank_at[30]==Engineer,"sample ignored hidden rank domain");
  auto hidden=s;
  hidden.rank_at[30]=hidden.rank_of[26]=Engineer;
  const auto alternate=seat_decision(hidden);
  const Move base{4,3};
  ResponseWork a,b;
  const auto x=response_choice(d,marks,prefix,base,rules,ShortWeights{},0,a);
  const auto y=response_choice(alternate,marks,prefix,base,rules,ShortWeights{},0,b);
  single_case_check(x==y && a.samples==b.samples && a.samples>0,"hidden answer changed rollout selection");
  ResponseWork hostile_a,hostile_b;
  const auto h1=response_choice(d,marks,prefix,base,rules,ShortWeights{},0,hostile_a,false,
                                RoutePriority::Joint,false,false,true);
  const auto h2=response_choice(alternate,marks,prefix,base,rules,ShortWeights{},0,hostile_b,false,
                                RoutePriority::Joint,false,false,true);
  single_case_check(h1==h2 && hostile_a.samples==hostile_b.samples && hostile_a.samples==a.samples,
                    "hostile replies changed the sample budget or used hidden truth");
  auto impossible=marks;
  impossible.ranks[56]=1u<<Mine;
  single_case_check(!public_sample(d.observation,impossible,712),"missing hidden flag admitted a scenario");
  const auto fallback=response_choice(d,impossible,prefix,base,rules,ShortWeights{},0,a);
  single_case_check(fallback==base && a.absent==1,"unsupported scenarios did not fall back");
}
