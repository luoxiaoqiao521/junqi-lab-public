#include "../sim/archive_review.h"
#include "../sim/public_reply_history.h"
#include "single_rules_cases.h"
#include "deadline_scenarios.h"
#include "replay_qh_wire.h"
#include <set>
using namespace jq;

#include "hypothesis_stock_intervals.h"
#include "hypothesis_capture_ledger.h"

static unsigned division_collision(bool our_attack,int opposing,bool revealed,bool moved,
                                   Rules rules=Rules{}) {
  const auto& board=Board::get();
  const Move hit{uint8_t(board.seat_node(1,4,1)),uint8_t(board.seat_node(1,4,2))};
  auto state=single_case_state();
  state.to_move=our_attack?0:1;
  state.flag_revealed[1]=revealed;
  place_single_case_piece(state,1,our_attack?hit.from:hit.to,Division);
  const auto dark=our_attack?hit.to:hit.from;
  place_single_case_piece(state,26,dark,opposing);
  const auto before=observe(state,0);
  OpeningMarks marks(before,false);
  if (moved) marks.ranks[dark]&=~((1u<<Flag)|(1u<<Mine));
  const auto event=recorded_move(state,hit,rules);
  const auto after=observe(state,0);
  advance_contact_ranks(marks,before,after,hit,event.outcome,rules);
  if (event.outcome==Outcome::BothRemoved) {
    single_case_check(marks.removed[0].size==1 && marks.removed[1].size==1,
                      "mutual loss did not consume each seat's own stock");
    return marks.removed[1].masks[0];
  }
  single_case_check(marks.removed[0].confirmed[Division]==1 && marks.removed[1].size==0,
                    "a survivor was consumed or a known division loss was forgotten");
  return compatible_rank_mask(after,marks,hit.to);
}

static std::array<unsigned,2> partner_casualties(int partner,int enemy) {
  const auto& board=Board::get();
  const Move hit{uint8_t(board.seat_node(1,4,1)),uint8_t(board.seat_node(1,4,2))};
  auto state=single_case_state();
  state.out[2]=false;
  state.to_move=2;
  place_single_case_piece(state,51,hit.from,partner);
  place_single_case_piece(state,26,hit.to,enemy);
  const auto before=observe(state,0);
  OpeningMarks marks(before,false);
  const Rules rules;
  const auto event=recorded_move(state,hit,rules);
  single_case_check(event.outcome==Outcome::BothRemoved,"partner casualty fixture must have a mutual loss");
  advance_contact_ranks(marks,before,observe(state,0),hit,event.outcome,rules);
  single_case_check(marks.removed[2].size==1 && marks.removed[1].size==1 && !marks.removed[0].size,
                    "partner fight was not charged to the two involved seats independently");
  return {marks.removed[2].masks[0],marks.removed[1].masks[0]};
}

static void inventory_fixture() {
  const unsigned leaders=(1u<<Marshal)|(1u<<Corps), mutual=(1u<<Bomb)|(1u<<Division);
  single_case_check(division_collision(false,Corps,false,false)==leaders &&
                    division_collision(false,Corps,true,false)==(1u<<Corps),
                    "opposing attack that removes a division has the wrong surviving ranks");
  single_case_check(division_collision(true,Corps,true,false)==((1u<<Mine)|(1u<<Corps)) &&
                    division_collision(true,Corps,true,true)==(1u<<Corps),
                    "attacking a possible mine was treated as a known corps");
  single_case_check(division_collision(true,Bomb,true,false)==mutual &&
                    division_collision(false,Division,true,false)==mutual,
                    "a mutual loss was assigned a unique opposing rank");
  Rules alternate;
  alternate.mine_survives=false;
  single_case_check(division_collision(true,Mine,true,false,alternate)==(mutual|(1u<<Mine)),
                    "alternate mine rule did not affect mutual-loss evidence");
  const auto partner=partner_casualties(Division,Division);
  single_case_check(partner==partner_casualties(Battalion,Bomb) && (partner[0]&(1u<<Battalion)) &&
                    (partner[1]&mutual)==mutual,"partner fight guessed private ranks");

  auto state=single_case_state();
  place_single_case_piece(state,1,4,Engineer);
  place_single_case_piece(state,26,30,Bomb);
  const auto view=observe(state,0);
  OpeningMarks marks(view,false);
  marks.ranks[30]=leaders|(1u<<Bomb);
  marks.removed[1].add(leaders);
  marks.removed[1].add(leaders);
  const auto unresolved=marks;
  constrain_rank_inventory(view,marks,1);
  single_case_check(marks.ranks[30]==(1u<<Bomb) && marks.removed[1].size==2 &&
                    marks.removed[1].confirmed[Marshal]==0 && marks.removed[1].confirmed[Corps]==0,
                    "ambiguous casualties did not jointly reserve their available ranks");
  for (uint64_t seed=0;seed<32;++seed) {
    const auto world=public_sample(view,unresolved,seed);
    single_case_check(world && world->rank_at[30]==Bomb,"sampler reused slots reserved by two deaths");
  }
  OpeningMarks same_rank(view,false);
  same_rank.ranks[30]=1u<<Engineer;
  same_rank.removed[1].add(1u<<Engineer);
  constrain_rank_inventory(view,same_rank,1);
  single_case_check(same_rank.ranks[30]==(1u<<Engineer) && !(same_rank.removed[1].exhausted&(1u<<Engineer)),
                    "one engineer lost incorrectly exhausted the rank");
}

static void rank_matching_fixture() {
  Rng rng(379);
  for (int attempt=0;attempt<64;++attempt) {
    RankAssignment inventory;
    const size_t pieces=1+rng.below(kPiecesPerSeat);
    for (size_t i=0;i<pieces;++i)
      inventory.allowed[i]=rng.below(all_evidence_ranks+1) | (1u<<inventory.slot_rank[i]);
    single_case_check(inventory.match(pieces),"fixture inventory must have a matching");
    const auto support=inventory.supported_ranks();
    for (size_t piece=0;piece<pieces;++piece) for (int rank=Flag;rank<RankCount;++rank) {
      if (!(inventory.allowed[piece]&(1u<<rank))) continue;
      auto forced=inventory;
      forced.allowed[piece]=1u<<rank;
      single_case_check(bool(support[piece]&(1u<<rank))==forced.match(pieces),
                        "inventory support disagreed with a forced-rank rematch");
    }
  }
}

static bool rejects_history(const PublicReplyHistory& history,const Observation& view,const OpeningMarks& actor) {
  try {
    history.for_actor(view,actor);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

static void history_fixture() {
  const auto& board=Board::get();
  const Move turn{uint8_t(board.seat_node(1,0,0)),uint8_t(board.seat_node(1,4,4))};
  auto state=single_case_state();
  state.to_move=1;
  state.ply=state.no_progress=0;
  place_single_case_piece(state,1,board.seat_node(0,0,2),Engineer);
  place_single_case_piece(state,26,turn.from,Engineer);
  const int dark=board.seat_node(1,4,1);
  place_single_case_piece(state,27,dark,Platoon);
  PublicReplyHistory history(state,false);
  const auto before=state;
  single_case_check(is_legal(state,turn),"engineer history fixture is not legal");
  const Rules rules;
  const auto event=step(state,turn,rules);
  history.advance(before,state,event,rules);
  const auto view=seat_decision(state).observation;
  OpeningMarks private_evidence(view,false);
  private_evidence.ranks[dark]=1u<<Platoon;
  const auto merged=history.for_actor(view,private_evidence);
  single_case_check(history.marks().moved[turn.to] && history.marks().ranks[turn.to]==(1u<<Engineer)
                    && history.last_moves()[1]==turn && merged.moved[turn.to],
                    "reply forgot a public rail turn or a prior move");
  single_case_check(merged.ranks[dark]==(1u<<Platoon) &&
                    (history.marks().ranks[dark]&(1u<<Mine)),
                    "an actor's private inference reached the shared history");
  auto stale=view;
  ++stale.quiet_clock;
  single_case_check(rejects_history(history,stale,private_evidence),"stale reply history admitted");
  auto unfiltered=view;
  unfiltered.ranks[dark]=Platoon;
  single_case_check(rejects_history(history,unfiltered,private_evidence),"foreign private rank admitted");

  auto controlled=state;
  StepInfo exit{};
  recorded_retirement(controlled,1,exit,ExitReason::NoLegalMove);
  history.after_control(controlled);
  single_case_check(history.marks().ranks[turn.to]==0 && !history.marks().moved[turn.to] &&
                    history.last_moves()[1]==turn,"control event erased history or left a ghost piece");
}

static void reveal_fixture() {
  auto root=single_case_state();
  place_single_case_piece(root,1,51,Engineer);
  place_single_case_piece(root,26,30,Platoon);
  const auto dark=seat_decision(root);
  single_case_check(dark.observation.revealed_flags==0 && dark.observation.rank_at(56)==Unknown,
                    "an owner's private flag must not imply public revelation");
  root.flag_revealed[0]=root.flag_revealed[1]=root.flag_revealed[2]=true;
  const auto light=seat_decision(root);
  OpeningMarks marks(light.observation,false);
  marks.ranks[30]=(1u<<Platoon)|(1u<<Regiment)|(1u<<Marshal);
  for (uint64_t seed=0;seed<64;++seed) {
    const auto world=public_sample(light.observation,marks,seed);
    single_case_check(world.has_value(),"public fixture could not reconstruct");
    const auto view=seat_decision(*world);
    single_case_check(view.observation.revealed_flags==7 && view.observation.rank_at(56)==Flag &&
                      view.observation.rank_at(30)==Unknown && view.observation.quiet_clock==69 &&
                      view.actions==light.actions,"hypothesis did not preserve the public root");
    single_case_check(world->rank_at[30]!=Marshal && world->out[2] &&
                      observe(*world,1).rank_at(26)==Flag,"hypothesis lost another seat's public evidence");
  }
  marks.ranks[30]=1u<<Marshal;
  single_case_check(!public_sample(light.observation,marks,0),"publicly removed marshal was resurrected");
  auto inconsistent=light.observation;
  inconsistent.revealed_flags=0;
  single_case_check(!public_sample(inconsistent,OpeningMarks(inconsistent,false),0),
                    "inconsistent public-flag projection was accepted");
}

struct Coverage {
  unsigned plies=0, roots=0, flags=0, foreign_flags=0, retired_flags=0, clocks=0;
  unsigned sampled_roots=0, worlds=0, rejected=0, old_hidden_flags=0;
  unsigned moved_roots=0, engineer_roots=0, reply_last_roots=0;
  unsigned stock_roots=0;
};

static void verify_sample_inventory(const State& world,const OpeningMarks& evidence) {
  int live[kSeats][RankCount]{};
  for (int id=0;id<kPieces;++id) if (world.alive[id]) ++live[id/kPiecesPerSeat][world.rank_of[id]];
  for (int seat=0;seat<kSeats;++seat) if (!world.out[seat])
    for (int rank=Flag;rank<RankCount;++rank)
      expect_value(live[seat][rank]+evidence.removed[seat].confirmed[rank]<=kRankCount[rank],
                   "archive hypothesis resurrected a confirmed casualty");
}

static void check_root(const State& actual,const OpeningMarks& marks,const PublicReplyHistory& history,
                       unsigned flags,Coverage& coverage) {
  const auto d=seat_decision(actual);
  const auto& view=d.observation;
  const auto merged=history.for_actor(view,marks);
  bool stock=false;
  for (int seat=0;seat<kSeats;++seat)
    stock|=!actual.out[seat] && seat!=view.viewer && (merged.removed[seat].exhausted&~(1u<<Marshal));
  coverage.stock_roots+=stock;
  ++coverage.roots;
  coverage.flags+=flags!=0;
  coverage.clocks+=actual.no_progress!=0;
  bool foreign=false, retired=false, moved=false, engineer=false, reply_last=false;
  for (int seat=0;seat<kSeats;++seat) if (flags&(1u<<seat)) {
    retired|=actual.out[seat];
    foreign|=!actual.out[seat] && seat!=view.viewer;
  }
  for (int seat=0;seat<kSeats;++seat)
    if (!actual.out[seat] && seat!=view.viewer) {
      const auto previous=history.last_moves()[seat];
      reply_last|=previous.from!=previous.to;
      for (int node=0;node<kNodes;++node)
        if (actual.seat_at[node]>=0 && actual.seat_at[node]!=seat && history.marks().moved[node]) {
          moved=true;
          engineer|=history.marks().ranks[node]==(1u<<Engineer);
        }
    }
  coverage.moved_roots+=moved;
  coverage.engineer_roots+=engineer;
  coverage.reply_last_roots+=reply_last;
  coverage.foreign_flags+=foreign;
  coverage.retired_flags+=retired;
  expect_value(view.revealed_flags==flags,"public mask disagrees with archive reveal events");
  const auto before=coverage.worlds;
  for (uint64_t trial=0;trial<8;++trial) {
    auto world=public_sample(view,merged,mix_seed(static_cast<uint64_t>(view.ply),trial));
    if (!world) {++coverage.rejected; continue;}
    ++coverage.worlds;
    verify_sample_inventory(*world,merged);
    std::vector<Move> legal;
    legal_moves(*world,legal);
    expect_value(legal==d.actions && same_observation(view,observe(*world,view.viewer)),
                 "archive hypothesis changed public decision or legal moves");
    for (int seat=0;seat<kSeats;++seat)
      expect_value(world->out[seat]==actual.out[seat] && observe(*world,seat).revealed_flags==flags,
                   "archive hypothesis changed public retirement or revealed status");
    for (int id=0;id<kPieces;++id)
      expect_value(!world->alive[id] || world->rank_of[id]!=Marshal ||
                   !(flags&(1u<<(id/kPiecesPerSeat))),"archive hypothesis resurrected a marshal");
    for (int node=0;node<kNodes;++node)
      if (world->seat_at[node]>=0)
        expect_value(history.marks().ranks[node]&(1u<<world->rank_at[node]),
                     "archive hypothesis contradicted the public movement or collision history");
    std::fill(std::begin(world->flag_revealed),std::end(world->flag_revealed),false);
    const auto legacy=observe(*world,view.viewer);
    const bool hides_flag=!std::equal(view.ranks,view.ranks+kNodes,legacy.ranks);
    expect_value(hides_flag==foreign,"legacy projection control did not match the visible flag");
    coverage.old_hidden_flags+=hides_flag;
  }
  coverage.sampled_roots+=coverage.worlds!=before;
}

static Coverage verify_roots(const ArchiveRows& raw,const Rules& rules) {
  ArchiveReview review;
  init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
  std::vector<OpeningMarks> marks;
  PublicReplyHistory history(review.state,true);
  std::array<Move,kSeats> tape_last{};
  for (int seat=0;seat<kSeats;++seat) marks.emplace_back(observe(review.state,seat),true);
  Coverage coverage;
  unsigned flags=0;
  bool changed=false;
  for (int i=0;i<raw.entries;++i) {
    const auto row=raw.row(i);
    if (raw.byte(row)==245) {
      review.control(raw,row,i+1,rules);
      history.after_control(review.state);
      changed|=review.last.eliminated_mask!=0;
      continue;
    }
    const State before=review.state;
    if (before.ply%16==0 || changed) check_root(before,marks[before.to_move],history,flags,coverage);
    review.play(raw,row,i+1,rules);
    history.advance(before,review.state,review.last,rules);
    tape_last[before.to_move]={uint8_t(raw.point(row+2)),uint8_t(raw.point(row+4))};
    expect_value(history.last_moves()==tape_last,"recorded action disappeared from reply history");
    review.frames.str("");
    flags|=review.last.revealed_mask;
    changed=review.last.revealed_mask!=0;
    for (int seat=0;seat<kSeats;++seat)
      if (!review.state.out[seat])
        advance_contact_ranks(marks[seat],observe(before,seat),observe(review.state,seat),
                              review.last.move,review.last.outcome,rules);
  }
  coverage.plies=static_cast<unsigned>(review.state.ply);
  return coverage;
}

static std::optional<Rules> strict_rules(const ArchiveRows& raw,const std::string& config,
                                         const std::string& visibility) {
  std::optional<Rules> result;
  for (bool ccw:{false,true}) {
    try {
      archive_review_attempt(raw,config,visibility,ccw);
    } catch (const std::runtime_error&) {
      continue;
    }
    if (result) return std::nullopt;
    Rules rules;
    std::string error;
    if (!parse_rules(archive_review_rules(config,ccw),rules,error))
      throw std::runtime_error(error);
    result=rules;
  }
  return result;
}

int main(int argc,char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--capture-course") {
      capture_course();
      return 0;
    }
    if (argc==2 && std::string(argv[1])=="--capture-ledger") {
      capture_fixture();
      capture_attribution_fixture();
      Rules rules;
      std::string error, hex;
      expect_value(parse_rules(archive_review_rules(read_text("config/ruleset-selfplay.json"),false),rules,error),
                   "bad capture archive profile");
      int ccw=0;
      while (std::cin>>ccw>>hex) {
        rules.ccw=bool(ccw);
        capture_archive(ArchiveRows(decode_archive_tape(hex)),rules);
      }
      return 0;
    }
    if (argc!=1) return 2;
    reveal_fixture();
    history_fixture();
    inventory_fixture();
    capture_attribution_fixture();
    rank_matching_fixture();
    stock_interval_fixture();
    const auto config=read_text("config/ruleset.json"), visibility=read_text("config/observation.json");
    std::set<std::string> tapes;
    std::string tape;
    while (std::cin>>tape) {
      std::optional<ArchiveRows> rows;
      try {
        rows.emplace(decode_archive_tape(tape));
      } catch (const std::runtime_error&) {
        std::cout<<"unsupported_format\n";
        continue;
      }
      if (!tapes.insert(rows->tape()).second) {std::cout<<"duplicate\n"; continue;}
      const auto rules=strict_rules(*rows,config,visibility);
      if (!rules) {std::cout<<"strict_replay_rejected\n"; continue;}
      const auto c=verify_roots(*rows,*rules);
      std::cout<<"verified "<<c.plies<<' '<<c.roots<<' '<<c.flags<<' '<<c.foreign_flags<<' '
               <<c.retired_flags<<' '<<c.clocks<<' '<<c.sampled_roots<<' '<<c.worlds<<' '
               <<c.rejected<<' '<<c.old_hidden_flags<<' '<<c.moved_roots<<' '<<c.engineer_roots<<' '
               <<c.reply_last_roots<<' '<<c.stock_roots<<'\n';
    }
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
