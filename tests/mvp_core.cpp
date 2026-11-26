#include "../sim/replay.h"
#include "../strategy/short_policy.h"
#include "../sim/opening_train.h"
#include "../sim/seat_episode.h"
#include "../sim/mine_evidence.h"
#include "../core/layout.h"
#include "../strategy/heuristic.h"
#include "mode_observation_cases.h"
#include "single_rules_cases.h"
#include "single_match_cases.h"
#include <chrono>
#include <cstring>
#include <iostream>
using namespace jq;
void check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
void invariant(const State& s) {
  int counts[kPieces] = {};
  for (int n = 0; n < kNodes; ++n) {
    int id = s.id_at[n];
    if (id < 0) { check(s.seat_at[n] == -1 && s.rank_at[n] == Unknown, "empty mismatch"); continue; }
    check(id < kPieces && s.alive[id] && s.node_of[id] == n, "piece location/alive");
    check(s.seat_at[n] == id / 25 && s.rank_at[n] == s.rank_of[id], "piece attributes");
    check(!s.out[id / 25], "eliminated piece remains"); ++counts[id];
  }
  for (int i = 0; i < kPieces; ++i) check(counts[i] == int(s.alive[i]), "piece ledger");
}
uint64_t game(int seed, int& result, int& plies) {
  Rng rng(seed); int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& l : layouts) generate_layout(l, rng);
  State s; Rules rules; init_state(s, layouts, int(rng.below(4)), rules);
  while (s.result == Result::Ongoing) {
    invariant(s); auto o = observe(s, s.to_move); State hidden = s;
    for (int n = 0; n < kNodes; ++n) if (hidden.seat_at[n] >= 0 && hidden.seat_at[n] != s.to_move) {
      int id = hidden.id_at[n]; hidden.rank_at[n] = Engineer; hidden.rank_of[id] = Engineer;
    }
    auto o2 = observe(hidden, hidden.to_move);
    check(std::memcmp(o.seats, o2.seats, sizeof o.seats) == 0 && std::memcmp(o.ranks, o2.ranks, sizeof o.ranks) == 0, "observation leakage");
    Move a, b; Rng copy = rng;
    check(random_move(o, rng, a) && random_move(o2, copy, b) && a == b, "policy leakage");
    Rng h1(123), h2(123); Move ha,hb;
    check(heuristic_move(o,h1,ha) && heuristic_move(o2,h2,hb) && ha==hb,"heuristic leakage");
    check(is_legal(s,ha),"heuristic legality");
    std::vector<Move> candidates; gen_moves(o,o.viewer,candidates);
    for(auto candidate:candidates) check(score_move(o,ha).total() >= score_move(o,candidate).total(),"heuristic maximum");
    check(is_legal(s, a), "illegal generated move");
    State before = s, undo; StepInfo info;
    check(!apply_checked(s, Move{255, 255}, rules, undo, info), "invalid accepted");
    check(std::memcmp(&s, &before, sizeof s) == 0, "invalid side effect");
    check(apply_checked(s, a, rules, undo, info), "valid rejected");
    State after = s; restore(s, undo); check(std::memcmp(&s, &before, sizeof s) == 0, "undo");
    step(s, a, rules); check(std::memcmp(&s, &after, sizeof s) == 0, "redo");
  }
  invariant(s); result = int(s.result); plies = s.ply; return state_hash(s);
}
void config_tests() {
  std::string text = read_text("../config/ruleset.json"), err; Rules r;
  check(parse_rules(text, r, err), "valid config");
  auto original = rules_hash(r); auto p = text.find("70");
  for (const auto& bad : {"\"70\"", "70x", "999999999999999999", "-1"}) {
    auto mutated = text; mutated.replace(p, 2, bad);
    check(!parse_rules(mutated, r, err) && rules_hash(r) == original, "bad config published");
  }
  for (const auto& bad : {"{", "[]x", "{\"a\":1,\"a\":2}", "{\"a\":01}", "{\"a\":}", "[1,]"}) {
    bool rejected = false; try { parse_json(bad); } catch (...) { rejected = true; }
    check(rejected, "bad JSON accepted");
  }
}
Observation blank() {
  Observation o;
  std::fill(std::begin(o.seats), std::end(o.seats), int8_t(-1));
  std::fill(std::begin(o.ranks), std::end(o.ranks), int8_t(Unknown)); return o;
}
void movement() {
  const auto& board = Board::get(); auto n = [&](int r, int c){return board.seat_node(0,r,c);};
  auto has = [](const Observation& o, int from, int to) {
    std::vector<Move> ms; gen_moves(o, 0, ms);
    return std::find(ms.begin(), ms.end(), Move{uint8_t(from),uint8_t(to)}) != ms.end();
  };
  auto o = blank(); int from = n(0,0); o.seats[from]=0; o.ranks[from]=Marshal;
  check(has(o,from,n(0,4)), "straight rail");
  check(!has(o,from,n(4,4)), "ordinary right turn allowed");
  o.ranks[from]=Engineer; check(has(o,from,n(4,4)), "engineer turn missing");
  o.ranks[from]=Marshal; o.seats[n(0,2)]=2;
  check(!has(o,from,n(0,2)) && !has(o,from,n(0,4)), "ally blocked rail");
  o=blank(); from=n(1,0); o.seats[from]=0; o.ranks[from]=Marshal;
  check(has(o,from,n(1,1)), "empty camp entry"); o.seats[n(1,1)]=1;
  check(!has(o,from,n(1,1)), "camp attack");
  o=blank(); from=n(5,1); o.seats[from]=0; o.ranks[from]=Engineer;
  std::vector<Move> ms; gen_moves(o,0,ms); check(ms.empty(), "HQ exit");
  // A prescribed corner arc remains a single ordinary rail line.
  o=blank(); from=n(3,4); int across=board.seat_node(1,3,0);
  o.seats[from]=0; o.ranks[from]=Marshal;
  check(has(o,from,across), "ordinary piece can follow the corner arc");
  int block=board.seat_node(1,1,0); o.seats[block]=1;
  check(has(o,from,block) && !has(o,from,across), "enemy stops the ordinary arc");
  o.seats[block]=2;
  check(!has(o,from,block) && !has(o,from,across), "partner blocks the ordinary arc");

  o=blank(); from=n(0,0); o.seats[from]=0; o.ranks[from]=Engineer;
  o.seats[n(0,2)]=1; o.seats[n(0,4)]=2;
  check(has(o,from,n(0,2)) && !has(o,from,n(0,3)),
        "engineer may take a rail blocker but cannot pass it");
  o=blank(); from=n(2,1); o.seats[from]=0; o.ranks[from]=Engineer;
  check(has(o,from,n(2,2)) && !has(o,from,n(2,3)), "engineer road move is one step");
}
void combat() {
  Rules r;
  for (int a=Engineer;a<RankCount;++a) for (int d=Engineer;d<RankCount;++d)
    check(resolve(a,d,r)==(a==d?Outcome::BothRemoved:a>d?Outcome::AttackerWins:Outcome::DefenderWins),"rank ordering");
  check(resolve(Engineer,Mine,r)==Outcome::AttackerWins,"engineer mine");
  check(resolve(Marshal,Mine,r)==Outcome::DefenderWins,"marshal mine");
  check(known_collision_objective(Marshal,Mine,r)==-60,"heuristic misreads a visible mine");
  check(resolve(Bomb,Mine,r)==Outcome::BothRemoved,"bomb mine");
  check(resolve(Bomb,Flag,r)==Outcome::BothRemoved,"bomb flag");
  // For an active platoon, a public double removal has two explanations.
  for (int d=Flag; d<RankCount; ++d)
    check((resolve(Platoon,d,r)==Outcome::BothRemoved)==(d==Platoon || d==Bomb),
          "platoon double-removal candidates under default combat rules");
  r.bomb_captures_flag=true; check(resolve(Bomb,Flag,r)==Outcome::AttackerWins,"bomb capture flag");
  r.bomb_mine_both_removed=false; check(resolve(Bomb,Mine,r)==Outcome::DefenderWins,"bomb mine option");
  check(known_collision_objective(Bomb,Mine,r)==-60,"heuristic ignored configured bomb/mine behavior");
  r.mine_survives=false; check(resolve(Marshal,Mine,r)==Outcome::BothRemoved,"mine option");
  check(resolve(Platoon,Mine,r)==Outcome::BothRemoved,"alternate mine rule adds a candidate");
}

void public_evidence() {
  const auto& board = Board::get();
  auto n = [&](int seat, int row, int col) { return uint8_t(board.seat_node(seat,row,col)); };
  auto visible = blank();
  const Move arc{n(1,3,4), n(2,3,0)};
  visible.seats[arc.from]=1;
  const auto ordinary_arc = public_route_ranks(visible, arc);
  check(ordinary_arc == (all_evidence_ranks & ~((1u << Flag) | (1u << Mine))),
        "arc allows all mobile ranks");
  visible.seats[n(2,1,0)]=0;
  check(!(public_route_ranks(visible, arc) & (1u << Marshal)), "blocked ordinary arc cannot explain move");

  const Move turn{n(1,0,0), n(1,4,4)};
  visible=blank(); visible.seats[turn.from]=1;
  check(public_route_ranks(visible, turn)==(1u << Engineer), "turn proves engineer");
  check(public_route_ranks(blank(), turn)==all_evidence_ranks, "missing source gives no evidence");

  State a{};
  a.seat_at[arc.from]=0; a.rank_at[arc.from]=Platoon;
  a.seat_at[arc.to]=1; a.rank_at[arc.to]=Platoon;
  State b=a; b.rank_at[arc.to]=Bomb;
  const Rules rules;
  for (const auto& world : {a,b}) {
    const auto view=observe(world,0);
    check(view.rank_at(arc.from)==Platoon && view.rank_at(arc.to)==Unknown, "identical visible collision");
    const auto result=resolve(world.rank_at[arc.from],world.rank_at[arc.to],rules);
    const auto ranks=public_defender_ranks(view.rank_at(arc.from),result,rules);
    check(ranks == ((1u << Platoon) | (1u << Bomb)) &&
          (ranks & (1u << world.rank_at[arc.to])), "both compatible truths survive identical evidence");
  }
  Rules alternate=rules; alternate.mine_survives=false;
  check(public_defender_ranks(Platoon,Outcome::BothRemoved,alternate) ==
        ((1u << Platoon) | (1u << Bomb) | (1u << Mine)), "mine variant changes mask");
  check(public_defender_ranks(Unknown,Outcome::BothRemoved,rules)==all_evidence_ranks,
        "unseen attacker gives no one-sided collision evidence");
}
void opening_marks() {
  const auto& board = Board::get();
  auto n = [&](int row, int col) { return uint8_t(board.seat_node(1,row,col)); };
  const auto front=n(0,0), rear=n(1,0), spare=n(0,1), hq=n(5,1);
  const unsigned bomb=1u << Bomb, mine=1u << Mine, flag=1u << Flag;
  auto seen=blank();
  for (int node : {front,rear,hq}) seen.seats[node]=1;
  OpeningMarks marks(seen,true);
  check(!(marks.ranks[front] & (bomb|mine|flag)), "opening front exclusions");
  const Rules rules;
  check(collision_outcome_mask(Marshal,marks.ranks[front],rules) ==
        collision_outcome_mask(Marshal,marks.ranks[front] | bomb,rules),
        "same rank preserves the double-removal outcome");
  check(collision_outcome_mask(Marshal,1u << Engineer,rules) !=
        collision_outcome_mask(Marshal,(1u << Engineer) | bomb,rules),
        "additional evidence can make bomb exclusion change immediate outcomes");
  check((marks.ranks[rear] & (bomb|mine|flag))==bomb, "opening second row excludes mine and flag");
  check((marks.ranks[hq] & (bomb|mine|flag))==(bomb|mine|flag), "HQ keeps flag possibility");
  check(OpeningMarks(seen,false).ranks[front]==all_evidence_ranks, "missing opening stays unknown");
  seen.ply=1;
  check(OpeningMarks(seen,true).ranks[front]==all_evidence_ranks, "midgame is not deployment");
  seen.seats[front]=-1; seen.seats[spare]=1;
  marks.advance({front,spare},Outcome::Move,seen);
  seen.seats[rear]=-1; seen.seats[front]=1;
  marks.advance({rear,front},Outcome::Move,seen);
  check((marks.ranks[front] & bomb) && !(marks.ranks[spare] & bomb), "opening marks follow piece not square");
  seen.seats[spare]=-1;
  marks.advance({spare,front},Outcome::DefenderWins,seen);
  check(marks.ranks[front] & bomb, "defender retains its opening evidence");
  seen.seats[front]=seen.seats[hq]=-1;
  seen.seats[spare]=0;
  seen.seats[spare]=-1;
  marks.advance({spare,front},Outcome::BothRemoved,seen);
  check(!marks.ranks[hq] && !marks.ranks[front], "double removal and clearing sweep discard marks");
}
void survivor_evidence() {
  const auto& board = Board::get();
  auto n = [&](int row, int col) { return uint8_t(board.seat_node(1,row,col)); };
  const Move clash{n(4,4),n(4,0)}, shift{clash.to,n(4,1)};
  auto before=blank();
  before.seats[clash.from]=0; before.ranks[clash.from]=Platoon;
  before.seats[clash.to]=1;
  const auto engineer=n(3,0);
  before.seats[engineer]=0; before.ranks[engineer]=Engineer;
  std::vector<Move> legal;
  gen_piece_moves(before,0,clash.from,legal);
  check(std::find(legal.begin(),legal.end(),clash)!=legal.end(),"legal surviving-defense fixture");
  OpeningMarks marks(before,true);
  auto held=before;
  held.seats[clash.from]=-1; held.ranks[clash.from]=Unknown; held.turn=1;
  Rules rules;
  SurvivorEvidence evidence;
  auto moved=held;
  moved.seats[shift.from]=-1; moved.seats[shift.to]=1; moved.turn=0;
  const unsigned route=public_route_ranks(held,shift);
  for (int truth : {Company, Battalion}) {
    SurvivorEvidence branch;
    branch.advance(before,held,clash,resolve(Platoon,truth,rules),marks,all_evidence_ranks,rules);
    check(branch.collision[clash.to] & (1u << Mine),"surviving defender might be a mine");
    branch.advance(held,moved,shift,Outcome::Move,marks,route,rules);
    check(branch.after_move[shift.to] & (1u << truth),"compatible survivor remains possible");
    if (evidence.after_move[shift.to])
      check(branch.after_move==evidence.after_move,"same history yields same survivor candidates");
    evidence=branch;
  }
  check(evidence.encounters==1 && evidence.first_moves==1,"survivor counts each stage once");
  check(survivor_changes_attack(moved,evidence,rules),"survivor movement removes possible engineer win");
  check(!survivor_changes_attack(moved,SurvivorEvidence{},rules),"missing encounter leaves no combined evidence");
  auto replacement=moved;
  replacement.seats[shift.to]=0;
  evidence.advance(moved,replacement,{engineer,shift.to},Outcome::AttackerWins,marks,all_evidence_ranks,rules);
  check(!evidence.collision[shift.to] && !evidence.after_move[shift.to],"replacement loses defender evidence");
  rules.mine_survives=false;
  evidence.advance(before,held,clash,Outcome::DefenderWins,marks,all_evidence_ranks,rules);
  check(!evidence.collision[clash.to],"alternate mine rule has no mine hypothesis");
}
void mine_mark_regression() {
  const auto& board=Board::get();
  const auto from=uint8_t(board.seat_node(1,4,0)), to=uint8_t(board.seat_node(1,4,1));
  auto before=blank(), stationary=blank();
  before.seats[from]=stationary.seats[to]=1;
  OpeningMarks moved(before,true);
  const OpeningMarks stayed(stationary,true);
  constexpr auto mine=1u << Mine;
  check((moved.ranks[from] & mine) && (stayed.ranks[to] & mine),
        "unmoved rear positions preserve the mine candidate");
  moved.advance({from,to},Outcome::Move,stationary);
  check(!moved.moved[from] && moved.moved[to] && !(moved.ranks[to] & mine),
        "publicly moved piece in the same rear position cannot be a mine");
}
void mine_formation_contract() {
  const auto& board=Board::get();
  int8_t layouts[kSeats][kPiecesPerSeat];
  Rng rng(241);
  for (auto& layout : layouts) generate_layout(layout,rng);
  auto& layout=layouts[1];
  const int flag_slot=int(std::find(std::begin(layout),std::end(layout),Flag)-std::begin(layout));
  const auto flag=layout_slot_node(1,flag_slot);
  const auto& roads=board.road(flag);
  const auto adjacent=[&](int slot) {
    const auto node=layout_slot_node(1,slot);
    return std::find(roads.begin(),roads.end(),node)!=roads.end();
  };
  int adjacent_slot=kNone, outside=kNone;
  for (int slot=0; slot<kPiecesPerSeat; ++slot) {
    if (!adjacent(slot)) continue;
    adjacent_slot=slot;
    if (layout[slot]==Mine) continue;
    for (int other=0; other<kPiecesPerSeat; ++other)
      if (layout[other]==Mine && !adjacent(other)) { std::swap(layout[slot],layout[other]); break; }
  }
  for (int slot=0; slot<kPiecesPerSeat; ++slot)
    if (!adjacent(slot) && layout[slot]!=Flag && deployment_rank_allowed(layout_slot_node(1,slot),Mine))
      outside=slot;
  check(outside!=kNone && adjacent_slot!=kNone && valid_layout(layout),"valid covered flag fixture");
  const Rules rules;
  State covered, open;
  init_state(covered,layouts,0,rules);
  std::swap(layout[outside],layout[adjacent_slot]);
  check(valid_layout(layout),"valid one-open-route fixture");
  init_state(open,layouts,0,rules);
  const auto a=seat_decision(covered), b=seat_decision(open);
  check(opening_mine_neighbors(covered)[1]==3 && opening_mine_neighbors(open)[1]==2 &&
        same_seat_view(covered,open,0) && a.actions==b.actions,
        "closed and open mine variants are legal and indistinguishable at opening");
  std::cout<<"PASS closed mine pattern and open-route counterexample\n";
}
void evidence_policy_contract() {
  const auto& board=Board::get();
  auto n=[&](int r,int c) { return uint8_t(board.seat_node(1,r,c)); };
  const Move turn{n(0,0),n(4,4)};
  auto before=blank();
  before.seats[turn.from]=1;
  const auto own=n(4,3), other=n(4,2), start=n(5,4);
  before.seats[own]=before.seats[start]=0;
  before.ranks[own]=before.ranks[start]=Marshal;
  before.seats[other]=1;
  OpeningMarks evidence(before,true);
  auto after=before;
  after.seats[turn.from]=-1; after.seats[turn.to]=1;
  advance_public_ranks(evidence,before,after,turn,Outcome::Move);
  const Move marked{start,turn.to};
  const Move comparison{own,other};
  std::vector<Move> legal; gen_moves(after,0,legal);
  for (const auto action : {marked,comparison})
    check(std::find(legal.begin(),legal.end(),action)!=legal.end(),"policy fixture move is legal");
  check(short_features(after,marked)==short_features(after,comparison),"legacy policy aliases the attacks");
  check(short_features(after,marked,false,&evidence.ranks)[5]==1 &&
        short_features(after,comparison,false,&evidence.ranks)[5]==0,"public history separates the attacks");
  ShortWeights weights{}; weights[5]=20;
  const std::vector<Move> actions{comparison,marked};
  int old_choices=0, new_choices=0;
  for (int seed=0;seed<32;++seed) {
    Rng old_rng(seed), new_rng(seed);
    old_choices += short_move(after,actions,weights,old_rng)==marked;
    new_choices += short_move(after,actions,weights,new_rng,nullptr,false,&evidence.ranks)==marked;
  }
  check(old_choices<32 && new_choices==32,"public history reaches real policy sampler");
}
void termination() {
  Rng rng(42); int8_t layouts[4][25]; for(auto& l:layouts)generate_layout(l,rng);
  Rules r; State s; init_state(s,layouts,0,r); std::vector<Move> ms; legal_moves(s,ms);
  check(!ms.empty(),"initial moves"); Move m=ms[0]; check(s.id_at[m.to]<0,"quiet fixture");
  State before=s; r.no_progress_limit=1; step(s,m,r); check(s.result==Result::Draw,"draw threshold");
  s=before;r.no_progress_limit=0;r.truncation_cap=1;step(s,m,r);check(s.result==Result::Truncated,"cap not draw");
  s=before;r.truncation_cap=3000;r.no_progress_limit=70;s.no_progress=68;step(s,m,r);
  check(s.result==Result::Ongoing&&s.no_progress==69,"below draw threshold");
}
void reveals_and_elimination() {
  State s{};
  std::fill(std::begin(s.id_at),std::end(s.id_at),int8_t(-1));
  std::fill(std::begin(s.seat_at),std::end(s.seat_at),int8_t(-1));
  std::fill(std::begin(s.rank_at),std::end(s.rank_at),int8_t(Unknown));
  auto put=[&](int id,int n,int rank){s.id_at[n]=int8_t(id);s.seat_at[n]=int8_t(id/25);s.rank_at[n]=int8_t(rank);s.rank_of[id]=int8_t(rank);s.node_of[id]=uint8_t(n);s.alive[id]=true;};
  const auto& b=Board::get();int from=b.seat_node(0,0,0),to=b.seat_node(0,0,1),flag=b.seat_node(1,5,1);
  put(0,from,Bomb);put(25,to,Marshal);put(26,flag,Flag);put(27,b.seat_node(1,0,0),Engineer);
  Rules r;r.marshal_reveals_flag=true;s.no_progress=69;
  auto info=step(s,Move{uint8_t(from),uint8_t(to)},r);
  check(info.revealed_mask==2&&s.flag_revealed[1]&&s.no_progress==0,"reveal/collision reset");
  check(observe(s,0).ranks[flag]==Flag,"public flag not shown");
  put(1,from,Engineer);s.to_move=0;s.result=Result::Ongoing;
  // Adjacent sparse fixture puts enemy flag on a reachable station for combat consequences only.
  s.alive[26]=false;s.id_at[flag]=-1;s.seat_at[flag]=-1;s.rank_at[flag]=Unknown;
  put(26,to,Flag);info=step(s,Move{uint8_t(from),uint8_t(to)},r);
  check(s.out[1]&&!s.alive[27]&&(info.eliminated_mask&2),"flag elimination cleanup");invariant(s);
}
void seat_episode_contract() {
  State s{};
  std::fill(std::begin(s.id_at),std::end(s.id_at),int8_t(-1));
  std::fill(std::begin(s.seat_at),std::end(s.seat_at),int8_t(-1));
  std::fill(std::begin(s.rank_at),std::end(s.rank_at),int8_t(Unknown));
  const auto& b = Board::get();
  auto put = [&](int id, int n, int rank) {
    s.id_at[n]=int8_t(id); s.seat_at[n]=int8_t(id/25); s.rank_at[n]=int8_t(rank);
    s.rank_of[id]=int8_t(rank); s.node_of[id]=uint8_t(n); s.alive[id]=true;
  };
  const int flag0=b.seat_node(0,5,1), flag1=b.seat_node(1,5,1);
  put(0,flag0,Flag); put(25,flag1,Flag);
  put(26,b.seat_node(0,4,1),Engineer);
  put(50,b.seat_node(1,4,1),Engineer);
  put(51,b.seat_node(2,5,1),Flag);
  s.out[3]=true;
  s.to_move=1;
  Rules rules;
  int calls=0;
  const auto learner = advance_to_seat(s,rules,0,[&](const Observation& o, const std::vector<Move>& actions) {
    const int target = calls++ == 0 ? flag0 : flag1;
    check(o.viewer == calls && o.rank_at(target) == Unknown, "response view leaked");
    const auto m = std::find_if(actions.begin(), actions.end(), [&](Move move){return move.to == target;});
    check(m != actions.end(), "fixture capture unavailable");
    return *m;
  });
  check(!learner && calls==2 && s.out[0] && s.result==Result::Team0Wins,
        "learner elimination stopped episode");
  check(team_score(s.result,0)==1 && team_score(s.result,1)==0, "team reward direction");
  check(team_score(Result::Draw,0)==0.5 && !team_score(Result::Truncated,0) &&
        !team_score(Result::Ongoing,0), "draw, truncation and ongoing conflated");
  Rng rng(42);
  int8_t layouts[4][25];
  for (auto& layout : layouts) generate_layout(layout,rng);
  init_state(s,layouts,0,rules);
  auto& enemy=layouts[1];
  auto other=std::find(enemy+1,enemy+5,enemy[0]==Engineer ? Marshal : Engineer);
  if (other==enemy+5) other=std::find_if(enemy+1,enemy+5,[&](int rank){return rank!=enemy[0];});
  check(other!=enemy+5,"hidden-layout fixture lacks distinct ranks");
  std::iter_swap(enemy,other);
  State hidden;
  init_state(hidden,layouts,0,rules);
  auto never=[](const Observation&,const std::vector<Move>&)->Move{throw std::runtime_error("learner turn skipped");};
  const auto a=advance_to_seat(s,rules,0,never), d=advance_to_seat(hidden,rules,0,never);
  check(a && d && state_hash(s)!=state_hash(hidden) && a->actions==d->actions &&
        std::memcmp(a->observation.seats,d->observation.seats,sizeof a->observation.seats)==0 &&
        std::memcmp(a->observation.ranks,d->observation.ranks,sizeof a->observation.ranks)==0,
        "hidden layout changed learner decision");
  std::cout<<"PASS single-seat contract: elimination, team score, truncation, hidden-layout invariance\n";
}
void partner_corridor_contract() {
  State initial{};
  std::fill(std::begin(initial.id_at),std::end(initial.id_at),int8_t(-1));
  std::fill(std::begin(initial.seat_at),std::end(initial.seat_at),int8_t(-1));
  std::fill(std::begin(initial.rank_at),std::end(initial.rank_at),int8_t(Unknown));
  const auto& board=Board::get();
  auto n=[&](int seat,int row,int col){return uint8_t(board.seat_node(seat,row,col));};
  auto put=[&](int id,uint8_t node,int8_t rank){
    initial.id_at[node]=int8_t(id); initial.seat_at[node]=int8_t(id/25); initial.rank_at[node]=rank;
    initial.rank_of[id]=rank; initial.node_of[id]=node; initial.alive[id]=true;
  };
  const auto blocker=n(0,0,2), ally=n(0,0,0), foe=n(0,0,4);
  for (int seat=0; seat<kSeats; ++seat) put(seat*25,n(seat,5,1),Flag);
  put(1,blocker,Platoon); put(26,foe,Marshal); put(51,ally,Brigade);
  put(27,n(1,4,2),Company); put(76,n(3,0,0),Company);
  check(observe(initial,0).rank_at(ally)==Unknown && observe(initial,0).rank_at(foe)==Unknown,
        "corridor fixture keeps partner and enemy hidden from acting seat");
  const Move clear{blocker,n(0,1,1)}, stay{blocker,n(0,0,3)};
  const Move raid{foe,ally}, wait{n(1,4,2),n(1,3,1)}, reinforce{ally,foe};
  const Rules rules;
  State open=initial, blocked=initial, undo;
  StepInfo info;
  check(apply_checked(open,clear,rules,undo,info) && open.to_move==1,"clearing move and intervening enemy turn");
  check(apply_checked(blocked,stay,rules,undo,info) && blocked.to_move==1,"blocking move and intervening enemy turn");
  check(is_legal(open,raid) && !is_legal(blocked,raid),"clearing also gives the enemy a route");
  State intercepted=open;
  check(apply_checked(intercepted,raid,rules,undo,info) && !intercepted.alive[51],
        "intervening enemy can remove the prospective reinforcement");
  for (auto* state : {&open,&blocked})
    check(apply_checked(*state,wait,rules,undo,info) && state->to_move==2,"partner gets the next decision");
  check(is_legal(open,reinforce) && !is_legal(blocked,reinforce),"partner route depends on clearing the line");
  std::cout<<"PASS partner corridor and intervening-enemy counterexample\n";
}
void camp_cutoff_contract() {
  const auto& board=Board::get();
  const auto n=[&](int seat,int row,int col){return uint8_t(board.seat_node(seat,row,col));};
  const Move seal{n(0,1,0),n(0,1,1)}, press{seal.from,n(0,0,0)};
  const auto other_camp=n(0,1,3);
  const Rules rules;
  for (int column : {1,2}) {
    const auto target=n(0,0,column);
    for (int rank : {Company,Engineer}) {
      State initial{};
      std::fill(std::begin(initial.id_at),std::end(initial.id_at),int8_t(-1));
      std::fill(std::begin(initial.seat_at),std::end(initial.seat_at),int8_t(-1));
      std::fill(std::begin(initial.rank_at),std::end(initial.rank_at),int8_t(Unknown));
      const auto put=[&](int id,uint8_t node,int piece_rank) {
        initial.id_at[node]=int8_t(id); initial.seat_at[node]=int8_t(id/kPiecesPerSeat);
        initial.rank_at[node]=int8_t(piece_rank); initial.rank_of[id]=int8_t(piece_rank);
        initial.node_of[id]=node; initial.alive[id]=true;
      };
      for (int seat=0; seat<kSeats; ++seat) put(seat*kPiecesPerSeat,n(seat,5,1),Flag);
      put(1,seal.from,Regiment); put(kPiecesPerSeat+1,target,rank);
      const auto before=observe(initial,0);
      check(before.rank_at(target)==Unknown && short_features(before,seal)==short_features(before,press),
            "same mover's quiet camp and rail moves must reproduce the five-input alias");
      const auto replies=[&](Move first) {
        auto branch=initial;
        State undo; StepInfo info;
        check(apply_checked(branch,first,rules,undo,info) && branch.to_move==1,
              "legal camp/rail branch followed by target seat");
        std::vector<Move> moves;
        gen_piece_moves(StateView{branch},1,target,moves);
        return moves;
      };
      const auto a=replies(seal), b=replies(press);
      const auto contains=[&](const auto& moves,uint8_t dest) {
        return std::find(moves.begin(),moves.end(),Move{target,dest})!=moves.end();
      };
      check(!contains(a,seal.to) && contains(b,seal.to),"target loses entry to the occupied camp");
      if (column==2)
        check(contains(a,other_camp) && contains(b,other_camp),"sealing one camp leaves the other available");
      std::cout<<"PASS camp cutoff: column="<<column<<" rank_case="<<(rank==Engineer ? "engineer" : "ordinary")
               <<" target_moves_seal="<<a.size()<<" target_moves_press="<<b.size()<<'\n';
    }
  }
}
void return_estimator_contract() {
  OpeningEpisode episode; episode.score=.5; episode.decisions=1; episode.gradient[0]=2;
  ShortWeights gradient{};
  check(add_team_return(gradient,episode,.25) && gradient[0]==-.5,
        "past return baseline enters policy advantage");
  episode.score=std::nullopt;
  const auto previous=gradient;
  check(!add_team_return(gradient,episode,.25) && gradient==previous,
        "unknown truncation does not update baseline estimator");
}
int main() {
  try {
    auto start=std::chrono::steady_clock::now(); config_tests(); movement(); combat(); public_evidence(); opening_marks(); survivor_evidence(); evidence_policy_contract(); termination(); reveals_and_elimination();
    seat_episode_contract(); return_estimator_contract(); partner_corridor_contract(); camp_cutoff_contract();
    mine_mark_regression(); mine_formation_contract(); mode_observation_contract(); single_dark_rule_contract();
    single_match_contract();
    int counts[5]={}, total=0;
    for(int seed=0;seed<100;++seed) {
      int a,b,p,q; auto h=game(seed,a,p); check(h==game(seed,b,q)&&a==b&&p==q,"determinism");
      ++counts[a]; total+=p;
    }
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"PASS config, moves, combat, isolation, invalid, undo; 100 seeds x2; plies="<<total<<" results=";
    for(int v:counts)std::cout<<v<<',';
    std::cout<<" seconds="<<seconds<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
