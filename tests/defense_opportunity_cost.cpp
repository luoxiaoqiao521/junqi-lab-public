#include "archive_replay_prefix.h"
#include "public_defender_probe.h"

using namespace jq;
using namespace jq::archive_probe;

static Observation quiet_position(Observation view,Move action) {
  view.seats[action.to]=view.seats[action.from];
  view.ranks[action.to]=view.ranks[action.from];
  view.seats[action.from]=kNone;
  view.ranks[action.from]=Unknown;
  return view;
}

static bool can_deploy(const Observation& view,int enemy,int origin,int front,unsigned ranks,
                       const Rules& rules) {
  for(int sample:{Platoon,Engineer}) {
    const unsigned mode=sample==Engineer ? 1u<<Engineer :
        all_evidence_ranks & ~((1u<<Flag)|(1u<<Mine)|(1u<<Bomb)|(1u<<Engineer));
    const auto supported=mode&ranks;
    if(!supported) continue;
    auto trial=view;
    trial.ranks[origin]=static_cast<int8_t>(sample);
    std::vector<Move> moves;
    gen_piece_moves(trial,enemy,origin,moves);
    if(!std::any_of(moves.begin(),moves.end(),[&](Move m){return m.to==front;})) continue;
    const int blocking=view.rank_at(front);
    if(view.seat_at(front)<0 || blocking==Unknown) return true;
    for(int rank=Engineer;rank<RankCount;++rank)
      if((supported&(1u<<rank)) && resolve(rank,blocking,rules)==Outcome::AttackerWins) return true;
  }
  return false;
}

static int approach_routes(const Observation& view,const OpeningMarks& marks,const Rules& rules) {
  const int enemy=next_visible_actor(view,rules);
  if(view.team(enemy)==view.team(view.viewer)) return 0;
  const auto& board=Board::get();
  int front=kNone;
  for(int col:{1,3})
    if(view.rank_at(board.seat_node(view.viewer,5,col))==Flag) front=board.seat_node(view.viewer,4,col);
  if(front==kNone) return 0;
  int origins=0;
  for(int n=0;n<kNodes;++n)
    if(view.seat_at(n)==enemy && can_deploy(view,enemy,n,front,compatible_rank_mask(view,marks,n),rules))
      ++origins;
  return origins;
}

static int window_threat(const Observation& view,const OpeningMarks& marks,const Rules& rules,bool early) {
  return early ? approach_routes(view,marks,rules) : visible_flag_threat(view,marks,rules).origins;
}

static bool blocks_flag_route(const Observation& before,const OpeningMarks& marks,Move action,
                              const Rules& rules,bool early) {
  if(before.seat_at(action.to)>=0) return false;
  const auto after=quiet_position(before,action);
  auto updated=marks;
  advance_public_ranks(updated,before,after,action,Outcome::Move);
  return window_threat(after,updated,rules,early)==0;
}


static bool removes_attacker(const Observation& view,const OpeningMarks& marks,Move move,
                             const Rules& rules,const PublicThreat& threat) {
  if(threat.origins!=1 || !threat.forced || move.to!=threat.target) return false;
  const auto outcomes=collision_outcome_mask(view.rank_at(move.from),
      compatible_rank_mask(view,marks,move.to),rules);
  return outcomes && !(outcomes & (1u<<static_cast<unsigned>(Outcome::DefenderWins)));
}

static void compare(const State& state,const Rules& rules,const ShortWeights& weights,
                    const std::string& id,const std::array<std::optional<Move>,5>& options) {
  Rules trial=rules;
  trial.truncation_cap=state.ply+240;
  const int viewer=state.to_move;
  for(int variant:{0,2}) for(int i=0;i<12;++i) {
    const auto seed=mix_seed(935231+state.ply,static_cast<uint64_t>(i));
    std::array<int,5> score{};
    for(size_t j=0;j<options.size();++j) {
      const auto old=std::find(options.begin(),options.begin()+j,options[j]);
      if(!options[j]) score[j]=-2;
      else if(old!=options.begin()+j) score[j]=score[old-options.begin()];
      else score[j]=review_outcome(review_rollout(state,*options[j],trial,variant,true,weights,seed),viewer);
    }
    std::cout<<"P "<<id<<' '<<variant<<' '<<i;
    for(int value:score) std::cout<<' '<<value;
    std::cout<<'\n';
  }
}

static void inspect(const State& state,Move actual,const Rules& rules,const ShortWeights& weights,
                    const OpeningMarks& marks,const std::string& id,bool early) {
  const auto d=seat_decision(state);
  const auto& view=d.observation;
  const auto reference=review_reference(d);
  const auto frozen=d.actions[std::max_element(reference.begin(),reference.end())-reference.begin()];
  const auto threat=visible_flag_threat(view,marks,rules);
  const int pressure=early ? (threat.origins ? 0 : approach_routes(view,marks,rules)) : threat.origins;
  std::array<std::optional<Move>,5> choices{frozen,actual,std::nullopt,std::nullopt,std::nullopt};
  const auto pick=[&](std::optional<Move>& current,Move action) {
    if(!current || score_move(view,action).total()>score_move(view,*current).total()) current=action;
  };
  if(pressure) {
    for(auto action:d.actions) {
      if(!early && removes_attacker(view,marks,action,rules,threat)) pick(choices[2],action);
      if(blocks_flag_route(view,marks,action,rules,early)) pick(choices[3],action);
    }
    if(!early) choices[4]=public_flag_finish(d,frozen,rules);
  }
  std::cout<<"D "<<id<<' '<<state.ply+1<<' '<<int(state.to_move)<<' '<<next_visible_actor(view,rules)
           <<' '<<pressure<<' '<<threat.forced;
  for(const auto& move:choices) {
    if(!move) { std::cout<<" -1 -1 -1 -1"; continue; }
    auto after=state;
    step(after,*move,rules);
    std::cout<<' '<<int(move->from)<<' '<<int(move->to)<<' '<<view.rank_at(move->from)
             <<' '<<next_enemy_flag_attack(after,state.to_move);
  }
  std::cout<<'\n';
  if(pressure && std::any_of(choices.begin()+2,choices.end(),
     [&](const auto& move){ return move && !(*move==frozen); }))
    compare(state,rules,weights,id,choices);
}

int main(int argc,char** argv) {
  const bool early=argc==2 && std::string(argv[1])=="approach";
  try {
    verify_threat_fixture();
    int count;
    ShortWeights weights{};
    if(!(std::cin>>count)||count<1) return 2;
    for(size_t j=0;j<5;++j) if(!(std::cin>>weights[j])||!std::isfinite(weights[j])) return 2;
    for(int i=0;i<count;++i) {
      std::string id;
      int size,direction;
      if(!(std::cin>>id>>size>>direction)||size<1||size>2000||(direction!=0&&direction!=1)) return 2;
      int8_t layouts[4][25]{};
      if(!read_layouts(layouts)) return 2;
      std::vector<Row> rows(static_cast<size_t>(size));
      for(auto& row:rows) if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return 2;
      Rules rules;
      rules.ccw=direction!=0;
      rules.marshal_reveals_flag=true;
      rules.no_progress_limit=71;
      rules.check_involved_no_move=true;
      State state{};
      MovedSquares moved{};
      Move actual;
      DefenderEvidence evidence;
      const ReplayObserver track=[&](const State& before,const StepInfo& event,const State& after) {
        evidence.track(before,event,after,rules);
      };
      if(!archive_prefix(state,moved,actual,rules,layouts,rows,true,track) ||
         !evidence.contact[state.to_move]) {
        std::cout<<"X "<<id<<'\n';
        continue;
      }
      inspect(state,actual,rules,weights,*evidence.contact[state.to_move],id,early);
    }
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 3;
  }
}
