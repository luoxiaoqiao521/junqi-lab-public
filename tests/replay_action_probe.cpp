#include "state.h"
#include "layout.h"
#include "../observe/filter.h"
#include "../strategy/short_policy.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace jq;
using Features=std::array<double,9>;

static Features replay_features(const Observation& obs,Move m) {
  const auto base=short_features(obs,m,false);
  const auto& board=Board::get();
  const auto& from=board.node(m.from);
  const auto& to=board.node(m.to);
  return {base[0],base[1],base[2],base[3],base[4],
          double(to.kind==NodeKind::Camp),double(from.kind==NodeKind::Camp),
          (enemy_hq_distance(obs.viewer,m.from)-enemy_hq_distance(obs.viewer,m.to))/12.0,
          double(from.seat!=to.seat)};
}

static double value(const Observation& obs,Move m,const Features& weights) {
  const auto f=replay_features(obs,m);
  double score=0;
  for(size_t i=0;i<f.size();++i) score+=f[i]*weights[i];
  return score;
}

static void set_cell(State& state,int node,int seat,int rank) {
  state.seat_at[node]=static_cast<int8_t>(seat);
  state.rank_at[node]=static_cast<int8_t>(rank);
}

static int find_node(int x,int y) {
  return x<0||x>16||y<0||y>16?kNone:Board::get().at_grid(y,x);
}

static bool emit_action(State& board,Move selected,int split) {
  const auto obs=observe(board,board.to_move);
  std::vector<Move> actions;
  gen_moves(obs,obs.viewer,actions);
  auto chosen=std::find(actions.begin(),actions.end(),selected);
  if(chosen==actions.end()) return false;
  std::cout<<split<<' '<<actions.size()<<' '<<chosen-actions.begin();
  for(auto action:actions)
    for(double x:replay_features(obs,action)) std::cout<<' '<<x;
  std::cout<<'\n';
  return true;
}

static void recorded_move(State& state,Move m,int outcome) {
  const int actor=state.to_move;
  const int defender=state.seat_at[m.to], rank=state.rank_at[m.from], hidden=state.rank_at[m.to];
  if(rank==Marshal && (outcome==2||outcome==3)) state.flag_revealed[actor]=true;
  if(hidden==Marshal && (outcome==1||outcome==3)) state.flag_revealed[defender]=true;
  set_cell(state,m.from,kNone,Unknown);
  if(outcome<=1) set_cell(state,m.to,actor,rank);
  else if(outcome==3) set_cell(state,m.to,kNone,Unknown);
  ++state.ply;
}

static bool read_game(int split,int rows) {
  State board{};
  for(int n=0;n<kNodes;++n) set_cell(board,n,kNone,Unknown);
  for(int j=0;j<100;++j) {
    int x,y,rank;
    if(!(std::cin>>x>>y>>rank)) return false;
    const int node=find_node(x,y);
    if(node<0||rank<2||rank>13) return false;
    set_cell(board,node,Board::get().node(node).seat,rank<5?rank-2:16-rank);
  }
  for(int j=0;j<rows;++j) {
    int x,y,xx,yy,outcome;
    if(!(std::cin>>x>>y>>xx>>yy>>outcome)) return false;
    if(x==-1) {
      if(y>=0 && (xx==3||xx==4))
        for(int n=0;n<kNodes;++n)
          if(board.seat_at[n]==y) set_cell(board,n,kNone,Unknown);
      continue;
    }
    const int from=find_node(x,y),to=find_node(xx,yy);
    if(from<0||to<0||board.seat_at[from]<0||board.ply>=40) return false;
    board.to_move=board.seat_at[from];
    const Move move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
    if(!emit_action(board,move,split)) return false;
    recorded_move(board,move,outcome);
  }
  return true;
}

static Move rank_move(const Observation& obs,const Features& weights,Rng& rng) {
  std::vector<Move> actions;
  gen_moves(obs,obs.viewer,actions);
  std::vector<double> logits;
  for(auto action:actions) logits.push_back(value(obs,action,weights));
  const auto top=*std::max_element(logits.begin(),logits.end());
  double total=0;
  for(auto& x:logits) {x=std::exp(x-top);total+=x;}
  double u=double(rng.next()>>11)*0x1.0p-53*total;
  for(size_t i=0;i<actions.size();++i) {u-=logits[i];if(u<0) return actions[i];}
  return actions.back();
}

static int simulate(State state,const Rules& rules,int seat,int context,uint64_t seed,
                    const Features& weights,const Features* after_opening) {
  std::array<Rng,4> streams{Rng(mix_seed(seed,0)),Rng(mix_seed(seed,1)),
                            Rng(mix_seed(seed,2)),Rng(mix_seed(seed,3))};
  while(state.result==Result::Ongoing) {
    const int actor=state.to_move;
    const auto obs=observe(state,actor);
    Move m;
    const auto& current=after_opening && state.ply>=40?*after_opening:weights;
    if(actor==seat) m=rank_move(obs,current,streams[actor]);
    else if(context && actor==(seat+3)%4) heuristic_move(obs,streams[actor],m);
    else random_move(obs,streams[actor],m);
    step(state,m,rules);
  }
  if(state.result==Result::Truncated) return -1;
  if(state.result==Result::Draw) return 1;
  return 2*int(state.result==(seat%2?Result::Team1Wins:Result::Team0Wins));
}

static bool read_layouts(int8_t (&layouts)[4][25]) {
  for(int j=0;j<100;++j) {
    int x,y,rank;
    if(!(std::cin>>x>>y>>rank)||rank<2||rank>13) return false;
    const int node=find_node(x,y);
    if(node<0) return false;
    const int seat=Board::get().node(node).seat;
    if(seat<0) return false;
    for(int slot=0;slot<25;++slot)
      if(layout_slot_node(seat,slot)==node) layouts[seat][slot]=static_cast<int8_t>(rank<5?rank-2:16-rank);
  }
  for(const auto& layout:layouts) if(!valid_layout(layout)) return false;
  return true;
}

static bool evaluate(uint64_t first,bool opening_only,bool from_replay=false) {
  std::array<Features,3> weights;
  for(auto& policy:weights) for(auto& w:policy) if(!(std::cin>>w) || !std::isfinite(w)) return false;
  Rules rules;
  rules.ccw=false; rules.marshal_reveals_flag=true; rules.no_progress_limit=71;
  rules.check_involved_no_move=true;
  int games=256;
  if(from_replay && (!(std::cin>>games)||games<1||games>256)) return false;
  for(uint64_t seed=first;seed<first+static_cast<uint64_t>(games);++seed) {
    Rng rng(seed);
    int8_t layouts[4][25]{};
    int starter;
    if(from_replay) {
      if(!(std::cin>>starter)||starter<0||starter>3||!read_layouts(layouts)) return false;
    } else {
      for(auto& layout:layouts) generate_layout(layout,rng);
      starter=static_cast<int>(rng.below(4));
    }
    const int seat=static_cast<int>(seed%4);
    State state;
    init_state(state,layouts,starter,rules);
    for(int context=0;context<2;++context) {
      std::cout<<seed<<' '<<context;
      for(const auto& model:weights)
        std::cout<<' '<<simulate(state,rules,seat,context,seed,model,opening_only?&weights[0]:nullptr);
      std::cout<<'\n';
    }
  }
  return true;
}

int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="evaluate") return evaluate(313800,false)?0:3;
  if(argc==2 && std::string(argv[1])=="fresh-full") return evaluate(314300,false)?0:3;
  if(argc==2 && std::string(argv[1])=="fresh-opening") return evaluate(314300,true)?0:3;
  if(argc==2 && std::string(argv[1])=="real-full") return evaluate(314700,false,true)?0:3;
  if(argc==2 && std::string(argv[1])=="real-opening") return evaluate(314700,true,true)?0:3;
  int count;
  if(!(std::cin>>count)) return 2;
  for(int j=0;j<count;++j) {
    int split,rows;
    if(!(std::cin>>split>>rows)||!read_game(split,rows)) return 4;
  }
  return 0;
}
