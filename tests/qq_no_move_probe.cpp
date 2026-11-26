#include "state.h"
#include "layout.h"
#include <algorithm>
#include <array>
#include <iostream>

using namespace jq;

struct Row { int x,y,xx,yy,result; };

static int node_at(int x,int y) {
  return x<0||x>16||y<0||y>16?kNone:Board::get().at_grid(y,x);
}
static void put(State& s,int node,int seat,int rank) {
  s.seat_at[node]=static_cast<int8_t>(seat);
  s.rank_at[node]=static_cast<int8_t>(rank);
}
static void remove_seat(State& s,int seat) {
  s.out[seat]=true;
  for(int i=0;i<kNodes;++i)
    if(s.seat_at[i]==seat) put(s,i,kNone,Unknown);
}
static bool game_over(const State& s) {
  return (s.out[0]&&s.out[2]) || (s.out[1]&&s.out[3]);
}

static bool movable_kind(const State& s,int seat) {
  for(int n=0;n<kNodes;++n)
    if(s.seat_at[n]==seat && s.rank_at[n]>=Bomb && Board::get().node(n).kind!=NodeKind::HQ)
      return true;
  return false;
}
static bool legal_move_exists(const State& s,int seat) {
  std::vector<Move> exits;
  for(int n=0;n<kNodes;++n) {
    if(s.seat_at[n]!=seat || s.rank_at[n]<Bomb || Board::get().node(n).kind==NodeKind::HQ) continue;
    gen_piece_moves(StateView{s},seat,n,exits);
    if(!exits.empty()) return true;
  }
  return false;
}

static unsigned forecast(State s,int actor,bool ccw,int immediate,unsigned scan) {
  unsigned removed=0;
  const int direction=ccw?1:3;
  if(immediate) {
    for(int j=1;j<=4&&!game_over(s);++j) {
      const int seat=(actor+j*direction)%4;
      if(!(scan&(1u<<seat)) || s.out[seat] ||
         (immediate==1?movable_kind(s,seat):legal_move_exists(s,seat))) continue;
      removed|=1u<<seat;
      remove_seat(s,seat);
    }
  }
  for(int j=1;j<=4&&!game_over(s);++j) {
    const int seat=(actor+j*direction)%4;
    if(s.out[seat]) continue;
    if(legal_move_exists(s,seat)) break;
    removed|=1u<<seat;
    remove_seat(s,seat);
  }
  return removed;
}

static int bits(unsigned n) {
  int count=0;
  while(n) {count+=n%2; n/=2;}
  return count;
}

struct Summary {
  int games=0,plies=0,contacts=0,bad_legal=0,bad_contact=0,no_moves=0,no_move_mismatches=0;
  int eligible=0,excluded=0,eligible_no_move=0;
  std::array<int,5> wrong_turns{},missed{},extra{};
  void print() const {
    std::cout<<games<<' '<<plies<<' '<<contacts<<' '<<bad_legal<<' '<<bad_contact
      <<' '<<no_moves<<' '<<no_move_mismatches<<' '<<eligible<<' '<<excluded<<' '<<eligible_no_move;
    for(size_t i=0;i<wrong_turns.size();++i) std::cout<<' '<<wrong_turns[i]<<' '<<missed[i]<<' '<<extra[i];
    std::cout<<'\n';
  }
};

struct Window {
  std::array<unsigned,5> prediction{};
  unsigned actual=0;
  bool present=false, external=false;
  void flush(Summary& stats) const {
    if(!present) return;
    if(external) {++stats.excluded; return;}
    ++stats.eligible;
    stats.eligible_no_move+=bits(actual);
    for(size_t i=0;i<prediction.size();++i) {
      stats.wrong_turns[i]+=prediction[i]!=actual;
      stats.missed[i]+=bits(actual&~prediction[i]);
      stats.extra[i]+=bits(prediction[i]&~actual);
    }
  }
};

static int recorded_move(State& s,Row row,Summary& stats) {
  int from=node_at(row.x,row.y),to=node_at(row.xx,row.yy);
  if(from<0||to<0||s.seat_at[from]<0||row.result<0||row.result>3) return kNone;
  const int actor=s.seat_at[from], rank=s.rank_at[from];
  std::vector<Move> options;
  gen_piece_moves(StateView{s},actor,from,options);
  const Move move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
  stats.bad_legal+=std::find(options.begin(),options.end(),move)==options.end();
  if(row.result) {
    ++stats.contacts;
    stats.bad_contact+=int(resolve(rank,s.rank_at[to],Rules{}))!=row.result;
  }
  ++stats.plies;
  put(s,from,kNone,Unknown);
  if(row.result<=1) put(s,to,actor,rank);
  else if(row.result==3) put(s,to,kNone,Unknown);
  return actor;
}

static Window move_window(const State& s,int actor,int defender,bool captured_flag,bool ccw) {
  State provisional=s;
  if(captured_flag) remove_seat(provisional,defender);
  Window window;
  window.present=true;
  for(int i=0;i<3;++i) window.prediction[i]=forecast(provisional,actor,ccw,i,15);
  const unsigned actor_bit=1u<<actor;
  const unsigned involved=actor_bit|(defender<0?0u:1u<<defender);
  window.prediction[3]=forecast(provisional,actor,ccw,2,involved);
  window.prediction[4]=forecast(provisional,actor,ccw,2,defender<0?actor_bit:15);
  return window;
}

static void control(State& s,Row row,Window& window,Summary& stats) {
  if(row.xx==2 || (row.xx==3 && row.result!=1 && row.result!=2) ||
      (row.xx==4 && row.y>=0 && !s.out[row.y]) ||
      (row.xx==5 && row.result!=4)) window.external=true;
  if(row.y<0 || (row.xx!=3 && row.xx!=4)) return;
  if(row.xx==3 && row.result==2) {
    window.actual|=1u<<row.y;
    ++stats.no_moves;
    stats.no_move_mismatches+=legal_move_exists(s,row.y);
  }
  remove_seat(s,row.y);
}

static bool read_game(int count,bool ccw,Summary& stats) {
  State s{};
  for(int n=0;n<kNodes;++n) put(s,n,kNone,Unknown);
  for(int m=0;m<100;++m) {
    int x,y,r;
    std::cin>>x>>y>>r;
    const int n=node_at(x,y);
    if(!std::cin||n<0||r<2||r>13||s.seat_at[n]>=0) return false;
    const auto& node=Board::get().node(n);
    if(node.seat<0||node.kind==NodeKind::Camp) return false;
    put(s,n,node.seat,r<5?r-2:16-r);
  }
  Window window;
  for(int p=0;p<count;++p) {
    Row row;
    if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return false;
    if(row.x==-1) { control(s,row,window,stats); continue; }
    window.flush(stats);
    const int target=node_at(row.xx,row.yy);
    if(target<0) return false;
    const int defender=s.seat_at[target];
    const bool captured=s.rank_at[target]==Flag && (row.result==1||row.result==3);
    const int actor=recorded_move(s,row,stats);
    if(actor<0) return false;
    window=move_window(s,actor,defender,captured,ccw);
  }
  window.flush(stats);
  ++stats.games;
  return true;
}

static bool strict_game(Rules rules,int rows,int8_t (&layouts)[4][25]) {
  State state;
  unsigned emitted=0, recorded=0;
  int moves=0, notices=0;
  for(int i=0;i<rows;++i) {
    Row row;
    if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return false;
    if(row.x==-1) {
      if(row.xx==3 && row.result==2 && row.y>=0 && row.y<4) {
        ++notices;
        recorded|=1u<<row.y;
      } else if(row.xx!=5 || row.result!=4) return false;
      continue;
    }
    if(emitted!=recorded) return false;
    const int from=node_at(row.x,row.y),to=node_at(row.xx,row.yy);
    if(from<0||to<0) return false;
    if(!moves) init_state(state,layouts,Board::get().node(from).seat,rules);
    const Move move{static_cast<uint8_t>(from), static_cast<uint8_t>(to)};
    if(!is_legal(state,move)) return false;
    auto event=step(state,move,rules);
    if(int(event.outcome)!=row.result) return false;
    emitted=event.eliminated_mask;
    recorded=0;
    ++moves;
  }
  if(moves!=245||notices!=2||emitted!=recorded||state.result!=Result::Team0Wins) return false;
  std::cout<<"PASS strict archive replay plies="<<moves<<" no_move_notices="<<notices<<" result="<<int(state.result)<<'\n';
  return true;
}

static bool strict_fixture() {
  int rows,ccw;
  if(!(std::cin>>rows>>ccw)) return false;
  int8_t layouts[4][25]{};
  for(int m=0;m<100;++m) {
    int x,y,r;
    if(!(std::cin>>x>>y>>r)||r<2||r>13) return false;
    const int n=node_at(x,y);
    if(n<0) return false;
    const int seat=Board::get().node(n).seat;
    if(seat<0) return false;
    for(int slot=0;slot<25;++slot) {
      if(layout_slot_node(seat,slot)==n) layouts[seat][slot]=static_cast<int8_t>(r<5?r-2:16-r);
    }
  }
  for(const auto& layout:layouts) if(!valid_layout(layout)) return false;
  Rules rules;
  rules.ccw=ccw;
  rules.check_involved_no_move=true;
  rules.marshal_reveals_flag=true;
  rules.no_progress_limit=71;
  return strict_game(rules,rows,layouts);
}

int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="strict") return strict_fixture()?0:3;
  std::array<Summary,3> sources;
  int count;
  if(!(std::cin>>count)||count<=0) return 1;
  for(int g=0;g<count;++g) {
    int source,rows,ccw;
    if(!(std::cin>>source>>rows>>ccw)||source<0||source>=3||!read_game(rows,ccw,sources[source]))
      return 2;
  }
  for(const auto& stats:sources) stats.print();
  return 0;
}
