#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#define __global__
using MUSTRATEGO_FLOAT_CUDA_DTYPE=float;
constexpr int NUM_ACTIONS=1800;
struct Index { int x; };
static thread_local Index blockIdx{0},blockDim{1},threadIdx{0};

#include "upstream_board.inc"
#include "upstream_action.inc"
#include "upstream_infostate.inc"

struct TwosquareState { uint8_t A,B,C,D; };
struct ChaseState { uint8_t* last_dst_pos[2]; uint8_t* last_src_pos[2]; int32_t* chase_length[2]; };
constexpr int MAX_CHASE_LENGTH=210;
using std::min;
using std::max;
#include "upstream_two_square.inc"
#include "upstream_chase.inc"
#include "upstream_terminal.inc"

namespace {
constexpr int kMemory=32, kPlanes=355+kMemory, kBuffer=MAX_CHASE_LENGTH+1;
constexpr int kQuietLimit=200, kCap=4000;
struct Match {
  std::array<StrategoBoard,kBuffer> boards{}; StrategoBoard initial{};
  std::array<int32_t,kBuffer> moves{}, actions{};
  std::array<uint8_t,kBuffer> terminated{};
  int32_t ply=0, quiet=0, since_reset=0;
  uint8_t actor=1, flag=0;
  int result=0;
  TwosquareState two_square[2]={{255,255,255,255},{255,255,255,255}};
  uint8_t chase_dst[2]={238,238},chase_src[2]={238,238};
  int32_t chase_length[2]{};
  int32_t chase_actions[MAX_CHASE_LENGTH]{};
  ChaseState chase() {return {{chase_dst,chase_dst+1},{chase_src,chase_src+1},{chase_length,chase_length+1}};}
};
template<class F> void launch(unsigned count,F operation) {
  for (unsigned i=0;i<count;++i) {
    threadIdx.x=int(i);
    operation();
  }
}
bool valid_layout(const uint8_t* layout) {
  if (!layout) return false;
  std::array<uint8_t,14> counts{};
  for (int i=0;i<40;++i) {
    if (layout[i]>=12) return false;
    ++counts[layout[i]];
  }
  return counts==CLASSIC_INITIAL_COUNTS;
}
void initialize(Match& match,const uint8_t* red,const uint8_t* blue) {
  auto& board=match.boards[0];
  board.prev_dst_abs=board.prev_prev_dst_abs=board.last_moved_piece_type=255;
  board.num_hidden_unmoved[0]=board.num_hidden_unmoved[1]=40;
  for (int p=0;p<12;++p) board.num_hidden[0][p]=board.num_hidden[1][p]=CLASSIC_INITIAL_COUNTS[p];
  for (int n=0;n<100;++n) {
    auto& cell=board.pieces[n/10][n%10];
    const int col=n%10;
    if (n<40) cell=_R(static_cast<PieceType>(red[n]),n);
    else if (n>=60) cell=_B(static_cast<PieceType>(blue[99-n]),(99-n));
    else if (col==2 || col==3 || col==6 || col==7) cell=_L(LAKE);
    else cell=_E(EMPTY);
  }
}
void legal(const Match& m,bool* mask,uint8_t actor,bool use_rules) {
  std::fill(mask,mask+NUM_ACTIONS,false);
  if (m.result) return;
  const auto* b=&m.boards[m.ply%kBuffer];
  const auto* terminal=m.terminated.data()+m.ply%kBuffer;
  launch(100,[&]{LegalActionsMaskKernel(mask,terminal,1,actor,b,false);});
  if (use_rules) {
    launch(1,[&]{RemoveTwosquareActionsKernel(mask,&m.two_square[actor-1],terminal,1);});
    launch(MAX_CHASE_LENGTH,[&]{RemoveIllegalChaseMovesKernel(mask,m.chase_actions,terminal,1);});
  }
}
void features(const Match& m,float* out,uint8_t* ids) {
  std::fill(out,out+kPlanes*100,0.f);
  const uint32_t stride=kPlanes*100;
  const auto* board=&m.boards[m.ply%kBuffer];
  const auto* first=&m.initial;
  launch(100,[&]{
    BoardStateKernel__OwnPieceTypes(out,m.actor,1,board,stride);
    BoardStateKernel__ProbTypes(out,m.actor,false,1,board,stride,1200);
    BoardStateKernel__ProbTypes(out,3-m.actor,true,1,board,stride,2400);
    BoardStateKernel__InvisiblesEmptyAndMoved(out,m.actor,1,&m.ply,&m.quiet,kCap,kQuietLimit,board,stride);
    BoardStateKernel__ThreatEvadeActiveAdj(out,m.actor,1,board,stride);
    BoardStateKernel__Deaths(out,m.actor,1,board,first,stride);
    BoardStateKernel__Protections(out,m.actor,1,board,stride);
  });
  launch(60,[&]{BoardStateKernel__DeathReasons(out,m.actor,1,board,stride);});
  launch(kMemory,[&]{
    InjectInfostateSrcDstKernel(out+35500,m.actions.data(),nullptr,m.terminated.data(),
      m.moves.data(),m.moves.data(),m.ply,kMemory,kBuffer,1,stride);
  });
  for (int n=0;n<100;++n) {
    const auto& piece=board->pieces[n/10][n%10];
    const int pos=m.actor==1?n:99-n;
    ids[pos]=(piece.color==3-m.actor)?99-piece.piece_id:piece.piece_id;
  }
}
}
#define API extern "C" __declspec(dllexport)
API Match* make_match(const uint8_t* red,const uint8_t* blue) {
  if (!valid_layout(red)||!valid_layout(blue)) return nullptr;
  auto* match=new Match;
  initialize(*match,red,blue);
  match->initial=match->boards[0];
  std::fill_n(match->chase_actions,MAX_CHASE_LENGTH,-1);
  return match;
}

API void free_match(Match* match) {delete match;}
API int query(Match* match,float* out,uint8_t* ids,uint8_t* legal_bytes) {
  if (match->result) return -match->result;
  bool mask[NUM_ACTIONS]{};
  legal(*match,mask,match->actor,true);
  std::copy(mask,mask+NUM_ACTIONS,legal_bytes);
  features(*match,out,ids);
  return match->actor;
}
API int play(Match* match,int32_t action) {
  if (action<0 || action>=NUM_ACTIONS || match->result) return -1;
  bool mask[NUM_ACTIONS]{};
  legal(*match,mask,match->actor,true);
  if (!mask[action]) return -1;
  auto& m=*match;
  const int row=m.ply%kBuffer;
  const int next=(m.ply+1)%kBuffer;
  m.boards[next]=m.boards[row];
  m.actions[row]=action;
  uint8_t summary[6]{},red_death=0,blue_death=0;
  auto* terminal=&m.terminated[next];
  launch(1,[&]{
    UpdateTwosquareActionKernel(&m.two_square[m.actor-1],&action,terminal,1);
    ApplyActionsKernel(&m.boards[next],&m.ply,&m.quiet,&m.since_reset,&m.flag,summary,
      &red_death,&blue_death,terminal,1,m.actor,&action);
    UpdateTwosquareDeathKernel(&m.two_square[0],&red_death,terminal,1);
    UpdateTwosquareDeathKernel(&m.two_square[1],&blue_death,terminal,1);
    UpdateChaseStateKernel(1,m.chase(),&m.boards[next],summary,terminal,m.actor);
  });
  m.moves[next]=m.ply;
  m.actor=3-m.actor;
  std::fill_n(m.chase_actions,MAX_CHASE_LENGTH,-1);
  launch(MAX_CHASE_LENGTH,[&]{
    ComputeIllegalChaseMovesKernel(m.chase_actions,m.chase(),m.boards.data(),nullptr,m.moves.data(),
      terminal,m.ply,kBuffer,m.actor,0,1);
  });
  uint8_t has_legal=0;
  for (uint8_t actor:{1,2}) {
    legal(m,mask,actor,actor==m.actor);
    if (std::any_of(mask,mask+NUM_ACTIONS,[](bool x){return x;})) has_legal|=actor;
  }
  launch(1,[&]{IncrementTerminationCounterKernel(terminal,&m.flag,&m.ply,&m.quiet,&m.since_reset,
    &has_legal,1,kCap,kQuietLimit);});
  if (!*terminal) return 0;
  float reward=0;
  launch(1,[&]{ComputeRewardPl0Kernel(&reward,terminal,&m.ply,&m.quiet,&m.flag,&has_legal,1,kCap,kQuietLimit,m.actor);});
  return m.result=(reward>0?1:reward<0?2:3);
}


