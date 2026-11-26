#include "../sim/replay.h"
#include "../sim/seat_episode.h"
#include <algorithm>
#include <deque>

using namespace jq;

namespace {
enum class Defense { Unsafe, Safe, Unknown };
struct Cut { State state; Move actual; };

Defense defend(const State& state,int flag,int owner,int remaining,const Rules& rules,int& left) {
  if (state.result!=Result::Ongoing) {
    const auto score=team_score(state.result,owner,state.match);
    if (score && *score==1) return Defense::Safe;
    if (score && *score==0) return Defense::Unsafe;
    if (!score) return Defense::Unknown;
  }
  if (state.rank_at[flag]!=Flag) return Defense::Unsafe;
  if (!remaining || state.result==Result::Draw) return Defense::Safe;
  if (!left) return Defense::Unknown;
  const bool defender=team_of(state.to_move)==team_of(owner);
  const auto goal=defender?Defense::Safe:Defense::Unsafe;
  std::vector<Move> actions;
  legal_moves(state,actions);
  expect_value(!actions.empty(),"unfinished state has no action");
  std::stable_partition(actions.begin(),actions.end(),[&](Move move){return move.to==flag;});
  bool unknown=false;
  for (auto move:actions) {
    if (!left) return Defense::Unknown;
    --left;
    auto child=state;
    step(child,move,rules);
    const auto result=defend(child,flag,owner,remaining-1,rules,left);
    if (result==goal) return goal;
    unknown|=result==Defense::Unknown;
  }
  if (unknown) return Defense::Unknown;
  return defender?Defense::Unsafe:Defense::Safe;
}

int move_key(Move move) { return move.from*kNodes+move.to; }

void inspect(const Cut& cut,int flag,int owner,int loss,const Rules& rules) {
  const auto& state=cut.state;
  const int depth=std::min(loss-state.ply,4);
  const auto observation=observe(state,state.to_move);
  const DecisionScorer scorer(observation,rules,DecisionKind::FlagRoute);
  std::vector<Move> actions;
  gen_moves(observation,state.to_move,actions);
  expect_value(is_legal(state,cut.actual),"recorded practice move is illegal");
  expect_value(std::find(actions.begin(),actions.end(),cut.actual)!=actions.end(),
               "four-dark legal set omitted the recorded move");
  for (const auto move:actions) {
    int left=1800;
    auto next=state;
    step(next,move,rules);
    const auto local=defend(next,flag,owner,depth-1,rules,left);
    const auto score=scorer.score(move);
    std::cout<<"D "<<state.ply+1<<' '<<loss<<' '<<int(state.to_move)<<' '<<owner<<' '<<flag<<' '
             <<depth<<' '<<move_key(move)<<' '<<(move==cut.actual)<<' '<<int(local)<<' '
             <<1801-left<<' '<<score.total()<<' '<<state_hash(state)<<'\n';
  }
}

void analyze(const char* path) {
  const auto body=read_text(path);
  const auto header=parse_json(body.substr(0,body.find('\n')));
  Rules rules;
  std::string error;
  expect_value(parse_rules(dump_json(header.at("rules")),rules,error),"invalid practice rules");
  expect_value(header.at("rules").at("id").str()=="flag-practice-1","incorrect practice rules");
  std::deque<Cut> latest;
  bool first_loss=false;
  replay(path,false,[&](const State& before,const StepInfo& event,const State& after) {
    if (first_loss) return;
    latest.push_back({before,event.move});
    if (latest.size()>13) latest.pop_front();
    const int flag=event.move.to;
    const int owner=before.seat_at[flag];
    if (before.rank_at[flag]!=Flag || after.rank_at[flag]==Flag) return;
    first_loss=true;
    const int loss=static_cast<int>(before.ply)+1;
    std::cout<<"L "<<loss<<' '<<owner<<' '<<flag<<' '<<move_key(event.move)<<'\n';
    for (const auto& cut:latest) {
      if (cut.state.ply>=before.ply || team_of(cut.state.to_move)!=team_of(owner)) continue;
      inspect(cut,flag,owner,loss,rules);
    }
  });
  if (!first_loss) std::cout<<"L 0 -1 -1 -1\n";
}
}

int main(int argc,char** argv) {
  try {
    expect_value(argc==2,"expected a generated four-dark practice replay");
    analyze(argv[1]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
