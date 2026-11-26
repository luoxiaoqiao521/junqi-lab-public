#include "archive_replay_prefix.h"
#include "public_defender_probe.h"

using namespace jq;
using namespace jq::archive_probe;

static std::array<OpeningMarks,3> rank_views(const State& state,const DefenderEvidence& evidence) {
  const int seat=state.to_move;
  std::array<OpeningMarks,3> result{*evidence.path[seat],*evidence.contact[seat],*evidence.contact[seat]};
  const auto view=observe(state,seat);
  // Oracle ceiling test. Only the standalone experiment receives archive truth.
  for(int n=0;n<kNodes;++n) if(view.seat_at(n)>=0 && view.rank_at(n)==Unknown)
    result[2].ranks[n]=1u<<state.rank_at[n];
  return result;
}

static bool danger_after(State state,Move action,const Rules& rules) {
  const int viewer=state.to_move;
  step(state,action,rules);
  return next_enemy_flag_attack(state,viewer)!=0;
}

static void compare(const std::string& group,const State& state,const Rules& rules,
                    const ShortWeights& weights,const std::array<Move,4>& options) {
  if(std::all_of(options.begin(),options.end(),[&](Move m){return m==options.front();})) return;
  Rules trial=rules;
  trial.truncation_cap=state.ply+240;
  for(int i=0;i<12;++i) {
    const auto streams=review_streams(mix_seed(307771+state.ply,i));
    std::array<int,4> score{};
    for(size_t j=0;j<options.size();++j) {
      const auto old=std::find(options.begin(),options.begin()+j,options[j]);
      if(old!=options.begin()+j) score[j]=score[old-options.begin()];
      else {
        const int result=finish_defense(state,options[j],trial,weights,streams,(state.to_move+1)%4);
        score[j]=result<0 ? result : 2-result;
      }
    }
    std::cout<<"P "<<group<<' '<<i;
    for(int v:score) std::cout<<' '<<v;
    std::cout<<'\n';
  }
}

static void inspect(const std::string& group,const State& state,const Rules& rules,
                    const ShortWeights& weights,const DefenderEvidence& book) {
  const auto d=seat_decision(state);
  const auto reference=review_reference(d);
  const auto fallback=d.actions[std::max_element(reference.begin(),reference.end())-reference.begin()];
  const auto marked=rank_views(state,book);
  std::array<PublicThreat,3> threat{};
  std::array<Move,4> actions{fallback,fallback,fallback,fallback};
  for(size_t i=0;i<marked.size();++i) {
    threat[i]=visible_flag_threat(d.observation,marked[i],rules);
    actions[i+1]=visible_intercept(d,marked[i],rules,fallback,threat[i]).value_or(fallback);
  }
  std::cout<<"D "<<group<<' '<<state.ply+1<<' '<<danger_after(state,fallback,rules);
  for(size_t i=0;i<threat.size();++i)
    std::cout<<' '<<threat[i].origins<<' '<<threat[i].forced<<' '<<!(actions[i+1]==fallback)
             <<' '<<danger_after(state,actions[i+1],rules);
  std::cout<<' '<<!(actions[1]==actions[2])<<' '<<!(actions[2]==actions[3])<<'\n';
  compare(group,state,rules,weights,actions);
}

int main() {
  try {
    verify_threat_fixture();
    int count;
    ShortWeights weights{};
    if(!(std::cin>>count)||count<1) return 2;
    for(size_t j=0;j<5;++j) if(!(std::cin>>weights[j])||!std::isfinite(weights[j])) return 2;
    Rules rules;
    rules.ccw=false; rules.marshal_reveals_flag=true;
    rules.no_progress_limit=71; rules.check_involved_no_move=true;
    for(int i=0;i<count;++i) {
      std::string group;
      int size;
      if(!(std::cin>>group>>size)||size<1||size>2000) return 2;
      int8_t layouts[4][25]{};
      if(!read_layouts(layouts)) return 2;
      std::vector<Row> rows(static_cast<size_t>(size));
      for(auto& row:rows) if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return 2;
      State state{};
      MovedSquares moved{};
      Move selected;
      DefenderEvidence evidence;
      const ReplayObserver track=[&](const State& before,const StepInfo& event,const State& after) {
        evidence.track(before,event,after,rules);
      };
      if(!archive_prefix(state,moved,selected,rules,layouts,rows,true,track)) {
        std::cout<<"X "<<group<<'\n';
        continue;
      }
      inspect(group,state,rules,weights,evidence);
    }
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 3;
  }
}
