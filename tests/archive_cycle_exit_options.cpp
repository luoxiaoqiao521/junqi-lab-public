#include "archive_replay_prefix.h"
#include "archive_collision_reference.h"
#include "archive_cycle_flag_condition.h"
#include <iomanip>

using namespace jq;
using namespace jq::archive_probe;

static void matched(const State& state,Move actual,Move back,const Rules& rules,
                    const ShortWeights& weights,const std::string& key,int ordinal,bool collision) {
  expect_value(!(actual==back) && state.seat_at[actual.to]<0 && state.seat_at[back.to]<0,
               "paired comparison requires distinct legal quiet actions");
  for(int variant:{0,2}) for(int sample=0;sample<12;++sample) {
    const uint64_t seed=mix_seed(886101+static_cast<uint64_t>(ordinal),static_cast<uint64_t>(sample));
    const auto rollout=[&](Move move) {
      return collision && variant==2 ? collision_rollout(state,move,rules,weights,seed)
          : review_rollout(state,move,rules,variant,true,weights,seed);
    };
    const auto a=rollout(actual),b=rollout(back);
    if(sample==0)
      expect_value(b==rollout(back),"frozen continuation changed");
    std::cout<<"P "<<key<<' '<<variant<<' '<<sample<<' '
      <<review_outcome(a,state.to_move)<<' '<<review_outcome(b,state.to_move)<<'\n';
  }
}

static bool inspect(const ShortWeights& weights,bool paired,int ordinal,bool collision,bool condition) {
  std::string key;
  int count,ccw,x,y,xx,yy;
  if(!(std::cin>>key>>count>>ccw>>x>>y>>xx>>yy)||count<1||count>10000) return false;
  int8_t layouts[4][25]{};
  if(!read_layouts(layouts)) return false;
  std::vector<Row> rows(static_cast<size_t>(count));
  for(auto& row:rows)
    if(!(std::cin>>row.x>>row.y>>row.xx>>row.yy>>row.result)) return false;
  Rules rules;
  rules.ccw=ccw!=0;
  rules.check_involved_no_move=true;
  rules.marshal_reveals_flag=true;
  rules.no_progress_limit=71;
  State state{};
  MovedSquares moved{};
  Move actual;
  if(!archive_prefix(state,moved,actual,rules,layouts,rows,!condition)) {
    std::cout<<"X "<<key<<'\n';
    return true;
  }
  const auto d=seat_decision(state);
  const int from=point(x,y),to=point(xx,yy);
  if(from<0||to<0) return false;
  const Move back{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
  if (condition) {
    print_flag_condition(state,d,back,rules,weights,key,ordinal);
    return true;
  }
  const auto found=std::find(d.actions.begin(),d.actions.end(),back);
  const bool available=found!=d.actions.end();
  const auto chosen=std::find(d.actions.begin(),d.actions.end(),actual);
  if(chosen==d.actions.end()) return false;
  const auto probabilities=review_learned(d,weights);
  const auto heuristic=review_reference(d);
  const auto at=static_cast<size_t>(chosen-d.actions.begin());
  const auto bt=static_cast<size_t>(found-d.actions.begin());
  const bool alias=available && !(back==actual) && short_features(d.observation,actual)==short_features(d.observation,back);
  std::cout<<"D "<<key<<' '<<state.ply<<' '<<int(state.to_move)<<' '<<d.actions.size()
    <<' '<<available<<' '<<(available && d.observation.seat_at(back.to)<0)<<' '<<alias
    <<' '<<probabilities[at]<<' '<<(available?probabilities[bt]:-1.)
    <<' '<<heuristic[at]<<' '<<(available?heuristic[bt]:-1.)<<'\n';
  if(paired) {
    if(!available) return false;
    matched(state,actual,back,rules,weights,key,ordinal,collision);
  }
  return true;
}

int main(int argc,char** argv) {
  const bool collision=argc==2 && std::string(argv[1])=="collision-pairs";
  const bool condition=argc==2 && std::string(argv[1])=="public-condition";
  const bool paired=collision || (argc==2 && std::string(argv[1])=="pairs");
  if(argc!=1 && !paired && !condition) return 2;
  if(collision) verify_collision_objective();
  if(condition) verify_flag_condition();
  int count;
  ShortWeights weights{};
  if(!(std::cin>>count)||count<1||count>2000) return 2;
  for(auto& weight:weights) if(!(std::cin>>weight)||!std::isfinite(weight)) return 2;
  std::cout<<std::setprecision(17);
  for(int i=0;i<count;++i) if(!inspect(weights,paired,i,collision,condition)) return 3;
  return 0;
}
