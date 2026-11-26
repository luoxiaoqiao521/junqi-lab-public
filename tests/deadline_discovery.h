#pragma once
#include "deadline_critical_reply.h"
#include "deadline_counterfactual.h"

struct DiscoveryWorld {
  jq::State state;
  uint64_t identity;
};

enum class DiscoveryProbe { Original, Replan, Heldout, Counterfactual, Calibration };

static std::vector<DiscoveryWorld> discovery_worlds(const jq::SeatDecision& root,
    const jq::OpeningMarks& evidence,uint64_t seed,size_t count,std::vector<uint64_t>& used) {
  using namespace jq;
  std::vector<DiscoveryWorld> worlds;
  for (uint64_t attempt=0;attempt<96 && worlds.size()<count;++attempt) {
    const auto sample=public_sample(root.observation,evidence,mix_seed(seed,attempt));
    if (!sample || seat_decision(*sample).actions!=root.actions) continue;
    const auto identity=fnv1a(sample->rank_at,sizeof(sample->rank_at));
    if (std::find(used.begin(),used.end(),identity)!=used.end()) continue;
    used.push_back(identity);
    worlds.push_back({*sample,identity});
  }
  return worlds;
}

static void discovery_confirm(const std::vector<DiscoveryWorld>& worlds,const std::vector<jq::Move>& actions,
    const jq::Rules& rules,const jq::ShortWeights& parent,uint64_t seed,
    const jq::OpeningMarks& evidence,const jq::PublicReplyHistory& history,DiscoveryProbe probe) {
  using namespace jq;
  for (size_t i=0;i<worlds.size();++i) for (int table:{1,2}) {
    const auto& state=worlds[i].state;
    const auto stream=mix_seed(seed,i*2+table);
    std::vector<std::pair<CriticalReply,CriticalReply>> replies;
    std::cout<<"P "<<worlds[i].identity<<' '<<table<<' '<<stream;
    for (size_t arm=0;arm<actions.size();++arm) {
      int elapsed=0;
      const auto result=review_rollout(state,actions[arm],rules,table,true,parent,stream,&elapsed);
      if (i==0 && arm==0) {
        int repeat=0;
        const auto same=review_rollout(state,actions[arm],rules,table,true,parent,stream,&repeat);
        single_case_check(same==result && repeat==elapsed,"A/A replay differed");
      }
      std::cout<<' '<<review_outcome(result,state.to_move)<<' '<<elapsed;
      if (probe==DiscoveryProbe::Replan || probe==DiscoveryProbe::Heldout) {
        const auto frozen=critical_reply_rollout(state,actions[arm],rules,table,parent,stream,evidence,history,false);
        single_case_check(frozen.outcome==review_outcome(result,state.to_move) && frozen.elapsed==elapsed,
                          "cutoff diagnostics changed the frozen continuation");
        replies.emplace_back(frozen,critical_reply_rollout(state,actions[arm],rules,table,parent,stream,evidence,history,true));
      }
    }
    std::cout<<'\n';
    for (size_t arm=0;arm<replies.size();++arm) {
      const auto& [old,current]=replies[arm];
      std::cout<<"C "<<worlds[i].identity<<' '<<table<<' '<<arm;
      for (const auto& item:{old,current})
        std::cout<<' '<<item.outcome<<' '<<item.elapsed<<' '<<item.changes<<' '<<item.first_change
                 <<' '<<item.flag_choice<<' '<<item.flag_guard<<' '<<item.cutoff_flag
                 <<' '<<item.flag_taken<<' '<<item.cutoff_contact<<' '<<item.contact_taken
                 <<' '<<item.cutoff_value;
      std::cout<<'\n';
    }
    if (probe==DiscoveryProbe::Counterfactual || probe==DiscoveryProbe::Calibration)
      for (size_t arm=0;arm<actions.size();++arm)
        test_enemy_reply(state,actions[arm],rules,table,parent,stream,evidence,history,worlds[i].identity,arm,
                         probe==DiscoveryProbe::Calibration);
  }
}

static void discovery_root(const EarlyPrefix& game,const jq::SeatDecision& root,jq::Move baseline,
    const jq::Rules& rules,const jq::ShortWeights& parent,const jq::ShortWeights& peer,uint64_t seed,
    DiscoveryProbe probe) {
  using namespace jq;
  const auto& view=root.observation;
  const auto evidence=game.reply_history.for_actor(view,game.marks[view.viewer]);
  const auto key=public_rollout_seed(view,evidence,view.quiet_clock);
  const auto shortlist=response_candidates(root,evidence,baseline,rules,key);
  std::cout<<"R "<<seed<<' '<<view.ply<<' '<<view.viewer<<' '<<root.actions.size()<<' '
           <<shortlist.size()<<' '<<early_action_key(baseline)<<'\n';
  if (public_flag_finish(root,baseline,rules)) {std::cout<<"X terminal\n";return;}
  if (shortlist.size()==root.actions.size()) {std::cout<<"X no-outside\n";return;}
  std::vector<uint64_t> used;
  const auto screen=discovery_worlds(root,evidence,mix_seed(key,301),4,used);
  const bool heldout=probe==DiscoveryProbe::Heldout;
  const auto confirm=discovery_worlds(root,evidence,mix_seed(key,heldout?2701:701),8,used);
  if (screen.size()!=4 || confirm.size()!=8) {std::cout<<"X sample "<<screen.size()<<' '<<confirm.size()<<'\n';return;}
  for (const auto& world:screen) std::cout<<"W "<<world.identity<<'\n';
  std::vector<Move> arms{baseline,baseline,baseline};
  double best[2]{-1e20,-1e20};
  std::vector<Move> outside;
  for (auto action:root.actions) {
    ResponseValue score;
    for (size_t i=0;i<screen.size();++i)
      response_trial(screen[i].state,action,evidence,game.reply_history,rules,peer,int(i%2),
                     mix_seed(mix_seed(key,1301),i),score,false,RoutePriority::Joint,true);
    const bool listed=std::find(shortlist.begin(),shortlist.end(),action)!=shortlist.end();
    if (!listed) outside.push_back(action);
    const int slot=listed?0:1;
    if (score.value>best[slot] || (score.value==best[slot] && early_action_key(action)<early_action_key(arms[slot+1]))) {
      best[slot]=score.value;
      arms[slot+1]=action;
    }
    std::cout<<"S "<<early_action_key(action)<<' '<<listed<<' '<<score.value<<' '
             <<score.wins<<' '<<score.draws<<' '<<score.losses<<' '<<score.unknown<<'\n';
  }
  outside.erase(std::find(outside.begin(),outside.end(),arms[2]));
  if (!outside.empty()) arms.push_back(outside[Rng(mix_seed(key,1601)).below(static_cast<uint32_t>(outside.size()))]);
  std::cout<<"N";
  for (auto action:arms) std::cout<<' '<<early_action_key(action);
  std::cout<<'\n';
  discovery_confirm(confirm,arms,rules,parent,mix_seed(key,heldout?3701:1701),evidence,game.reply_history,probe);
}

static void discover_public_actions(uint64_t seed,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& peer,DiscoveryProbe probe) {
  using namespace jq;
  EarlyPrefix game(opening(seed,rules),seed);
  const int learner=static_cast<int>(seed%kSeats);
  for (int ply:{40,88}) {
    while (game.state.result==Result::Ongoing && game.state.ply<ply) {
      const auto d=seat_decision(game.state);
      game.play(early_normal(game,d,learner,1,rules,parent,peer),learner,rules);
    }
    if (game.state.result!=Result::Ongoing) {
      std::cout<<"R "<<seed<<' '<<ply<<" -1 0 0 -1\nX game-ended\n";
      continue;
    }
    const auto d=seat_decision(game.state);
    auto copy=game;
    const auto baseline=early_normal(copy,d,d.observation.viewer,1,rules,parent,peer);
    discovery_root(game,d,baseline,rules,parent,peer,seed,probe);
  }
}
