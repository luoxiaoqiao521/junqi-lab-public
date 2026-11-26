#pragma once

struct PostResult {
  int leaf_ply, outcome, ending, friendly_exit, enemy_exit;
  uint64_t trace;
};

static PostResult post_continuation(EarlyPrefix game,const ExchangePolicy& fixed) {
  while (game.state.result==Result::Ongoing) {
    const auto decision=seat_decision(game.state);
    const auto move=finish_policy(game,decision,fixed,true);
    game.play(move.base,fixed.learner,fixed.rules);
    if (move.public_win)
      single_case_check(review_outcome(game.state.result,fixed.learner)==2,"public terminal proof mismatch");
  }
  return {game.state.ply,review_outcome(game.state.result,fixed.learner),int(game.ending),
          game.friendly_exit,game.enemy_exit,game.trace};
}

static void post_labels(EarlyPrefix& game,const SeatDecision& decision,const FinishPolicy& choice,
                        const ExchangePolicy& fixed,uint64_t seed,int context) {
  print_learning_root(std::cout,decision,game.marks[fixed.learner],decision.observation.quiet_clock,
                      seed,fixed.learner,context,game.trace);
  for (int repeat=0;repeat<2;++repeat) for (const auto move:choice.shortlist) {
    auto fork=game;
    fork.streams=review_streams(mix_seed(game.trace,69211+repeat));
    fork.play(move,fixed.learner,fixed.rules);
    const auto result=post_continuation(fork,fixed);
    std::cout<<"A "<<seed<<' '<<fixed.learner<<' '<<context<<' '<<repeat<<' '<<game.trace<<' '
             <<early_action_key(move)<<' '<<result.leaf_ply<<' '<<result.outcome<<' '<<result.ending<<'\n';
  }
}

static void post_result(const EarlyPrefix& game,const ExchangePolicy& fixed,int initial,uint64_t seed,
                        int arm,const std::optional<Move> action,int trigger) {
  auto branch=game;
  if (action) branch.play(*action,fixed.learner,fixed.rules);
  const auto result=post_continuation(branch,fixed);
  std::cout<<"G "<<seed<<' '<<fixed.learner<<' '<<fixed.partner<<' '<<fixed.table<<' '<<arm<<' '
           <<initial<<' '<<trigger<<' '<<(action?early_action_key(*action):kNone)<<' '
           <<result.leaf_ply<<' '<<result.outcome<<' '<<result.ending<<' '<<result.friendly_exit<<' '
           <<result.enemy_exit<<' '<<game.trace<<' '<<result.trace<<'\n';
}

static void post_retirement(uint64_t seed,const ExchangePolicy& fixed,const StudentNetwork* updated) {
  EarlyPrefix game(opening(seed,fixed.rules),seed);
  const int initial=game.state.to_move;
  const int context=fixed.partner*2+fixed.table;
  while (game.state.result==Result::Ongoing) {
    const auto decision=seat_decision(game.state);
    const auto choice=finish_policy(game,decision,fixed,true);
    const bool eligible=game.enemy_exit && decision.observation.viewer==fixed.learner &&
        occupied_seats(decision.observation).teams[1-team_of(fixed.learner)]==1 &&
        !choice.public_win && choice.shortlist.size()>1;
    if (eligible) {
      if (updated) {
        const auto next=student_pick(decision,game.marks[fixed.learner],choice.pressure,
                                     choice.shortlist,*updated,fixed.rules);
        const auto clock=decision.observation.ply<140?next:student_pick(decision,game.marks[fixed.learner],
            choice.pressure,choice.shortlist,*updated,fixed.rules,false);
        const std::array<Move,3> arms{choice.base,next,clock};
        for (int arm=0;arm<int(arms.size());++arm)
          post_result(game,fixed,initial,seed,arm,arms[arm],game.state.ply+1);
      } else {
        post_labels(game,decision,choice,fixed,seed,context);
        std::cout<<"T "<<seed<<' '<<fixed.learner<<' '<<context<<' '<<early_action_key(choice.base)<<'\n';
      }
      return;
    }
    game.play(choice.base,fixed.learner,fixed.rules);
    if (choice.public_win)
      single_case_check(review_outcome(game.state.result,fixed.learner)==2,"opening proof did not win at once");
  }
  if (updated) {
    for (int arm:{0,1,2}) post_result(game,fixed,initial,seed,arm,std::nullopt,0);
  } else {
    std::cout<<"T "<<seed<<' '<<fixed.learner<<' '<<context<<" -1\n";
  }
}
