#pragma once

static void export_conversion_history(uint64_t seed,const ExchangePolicy& p,bool early) {
  ConversionJournal journal;
  const auto prefix=conversion_prefix(seed,p,early,&journal);
  std::cout<<seed<<' '<<p.learner<<' '<<p.partner<<' '<<p.table<<' '<<early<<' '<<prefix.early_at
           <<' '<<prefix.early_move<<' '<<conversion_checkpoint(prefix.game);
  for (const auto hash:journal.views) std::cout<<' '<<hash;
  std::cout<<' '<<journal.actions.size();
  for (const int action:journal.actions) std::cout<<' '<<action;
  std::cout<<'\n';
}

static void restore_conversion_history(uint64_t seed,const ExchangePolicy& p,bool early) {
  uint64_t layout,checkpoint;
  int learner,partner,table,arm,n;
  std::array<uint64_t,kSeats> views{};
  ConversionPrefix prefix{EarlyPrefix(opening(seed,p.rules),seed)};
  single_case_check(bool(std::cin>>layout>>learner>>partner>>table>>arm>>prefix.early_at>>prefix.early_move>>checkpoint),
                    "invalid conversion journal header");
  single_case_check(layout==seed && learner==p.learner && partner==p.partner && table==p.table && arm==int(early),
                    "invalid conversion journal identity");
  for (auto& hash:views) single_case_check(bool(std::cin>>hash),"missing per-seat observed history");
  single_case_check(bool(std::cin>>n) && n>0 && n<=p.rules.truncation_cap &&
                    prefix.early_at>=0 && prefix.early_at<=n,"invalid conversion journal length");
  ConversionJournal history;
  auto& game=prefix.game;
  for (int i=0;i<n;++i) {
    int action;
    single_case_check(bool(std::cin>>action) && action>=0 && action<kNodes*kNodes && game.state.result==Result::Ongoing,
                      "invalid conversion journal move");
    const auto d=seat_decision(game.state);
    const bool changed=early && i+1==prefix.early_at;
    std::vector<std::pair<Move,int>> scores;
    auto expected=early_normal(game,d,p.learner,p.table,p.rules,p.parent,p.history,p.partner,changed?&scores:nullptr);
    if (changed) {
      const auto offer=choose_approach(game,d,expected,p.rules,scores);
      single_case_check(offer.has_value() && d.observation.viewer==p.learner && !game.enemy_exit,
                        "invalid conversion journal intervention");
      expected=offer->move;
    }
    single_case_check(action==early_action_key(expected) && (i+1!=prefix.early_at || action==prefix.early_move),
                      "conversion journal move disagrees with the locked policy");
    history.play(game,expected,p);
  }
  single_case_check((game.state.result!=Result::Ongoing || (game.enemy_exit && game.state.to_move==p.learner)) &&
                    history.views==views && conversion_checkpoint(game)==checkpoint,"conversion journal checkpoint mismatched");
  print_conversion(seed,p,early,prefix);
}
