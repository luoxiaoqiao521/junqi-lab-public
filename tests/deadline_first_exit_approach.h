#pragma once

struct ApproachOffer {
  jq::Move move;
  int target, owner, proved;
};

static std::optional<ApproachOffer> choose_approach(const EarlyPrefix& game,const jq::SeatDecision& d,
    jq::Move ordinary,const jq::Rules& rules,const std::vector<std::pair<jq::Move,int>>& scores) {
  using namespace jq;
  const auto& view=d.observation;
  const auto evidence=game.reply_history.for_actor(view,game.marks[view.viewer]);
  const auto targets=flag_targets(view,evidence);
  if (targets.empty()) return std::nullopt;
  for (auto move:d.actions) {
    if (std::find(targets.begin(),targets.end(),move.to)!=targets.end() &&
        compatible_rank_mask(view,evidence,move.to)==(1u<<Flag) &&
        resolve(view.rank_at(move.from),Flag,rules)!=Outcome::DefenderWins) return std::nullopt;
  }
  const auto quiet_route=[&](Move move) -> std::optional<ApproachOffer> {
    if (view.seat_at(move.to)>=0 || resolve(view.rank_at(move.from),Flag,rules)==Outcome::DefenderWins)
      return std::nullopt;
    const auto projected=project_quiet(view,move);
    std::vector<Move> next;
    gen_piece_moves(projected,view.viewer,move.to,next);
    for (const int target:targets) if (std::find(next.begin(),next.end(),Move{move.to,static_cast<uint8_t>(target)})!=next.end())
      return ApproachOffer{move,target,view.seat_at(target),compatible_rank_mask(view,evidence,target)==(1u<<Flag)};
    return std::nullopt;
  };
  if (quiet_route(ordinary)) return std::nullopt;
  std::optional<ApproachOffer> best;
  int best_score=0;
  for (const auto& [move,score]:scores) {
    const auto route=quiet_route(move);
    if (route && (!best || score>best_score ||
                  (score==best_score && early_action_key(move)<early_action_key(best->move)))) {
      best=route;
      best_score=score;
    }
  }
  return best;
}

struct ApproachFollow {
  int return_ply=0, piece_present=0, legal_attack=0, played_attack=0;
  int target_exit=0, target_reason=0, second_enemy=0;
};

static void approach_tail(EarlyPrefix& game,const ApproachOffer& offer,jq::Move first,int learner,int table,
    int partner,const jq::Rules& rules,const jq::ShortWeights& parent,const jq::ShortWeights& peer) {
  using namespace jq;
  ApproachFollow log;
  const int piece=game.state.id_at[offer.move.from];
  const auto play=[&](Move move) {
    StepInfo event;
    game.play(move,learner,rules,&event);
    if (event.eliminated_mask&(1u<<offer.owner)) {
      log.target_exit=game.state.ply;
      log.target_reason=int(event.exit_reason[offer.owner]);
    }
    if (!log.second_enemy && game.state.out[(learner+1)%kSeats] && game.state.out[(learner+3)%kSeats])
      log.second_enemy=game.state.ply;
  };
  play(first);
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    const auto action=early_normal(game,d,learner,table,rules,parent,peer,partner);
    if (!log.return_ply && d.observation.viewer==learner) {
      log.return_ply=game.state.ply+1;
      log.piece_present=game.state.id_at[offer.move.from]==piece || game.state.id_at[offer.move.to]==piece;
      const auto attacks=[&](Move move){return move.to==offer.target && d.observation.seat_at(move.to)==offer.owner;};
      log.legal_attack=std::any_of(d.actions.begin(),d.actions.end(),attacks);
      log.played_attack=attacks(action);
    }
    play(action);
  }
  std::cout<<' '<<log.return_ply<<' '<<log.piece_present<<' '<<log.legal_attack<<' '<<log.played_attack
           <<' '<<log.target_exit<<' '<<log.target_reason<<' '<<log.second_enemy;
}

static void print_approach_result(const EarlyPrefix& game,int learner) {
  std::cout<<' '<<game.state.ply<<' '<<review_outcome(game.state.result,learner)<<' '<<int(game.ending)
           <<' '<<game.enemy_exit<<' '<<game.friendly_exit<<' '<<game.trace;
}

static void paired_first_exit(uint64_t seed,int learner,int partner,int table,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& peer,bool control) {
  using namespace jq;
  EarlyPrefix game(opening(seed,rules),seed);
  std::cout<<seed<<' '<<learner<<' '<<partner<<' '<<table;
  while (game.state.result==Result::Ongoing) {
    const auto d=seat_decision(game.state);
    const bool inspect=!control && !game.enemy_exit && game.state.ply<140 && d.observation.viewer==learner;
    std::vector<std::pair<Move,int>> scores;
    const auto action=early_normal(game,d,learner,table,rules,parent,peer,partner,inspect?&scores:nullptr);
    const auto offer=inspect?choose_approach(game,d,action,rules,scores):std::nullopt;
    if (offer) {
      std::cout<<' '<<game.state.ply+1<<' '<<game.trace<<' '<<early_action_key(action)
               <<' '<<early_action_key(offer->move)<<' '<<offer->target<<' '<<offer->owner<<' '<<offer->proved;
      auto trial=game;
      approach_tail(game,*offer,action,learner,table,partner,rules,parent,peer);
      print_approach_result(game,learner);
      approach_tail(trial,*offer,offer->move,learner,table,partner,rules,parent,peer);
      print_approach_result(trial,learner);
      std::cout<<'\n';
      return;
    }
    game.play(action,learner,rules);
  }
  std::cout<<" 0";
  print_approach_result(game,learner);
  std::cout<<'\n';
}
