#pragma once

struct FirstExitCoverage {
  int visits=0;
  std::array<int,3> legal{}, listed{}, selected{};
  std::array<int,3> legal_alternatives{}, listed_alternatives{};
  int geometry_filtered=0, geometry_below=0, geometry_tied=0;
  int returned=0, lost_piece=0, route_remains=0, followed=0, captured=0;
};

struct FirstExitPreparation {
  int piece, node;
  std::vector<int> flags;
};

static bool audit_preparation(const FirstExitPreparation& prior,FirstExitCoverage& count,
    const EarlyPrefix& game,const jq::SeatDecision& d,const jq::OpeningMarks& evidence,jq::Move choice) {
  ++count.returned;
  if (game.state.id_at[prior.node]!=prior.piece) {++count.lost_piece;return false;}
  const auto applies=[&](jq::Move move) {
    return move.from==prior.node && std::find(prior.flags.begin(),prior.flags.end(),move.to)!=prior.flags.end() &&
           d.observation.seat_at(move.to)>=0 && (compatible_rank_mask(d.observation,evidence,move.to)&(1u<<jq::Flag));
  };
  count.route_remains+=std::any_of(d.actions.begin(),d.actions.end(),applies);
  const bool follows=applies(choice);
  count.followed+=follows;
  return follows;
}

static bool count_first_exit_choices(FirstExitCoverage& count,const EarlyPrefix& game,
    std::optional<FirstExitPreparation>& preparation,
    const jq::SeatDecision& d,jq::Move choice,const jq::Rules& rules,
    const std::vector<std::pair<jq::Move,int>>& scores) {
  using namespace jq;
  ++count.visits;
  const auto& view=d.observation;
  const auto& actor=game.marks[view.viewer];
  const auto evidence=game.reply_history.for_actor(view,actor);
  const bool follows=preparation && audit_preparation(*preparation,count,game,d,evidence,choice);
  preparation.reset();
  const auto targets=flag_targets(view,evidence);
  if (targets.empty()) return follows;
  const auto& board=Board::get();
  const auto shortlist=response_candidates(d,evidence,choice,rules,public_rollout_seed(view,actor,view.quiet_clock));
  std::array<bool,3> legal{}, listed{}, selected{};
  const auto score_for=[&](Move action) {
    return std::find_if(scores.begin(),scores.end(),[&](const auto& item) {return action==item.first;});
  };
  const auto chosen_score=score_for(choice);
  const auto account=[&](size_t category,Move move) {
    legal[category]=true;
    const bool nominated=std::find(shortlist.begin(),shortlist.end(),move)!=shortlist.end();
    listed[category]|=nominated;
    selected[category]|=move==choice;
    if (move==choice) return;
    ++count.legal_alternatives[category];
    count.listed_alternatives[category]+=nominated;
    if (category!=2) return;
    const auto score=score_for(move);
    if (score==scores.end()) {++count.geometry_filtered;return;}
    single_case_check(chosen_score!=scores.end() && score->second<=chosen_score->second,
                      "natural choice differed from its recorded scores");
    count.geometry_below+=score->second<chosen_score->second;
    count.geometry_tied+=score->second==chosen_score->second;
  };
  for (auto action:d.actions) {
    if (std::find(targets.begin(),targets.end(),action.to)!=targets.end()) {
      if (resolve(view.rank_at(action.from),Flag,rules)==Outcome::DefenderWins) continue;
      account(1,action);
      if (compatible_rank_mask(view,evidence,action.to)!=(1u<<Flag)) continue;
      auto proof=game.state;
      const int owner=view.seat_at(action.to);
      const auto event=step(proof,action,rules);
      single_case_check((event.eliminated_mask&(1u<<owner)) && event.exit_reason[owner]==ExitReason::FlagRemoved,
                        "public flag proof did not cause retirement");
      account(0,action);
    }
    if (view.seat_at(action.to)>=0 || !can_leave_node(view.rank_at(action.from),action.to) ||
        resolve(view.rank_at(action.from),Flag,rules)==Outcome::DefenderWins) continue;
    const auto& road=board.road(action.to);
    const auto adjacent=[&](int flag) {
      return std::find(road.begin(),road.end(),flag)!=road.end();
    };
    if (std::any_of(targets.begin(),targets.end(),adjacent)) {
      account(2,action);
      if (action==choice) {
        preparation=FirstExitPreparation{game.state.id_at[action.from],action.to,{}};
        std::copy_if(targets.begin(),targets.end(),std::back_inserter(preparation->flags),adjacent);
      }
    }
  }
  for (size_t category=0;category<legal.size();++category) {
    count.legal[category]+=legal[category]; count.listed[category]+=listed[category]; count.selected[category]+=selected[category];
  }
  return follows;
}

static void audit_first_exit(uint64_t seed,int seat,int partner,int table,const jq::Rules& rules,
    const jq::ShortWeights& parent,const jq::ShortWeights& peer) {
  using namespace jq;
  EarlyPrefix game(opening(seed,rules),seed);
  FirstExitCoverage count;
  std::optional<FirstExitPreparation> preparation;
  while (game.state.result==Result::Ongoing && game.state.ply<140) {
    const auto d=seat_decision(game.state);
    const bool inspect=!game.enemy_exit && d.observation.viewer==seat;
    std::vector<std::pair<Move,int>> scores;
    const auto action=early_normal(game,d,seat,table,rules,parent,peer,partner,inspect?&scores:nullptr);
    const bool follows=inspect && count_first_exit_choices(count,game,preparation,d,action,rules,scores);
    game.play(action,seat,rules);
    if (follows && game.enemy_exit==game.state.ply && game.enemy_reason==int(ExitReason::FlagRemoved)) ++count.captured;
  }
  std::cout<<seed<<' '<<seat<<' '<<partner<<' '<<table<<' '<<game.state.ply<<' '
           <<game.enemy_exit<<' '<<game.friendly_exit<<' '<<review_outcome(game.state.result,seat)<<' '
           <<int(game.ending)<<' '<<count.visits;
  for (size_t category=0;category<count.legal.size();++category)
    std::cout<<' '<<count.legal[category]<<' '<<count.listed[category]<<' '<<count.selected[category];
  std::cout<<' '<<count.returned<<' '<<count.lost_piece<<' '<<count.route_remains<<' '
           <<count.followed<<' '<<count.captured;
  for (size_t category=0;category<count.legal.size();++category)
    std::cout<<' '<<count.legal_alternatives[category]<<' '<<count.listed_alternatives[category];
  std::cout<<' '<<count.geometry_filtered<<' '<<count.geometry_below<<' '<<count.geometry_tied<<'\n';
}
