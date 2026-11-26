#pragma once

static unsigned partner_hq_loss(const Rules& chosen,int target,bool active_mover) {
  constexpr Move attack{53,58};
  auto state=single_case_state();
  state.to_move=2;
  state.out[2]=false;
  place_single_case_piece(state,1,4,Engineer);
  place_single_case_piece(state,51,53,Marshal);
  place_single_case_piece(state,52,64,Bomb);
  if (target==Flag) {
    place_single_case_piece(state,25,58,Flag);
    place_single_case_piece(state,26,56,Corps);
  } else {
    place_single_case_piece(state,26,58,target);
  }
  if (active_mover) place_single_case_piece(state,27,50,Engineer);
  auto rules=chosen;
  rules.no_progress_limit=0;
  rules.marshal_reveals_flag=false;
  EarlyPrefix game(state,0);
  DarkOpeningPath tracked(state,0);
  PublicReplyHistory raw(state,true);
  single_case_check(is_legal(state,attack),"partner HQ case must use a legal attack");
  StepInfo event;
  single_case_check(game.play(attack,0,rules,&event)==Outcome::AttackerWins,"partner HQ collision changed");
  tracked.play(attack,0,rules);
  raw.advance(state,game.state,event,rules);
  const auto loss=game.marks[0].removed[1].masks[0];
  single_case_check(loss==game.reply_history.marks().removed[1].masks[0] &&
                    loss==tracked.game.reply_history.marks().removed[1].masks[0] &&
                    !game.marks[0].removed[0].size && !game.marks[0].removed[2].size,
                    "actor and referee-prefix partner collision ledgers disagree");
  single_case_check((raw.marks().removed[1].masks[0]&(1u<<Flag))!=0,
                    "counterexample could not reproduce outcome-only ambiguity");
  const auto settled=settled_flag_capture(observe(state,kNone),event,rules);
  single_case_check(!settled || *settled==(target==Flag),"public flag settlement reason is wrong");
  if (!active_mover && target!=Flag)
    single_case_check(event.exit_reason[1]==ExitReason::MovablePiecesExhausted,
                      "other exit reason must be exercised without assuming flag capture");
  return loss;
}

static void dark_partner_hq_fixtures(Rules rules) {
  rules.flag_eliminates=true;
  rules.eliminate_exhausted_movers=true;
  const auto other=partner_hq_loss(rules,Corps,true);
  single_case_check(other==partner_hq_loss(rules,Division,true) && !(other&(1u<<Flag)),
                    "private reserve ranks changed an observer's nonflag casualty");
  single_case_check(partner_hq_loss(rules,Corps,false)==other,"exhaustion was treated as flag removal");
  single_case_check(partner_hq_loss(rules,Flag,true)==(1u<<Flag),"confirmed flag was not debited");
  rules.flag_eliminates=false;
  single_case_check(partner_hq_loss(rules,Corps,true)&(1u<<Flag),
                    "nonretirement variant inferred an unseen flag from missing settlement");
}

static int partner_possible_positions(const Observation& view,const OpeningMarks& evidence,int partner) {
  int count=0;
  for (int n=0;n<kNodes;++n)
    if (view.seat_at(n)==partner && can_leave_node(Engineer,n) &&
        (compatible_rank_mask(view,evidence,n)&(1u<<Engineer))) ++count;
  return count;
}

static int partner_stock_audit(const ReplanRoot& root,const SeatDecision& d,const OpeningMarks& evidence,
                               const Rules& rules,bool control) {
  const int viewer=d.observation.viewer,partner=(viewer+2)%kSeats;
  const auto& view=d.observation;
  const auto& state=root.ordinary.state;
  const auto strict=root.path.game.reply_history.for_actor(view,root.path.game.marks[viewer]);
  for (int seat=0;seat<kSeats;++seat) for (int rank:{Engineer,Bomb,Corps,Marshal}) {
    const auto stock=rank_stock(view,evidence,seat,rank);
    const auto joint=rank_stock(view,strict,seat,rank);
    int live=0;
    for (int node=0;node<kNodes;++node)
      live+=view.seat_at(node)==seat && state.rank_at[node]==rank;
    single_case_check(stock.on_board.minimum<=live && live<=stock.on_board.maximum &&
                      joint.on_board.minimum<=live && live<=joint.on_board.maximum,
                      "public inventory excluded a physically present resource");
    if (seat==partner)
      std::cout<<"S "<<rank<<' '<<stock.on_board.minimum<<' '<<stock.on_board.maximum<<' '
               <<joint.on_board.minimum<<' '<<joint.on_board.maximum<<' '
               <<joint.collision_lost.minimum<<' '<<joint.collision_lost.maximum<<'\n';
  }
  if (!control) return 0;
  const auto seed=public_rollout_seed(view,evidence,view.quiet_clock);
  const auto worlds=assemble_replan_worlds(root.path,d,rules,seed,2);
  if (worlds.games.empty()) return 0;
  const auto& other=worlds.games.back();
  const auto alt=seat_decision(other.state);
  const auto marks=other.reply_history.for_actor(alt.observation,other.marks[viewer]);
  single_case_check(same_observation(view,alt.observation) && d.actions==alt.actions &&
                    equal_dark_marks(evidence,marks),"partner evidence changed with unobserved identity");
  return 1;
}

static void partner_probe_actions(const SeatDecision& d,const OpeningMarks& evidence,const Rules& rules,
                                  const StudentNetwork& first,const StudentNetwork& second,
                                  Move fallback,const std::vector<Move>& list) {
  const auto& v=d.observation;
  for (const auto move:d.actions) {
    const auto x=student_features(d,evidence,move,&rules);
    const auto rank=compatible_rank_mask(v,evidence,move.to);
    const auto outcome=rank?collision_outcome_mask(v.rank_at(move.from),rank,rules):0;
    const int known = outcome==outcome_bit(Outcome::AttackerWins)?1:
                      outcome==outcome_bit(Outcome::DefenderWins)?0:-1;
    const bool shortlisted=std::find(list.begin(),list.end(),move)!=list.end() || move==fallback;
    std::cout<<"A "<<early_action_key(move)<<' '<<known<<' '<<shortlisted<<' '
             <<std::setprecision(17)<<first.score(x,false)<<' '<<second.score(x,false);
    for (double value:x) std::cout<<' '<<value;
    std::cout<<'\n';
  }
}

static void run_dark_partner_evidence(const Rules& rules) {
  dark_partner_hq_fixtures(rules);
  StudentNetwork frozen, comparison;
  single_case_check(frozen.load(std::cin,true) && comparison.load(std::cin,true),
                    "partner course requires the current 44-feature models");
  uint64_t seed;
  int viewer,table,cut;
  while (std::cin>>seed>>viewer>>table>>cut) {
    single_case_check(viewer>=0 && viewer<4 && (table==0 || table==1) && cut>=0 && cut<=160,
                      "invalid partner evidence assignment");
    const OrdinaryDarkPeers peers{nullptr,false};
    const auto root=restore_replan_root(seed,viewer,table,cut,-1,rules,frozen,peers,true);
    const auto& game=root.ordinary;
    const bool valid=game.state.to_move==viewer && game.state.result==Result::Ongoing && root.anchor.has_value();
    std::cout<<"F "<<seed<<' '<<viewer<<' '<<table<<' '<<game.state.ply<<' '<<valid<<'\n';
    if (!valid) continue;
    const auto d=seat_decision(game.state);
    const auto e=game.reply_history.for_actor(d.observation,game.marks[viewer]);
    const int isolation=partner_stock_audit(root,d,e,rules,seed%16==0);
    auto stream=root.streams[viewer];
    const auto fallback=ordinary_dark_move(game,d,rules,OrdinaryDarkPolicy::Tactical,stream);
    const auto list=budget_candidates(d,fallback,rules,4);
    const auto pick=[&](const StudentNetwork& model) {
      return early_action_key(student_pick(d,e,fallback,list,model,rules,false));
    };
    std::cout<<"C "<<isolation<<' '<<early_action_key(fallback)<<' '<<pick(frozen)<<' '<<pick(comparison)
             <<' '<<partner_possible_positions(d.observation,e,(viewer+2)%kSeats)<<'\n';
    partner_probe_actions(d,e,rules,frozen,comparison,fallback,list);
  }
  single_case_check(std::cin.eof(),"incomplete partner evidence assignment");
}
