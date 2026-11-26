#pragma once

struct RelayAccess {
  Move move;
  int own_turns;
};

static std::optional<RelayAccess> relay_flag_access(const State& state,int player,int flag,
                                                   const Rules& rules,std::optional<Move> preferred={}) {
  if (state.out[player] || state.rank_at[flag]!=Flag || team_of(state.seat_at[flag])==team_of(player)) return {};
  std::vector<Move> moves;
  gen_moves(StateView{state},player,moves);
  const auto& gates=Board::get().road(flag);
  std::optional<RelayAccess> best;
  for (const auto move:moves) {
    const int rank=state.rank_at[move.from];
    if (resolve(rank,Flag,rules)==Outcome::DefenderWins) continue;
    const bool approach=state.seat_at[move.to]==kNone && can_leave_node(rank,move.to) &&
                        std::find(gates.begin(),gates.end(),move.to)!=gates.end();
    const int turns=move.to==flag?1:approach?2:0;
    if (turns && (!best || turns<best->own_turns ||
                  (turns==best->own_turns && preferred==move))) best=RelayAccess{move,turns};
  }
  return best;
}

struct RelaySetupScan {
  std::string key;
  uint64_t seed=0;
  int examined=0, flag=kNone, partner=kNone, observed=kNone, own_turns=0;
  int spoiler=kNone, spoil_action=kNone, spoil_ply=0;
  std::optional<State> preparation, conversion;
  Move prep_human{}, finish_take{};

  void inspect_enemy_reply(const State& root,Move human,const Rules& rules) {
    if (spoiler!=kNone || team_of(root.to_move)==team_of(partner)) return;
    const auto before=relay_flag_access(root,partner,flag,rules);
    if (!before || before->own_turns>own_turns) return;
    auto post=root;
    step(post,human,rules);
    const auto after=relay_flag_access(post,partner,flag,rules);
    if (after && after->own_turns<=own_turns && post.result==Result::Ongoing) return;
    spoiler=root.to_move;
    spoil_action=early_action_key(human);
    spoil_ply=post.ply;
  }

  void inspect(const State& root,Move human,const Rules& rules) {
    if (preparation) {
      if (observed!=kNone) return;
      if (root.out[partner] || root.rank_at[flag]!=Flag) { observed=4; return; }
      if (root.to_move!=partner) { inspect_enemy_reply(root,human,rules); return; }
      conversion=root;
      if (const auto take=relay_flag_access(root,partner,flag,rules,human);
          take && take->own_turns<=own_turns) {
        finish_take=take->move;
        observed=human==finish_take?1:2;
      } else observed=3;
      return;
    }
    if (root.ply<24 || std::any_of(std::begin(root.out),std::end(root.out),[](bool out){return out;})) return;
    const int actor=root.to_move, ally=(actor+2)%4;
    ++examined;
    auto post=root;
    step(post,human,rules);
    if (post.result!=Result::Ongoing || std::any_of(std::begin(post.out),std::end(post.out),[](bool out){return out;})) return;
    for (int target=0;target<kNodes;++target) {
      if (root.rank_at[target]!=Flag || team_of(root.seat_at[target])==team_of(ally)) continue;
      const auto after=relay_flag_access(post,ally,target,rules);
      if (!after) continue;
      const auto before=relay_flag_access(root,ally,target,rules);
      if (before && before->own_turns<=after->own_turns) continue;
      preparation=root;
      prep_human=human;
      flag=target;
      partner=ally;
      own_turns=after->own_turns;
      return;
    }
  }

  void report(const Rules& rules,int (&left)[2]) const {
    if (!preparation) return;
    const auto emit=[&](const State& root,Move candidate,int phase) {
      Rng rng(42);
      const auto portfolio=route_baseline(root,rules,rng);
      const bool test=!(portfolio==candidate) && left[phase]>0;
      std::cout<<"P "<<key<<' '<<phase<<' '<<root.ply+1<<' '<<root.to_move+0<<' '
               <<flag<<' '<<own_turns<<' '<<observed<<' '<<early_action_key(portfolio)<<' '
               <<early_action_key(candidate)<<' '<<test<<' '<<spoil_ply<<' '<<spoiler<<' '<<spoil_action<<'\n';
      if (!test) return;
      --left[phase];
      for (int table:{0,1}) for (int stream=0;stream<4;++stream) {
        const uint64_t paired=mix_seed(seed,stream);
        route_continuation(root,portfolio,rules,root.to_move,table,paired,key,root.ply+1,0,stream);
        route_continuation(root,candidate,rules,root.to_move,table,paired,key,root.ply+1,1,stream);
      }
    };
    emit(*preparation,prep_human,0);
    if (conversion && (observed==1 || observed==2)) emit(*conversion,finish_take,1);
  }
};

static void run_relay_setup(const char* rules_path,const Rules& rules) {
  std::string tag,key,tape,error;
  const auto config=read_text(rules_path);
  int ccw,cap,remaining[2]{6,6};
  uint64_t seed;
  while (std::cin>>tag>>key>>ccw>>cap>>seed) {
    single_case_check((ccw==0 || ccw==1) && cap>=72 && cap<=224,"invalid relay replay plan");
    RelaySetupScan scan;
    scan.key=key;
    scan.seed=seed;
    auto local=rules;
    local.ccw=ccw!=0;
    RuleComparison result;
    if (tag=="H") result=scan_jqh_pressure(scan,local,cap);
    else {
      Rules archive;
      single_case_check(tag=="G" && bool(std::cin>>tape) && parse_rules(archive_review_rules(config,local.ccw),archive,error),
                        "invalid relay replay input");
      local.marshal_reveals_flag=archive.marshal_reveals_flag;
      result=scan_jgs_pressure(scan,ArchiveRows(decode_archive_tape(tape)),archive,local,cap);
    }
    std::cout<<"A "<<key<<' '<<result.agrees<<' '<<result.divergence<<' '<<result.reason<<' '<<scan.examined<<'\n';
    scan.report(local,remaining);
  }
  single_case_check(std::cin.eof(),"incomplete relay payload");
}
