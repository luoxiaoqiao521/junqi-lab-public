#pragma once

struct VisibleReplyBranch {
  jq::Observation view;
  jq::StepInfo first;
  std::optional<jq::StepInfo> reply;
  std::vector<jq::Move> shared;
};

static bool same_reply_event(const jq::StepInfo& a,const jq::StepInfo& b) {
  return a.move==b.move && a.seat==b.seat && a.outcome==b.outcome &&
         a.eliminated_mask==b.eliminated_mask && a.revealed_mask==b.revealed_mask &&
         a.skipped_mask==b.skipped_mask;
}

static bool same_reply_branch(const VisibleReplyBranch& a,const VisibleReplyBranch& b) {
  const auto& x=a.view;
  const auto& y=b.view;
  return x.viewer==y.viewer && x.turn==y.turn && x.ply==y.ply && x.match==y.match &&
         std::equal(std::begin(x.seats),std::end(x.seats),std::begin(y.seats)) &&
         std::equal(std::begin(x.ranks),std::end(x.ranks),std::begin(y.ranks)) &&
         same_reply_event(a.first,b.first) && a.reply.has_value()==b.reply.has_value() &&
         (!a.reply || same_reply_event(*a.reply,*b.reply));
}

struct ConditionalFinish {
  bool supported=false;
  size_t branches=0, unresolved=0;
};

// The worlds start at the same observable history. Group only across their
// extension by this candidate and the next public event.
static ConditionalFinish conditional_reply_finish(const std::vector<jq::State>& worlds,
    jq::Move candidate,const jq::Rules& rules,int team) {
  using namespace jq;
  std::vector<VisibleReplyBranch> grouped;
  const auto winner=team==0?Result::Team0Wins:Result::Team1Wins;
  auto consume=[&](const State& state,const StepInfo& first,std::optional<StepInfo> reply) {
    if (state.result==winner) return;
    VisibleReplyBranch branch{observe(state,state.to_move),first,reply,terminal_moves(state,rules,team)};
    const auto match=std::find_if(grouped.begin(),grouped.end(),[&](const auto& other) {
      return same_reply_branch(branch,other);
    });
    if (match==grouped.end()) {
      grouped.push_back(std::move(branch));
      return;
    }
    auto& shared=match->shared;
    shared.erase(std::remove_if(shared.begin(),shared.end(),[&](const auto move) {
      return std::find(branch.shared.begin(),branch.shared.end(),move)==branch.shared.end();
    }),shared.end());
  };
  for (auto after:worlds) {
    single_case_check(is_legal(after,candidate),"conditional continuation has an illegal root action");
    const auto first=step(after,candidate,rules);
    if (after.result!=Result::Ongoing || team_of(after.to_move,after.match)==team) {
      consume(after,first,std::nullopt);
      continue;
    }
    std::vector<Move> replies;
    legal_moves(after,replies);
    for (const auto reply:replies) {
      auto next=after;
      const auto event=step(next,reply,rules);
      consume(next,first,event);
    }
  }
  const auto unresolved=std::count_if(grouped.begin(),grouped.end(),[](const auto& branch) {
    return branch.shared.empty();
  });
  return {!worlds.empty() && unresolved==0,grouped.size(),static_cast<size_t>(unresolved)};
}

static void conditional_public_test(const jq::Rules& rules) {
  using namespace jq;
  auto weaker=selfplay_fixture();
  const Move prepare{0,1};
  place_single_case_piece(weaker,1,0,Platoon);
  place_single_case_piece(weaker,2,51,Division);
  place_single_case_piece(weaker,3,55,Regiment);
  place_single_case_piece(weaker,26,50,Bomb);
  place_single_case_piece(weaker,27,30,Platoon);
  place_single_case_piece(weaker,28,31,Platoon);
  place_single_case_piece(weaker,29,32,Company);
  place_single_case_piece(weaker,30,35,Battalion);
  auto after=weaker;
  step(after,prepare,rules);
  const std::array<Move,2> blocks{{{50,51},{50,55}}};
  const std::array<Move,2> finish{{{55,56},{51,56}}};
  for(size_t i=0;i<blocks.size();++i) {
    auto replied=after;
    single_case_check(is_legal(replied,blocks[i]),"expected opposing block is not legal");
    step(replied,blocks[i],rules);
    single_case_check(terminal_moves(replied,rules,0)==std::vector<Move>{finish[i]},
                      "each public block should require a different finish");
  }
  const auto old=public_reply_proof({weaker},prepare,rules,0);
  const auto changed=conditional_reply_finish({weaker},prepare,rules,0);
  single_case_check(!old.finishes && changed.supported && changed.branches>=2,
                    "known replies could not trigger different continuations");
  std::cout<<"different_public_blocks "<<old.finishes<<' '<<changed.supported<<' '
           <<changed.branches<<' '<<changed.unresolved<<'\n';
}

static void conditional_hidden_test(const jq::Rules& rules) {
  using namespace jq;
  auto left=selfplay_fixture();
  place_single_case_piece(left,1,0,Platoon);
  place_single_case_piece(left,2,51,Division);
  place_single_case_piece(left,3,53,Regiment);
  place_single_case_piece(left,26,30,Platoon);
  place_single_case_piece(left,27,58,Mine);
  auto right=left;
  right.rank_at[56]=right.rank_of[25]=Mine;
  right.rank_at[58]=right.rank_of[27]=Flag;
  const Move prepare{0,1};
  const auto left_only=conditional_reply_finish({left},prepare,rules,0);
  const auto right_only=conditional_reply_finish({right},prepare,rules,0);
  const auto old=public_reply_proof({left,right},prepare,rules,0);
  const auto same=conditional_reply_finish({left,right},prepare,rules,0);
  single_case_check(same_seat_view(left,right,0) && left_only.supported && right_only.supported &&
                    !old.finishes && !same.supported && same.unresolved>0,
                    "conditional proof used hidden flag identity as a branch key");
  std::cout<<"identical_view_hidden_flag_swap "<<old.finishes<<' '<<same.supported<<' '
           <<same.branches<<' '<<same.unresolved<<'\n';
}
