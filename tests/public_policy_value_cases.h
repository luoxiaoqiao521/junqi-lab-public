#pragma once
#include "../sim/public_policy_value_input.h"
#include "single_rules_cases.h"
#include <chrono>
#include <numeric>
#include <utility>

namespace jq {

inline size_t pv_index(const PublicDecision& input, Move move) {
  const auto i=std::find(input.legal.begin(),input.legal.end(),move);
  expect_value(i!=input.legal.end(),"missing legal fixture action");
  return static_cast<size_t>(i-input.legal.begin());
}

inline Move pv_fixed_reply(const Observation& view, const std::vector<Move>& legal, bool retaliate) {
  if (retaliate) for (const auto move : legal)
    if (view.rank_at(move.to)==Flag) return move;
  for (const auto move : legal) if (view.seat_at(move.to)<0) return move;
  return legal.front();
}

inline float pv_fixture_return(State state, const Rules& rules, Move choice, bool retaliate,
                               Outcome outcome, unsigned* plies) {
  const int learner=state.to_move;
  expect_value(is_legal(state,choice),"illegal probe or alternative");
  const auto first=step(state,choice,rules);
  expect_value(first.outcome==outcome,"fixture collision did not match its public outcome");
  int decisions=1;
  while (state.result==Result::Ongoing && decisions<=rules.no_progress_limit+2) {
    const auto next=seat_decision(state);
    const auto reply=pv_fixed_reply(next.observation,next.actions,retaliate);
    expect_value(is_legal(state,reply),"fixed public reply is illegal");
    step(state,reply,rules);
    ++decisions;
  }
  *plies+=static_cast<unsigned>(decisions);
  const auto score=team_score(state.result,learner);
  expect_value(score.has_value(),"unsettled tactical fixture");
  return static_cast<float>(*score);
}

inline std::vector<PublicExample> pv_tactical_cases(const Rules& rules) {
  unsigned plies=0;
  std::vector<PublicExample> examples;
  for (bool sacrifice : {true,false}) {
    auto state=single_case_state();
    state.no_progress=rules.no_progress_limit-(sacrifice ? 2 : 1);
    state.flag_revealed[0]=true;
    place_single_case_piece(state,1,22,Platoon);
    place_single_case_piece(state,2,0,Engineer);
    place_single_case_piece(state,26,21,sacrifice ? Bomb : Engineer);
    if (!sacrifice) place_single_case_piece(state,27,23,Engineer);
    const Move quiet{22,17}, probe{22,static_cast<uint8_t>(sacrifice ? 21 : 23)};
    const auto outcome=sacrifice ? Outcome::BothRemoved : Outcome::AttackerWins;
    for (bool retaliate : {true,false}) {
      const auto a=pv_fixture_return(state,rules,probe,retaliate,outcome,&plies);
      const auto b=pv_fixture_return(state,rules,quiet,retaliate,Outcome::Move,&plies);
      expect_value(a==(sacrifice ? 1.f : retaliate ? 0.f : .5f),"wrong probe team return");
      expect_value(b==(sacrifice && retaliate ? 0.f : .5f),"wrong quiet team return");
      if (retaliate) {
        auto input=encode_public_decision(observe(state,0),{},rules.ccw);
        const auto preferred=pv_index(input,a>b ? probe : quiet);
        examples.push_back({std::move(input),preferred,a,true});
      }
      std::cout << "PV tactical sacrifice=" << sacrifice << " retaliate=" << retaliate
                << " probe=" << a << " quiet=" << b << '\n';
    }
  }
  std::cout << "PASS public reply counterexamples branches=8 plies=" << plies << '\n';
  return examples;
}

inline void pv_collision_evidence(const Rules& rules) {
  auto state=single_case_state();
  state.ply=state.no_progress=0;
  place_single_case_piece(state,1,22,Engineer);
  place_single_case_piece(state,2,0,Platoon);
  place_single_case_piece(state,26,23,Division);
  const auto before=observe(state,0);
  PublicPrefix prefix(before);
  const Move attack{22,23};
  expect_value(is_legal(state,attack),"public collision fixture action illegal");
  const auto event=step(state,attack,rules);
  const auto after=observe(state,0);
  prefix.advance(before,after,attack,event.outcome,event.skipped_mask,event.eliminated_mask,rules);
  const auto input=encode_public_decision(after,prefix.history(),rules.ccw);
  const auto node=23*pv_node_width;
  expect_value(event.outcome==Outcome::DefenderWins && input.board[node+4]==1.f &&
               input.board[node+19]==1.f && input.board[node+20+Bomb]==1.f &&
               input.board[node+20+Engineer]==1.f && input.board[node+20+Division]==0.f &&
               input.legal.empty() && predict_public(PublicModel{},input).policy.empty(),
               "public collision evidence, exclusion or non-actor mask failed");
}

inline void pv_retirement_cases(const Rules& rules, const PublicModel& model) {
  for (bool partner : {false,true}) {
    auto state=single_case_state();
    state.no_progress=0;
    state.out[2]=state.out[3]=false;
    place_single_case_piece(state,1,partner ? 23 : 22,Platoon);
    place_single_case_piece(state,26,partner ? 30 : 23,Engineer);
    place_single_case_piece(state,50,86,Flag);
    place_single_case_piece(state,51,21,partner ? Bomb : Engineer);
    place_single_case_piece(state,75,116,Flag);
    place_single_case_piece(state,76,90,Engineer);
    const int viewer=partner ? 2 : 0, retired=partner ? 0 : 1;
    // The acting last mover locks itself in HQ, or the actor clears an enemy's last mover.
    const auto action=partner ? Move{23,28} : Move{22,23};
    const auto before=encode_public_decision(observe(state,viewer),{},rules.ccw);
    expect_value(is_legal(state,action),"retirement fixture action illegal");
    const auto event=step(state,action,rules);
    expect_value(event.eliminated_mask==(1u<<retired) && state.result==Result::Ongoing &&
                 state.to_move==(partner ? 1 : 2),"actual seat order after retirement");
    const auto after=encode_public_decision(observe(state,viewer),{},rules.ccw);
    const int active=kNodes*pv_node_width+relative_seat(retired,viewer);
    expect_value(before.board[active]==1.f && after.board[active]==0.f,"stale retirement input");
    unsigned calls=0;
    predict_public(model,encode_public_decision(observe(state,state.to_move),{},rules.ccw),&calls);
    expect_value(calls==1,"post-retirement decision was not encoded");
  }
}

inline void pv_hidden_worlds(const PublicModel& model, const Rules& rules) {
  Rng rng(2119);
  int8_t layouts[kSeats][kPiecesPerSeat];
  for (auto& layout : layouts) generate_layout(layout,rng);
  State a,b;
  init_state(a,layouts,0,rules);
  const auto left=std::find(std::begin(layouts[3]),std::end(layouts[3]),Battalion);
  const auto right=std::find(std::begin(layouts[3]),std::end(layouts[3]),Regiment);
  std::iter_swap(left,right);
  init_state(b,layouts,0,rules);
  expect_value(state_hash(a)!=state_hash(b),"hidden truth was not perturbed");
  PublicPrefix first(observe(a,0)), second(observe(b,0));
  for (int i=0; i<8; ++i) {
    const auto next=seat_decision(a);
    const auto move=std::find_if(next.actions.begin(),next.actions.end(),[&](Move m) {
      return next.observation.seat_at(m.to)<0;
    });
    expect_value(move!=next.actions.end() && is_legal(b,*move),"no common quiet history");
    const auto before_a=observe(a,0), before_b=observe(b,0);
    const auto ea=step(a,*move,rules), eb=step(b,*move,rules);
    first.advance(before_a,observe(a,0),*move,ea.outcome,ea.skipped_mask,ea.eliminated_mask,rules);
    second.advance(before_b,observe(b,0),*move,eb.outcome,eb.skipped_mask,eb.eliminated_mask,rules);
  }
  const auto x=encode_public_decision(observe(a,0),first.history(),rules.ccw);
  const auto y=encode_public_decision(observe(b,0),second.history(),rules.ccw);
  expect_value(!x.legal.empty() && x.board==y.board && x.legal==y.legal && x.action==y.action,
               "hidden truth changed public model inputs");
  const auto px=predict_public(model,x), py=predict_public(model,y);
  expect_value(px.policy==py.policy && px.value==py.value,"hidden truth changed scores");
  const auto unknown=encode_public_decision(observe(a,0),{},rules.ccw);
  expect_value(x.board!=unknown.board && x.legal==unknown.legal,"unknown prefix evidence not marked");
}

inline PublicExample pv_expression_case(int left, int right, bool heldout) {
  auto state=single_case_state();
  state.ply=14;
  place_single_case_piece(state,1,22,Platoon);
  place_single_case_piece(state,26,23,Engineer);
  for (int i=0; i<3; ++i) {
    place_single_case_piece(state,27+i,40+5*left+i,Platoon);
    place_single_case_piece(state,30+i,70+5*right+i,Engineer);
  }
  if (heldout) place_single_case_piece(state,33,111,Engineer);
  PublicExample row{encode_public_decision(observe(state,0),{},true)};
  row.preferred=pv_index(row.input,left!=right ? Move{22,23} : Move{22,18});
  row.team_return=float(left!=right);
  row.settled=true;
  return row;
}

inline void pv_interaction_contract(const std::array<PublicExample,4>& rows) {
  for (size_t i=0; i<rows[0].input.board.size(); ++i)
    expect_value(rows[0].input.board[i]+rows[3].input.board[i] ==
                 rows[1].input.board[i]+rows[2].input.board[i],"linear contrast is not additive");
  for (const auto& row : rows)
    expect_value(row.input.legal==rows[0].input.legal && row.input.action==rows[0].input.action,
                 "linear contrast changed the candidate set or action features");
}

struct PVLinear {
  // Bilinear in the raw board and action, with neither hidden layer.
  std::vector<float> weights=std::vector<float>(pv_width*pv_action_width);
  std::array<float,pv_action_width> bias{};
  std::array<float,pv_width> value_weights{};
  float value_bias=0.f;
  std::vector<float> predict(const PublicDecision& input) const {
    auto context=bias;
    for (int i=0; i<pv_width; ++i) if (input.board[i])
      for (int j=1; j<pv_action_width; ++j)
        context[j]+=weights[i*pv_action_width+j]*input.board[i];
    return public_distribution(input,context);
  }
  float value(const PublicDecision& input) const {
    float logit=value_bias;
    for (int i=0; i<pv_width; ++i) if (input.board[i]) logit+=value_weights[i]*input.board[i];
    return 1.f/(1.f+std::exp(-logit));
  }
  void update(const PublicExample& row, float rate) {
    if (!row.settled) return;
    const auto error=public_policy_error(row,predict(row.input));
    const float v=value(row.input), value_error=(v-row.team_return)*v*(1.f-v);
    value_bias-=rate*value_error;
    for (int j=1; j<pv_action_width; ++j) bias[j]-=rate*error[j];
    for (int i=0; i<pv_width; ++i) if (row.input.board[i]) {
      value_weights[i]-=rate*value_error*row.input.board[i];
      for (int j=1; j<pv_action_width; ++j)
        weights[i*pv_action_width+j]-=rate*error[j]*row.input.board[i];
    }
  }
};

inline void pv_check_bad_models(const std::filesystem::path& path, uint64_t rule_hash) {
  const auto original=read_text(path.string());
  const std::array<std::pair<size_t,std::string>,4> edits{{
    {0,"X"}, {8,"X"}, {28,"X"}, {36,std::string("\0\0\xc0\x7f",4)}
  }};
  auto bad=path;
  bad += ".bad";
  for (const auto& edit : edits) {
    auto damaged=original;
    damaged.replace(edit.first,edit.second.size(),edit.second);
    std::ofstream out(bad,std::ios::binary);
    out.write(damaged.data(),static_cast<std::streamsize>(damaged.size()));
    out.close();
    bool rejected=false;
    try { load_public_model(bad,rule_hash); } catch (const std::runtime_error&) { rejected=true; }
    expect_value(rejected,"corrupt public model accepted");
  }
  std::filesystem::remove(bad);
}

inline void pv_write_delayed_overflow_model(std::filesystem::path path, uint64_t rule_hash) {
  PublicModel extreme;
  const float largest=std::numeric_limits<float>::max();
  // Any moved bit activates two shared units. Their finite weights overflow one policy coefficient.
  for (int node=0; node<kNodes; ++node) extreme.weights[pv_w1+node*pv_node_width+17]=100.f;
  extreme.weights[pv_w2]=100.f;
  extreme.weights[pv_w2+pv_first]=100.f;
  extreme.weights[pv_policy+pv_hidden]=largest;
  extreme.weights[pv_policy+pv_hidden+1]=largest;
  path += ".overflow";
  save_public_model(path,extreme,rule_hash);
}

inline void pv_evaluate(const PublicModel& model, const PublicModel& loaded, const PVLinear& linear) {
  const std::array<PublicExample,4> heldout{pv_expression_case(0,0,true),pv_expression_case(0,1,true),
                                           pv_expression_case(1,0,true),pv_expression_case(1,1,true)};
  pv_interaction_contract(heldout);
  int nonlinear_hits=0, linear_hits=0;
  double squared_error=0, linear_squared_error=0;
  for (size_t index=0; index<heldout.size(); ++index) {
    const auto& row=heldout[index];
    const auto other=pv_index(row.input,row.team_return==1.f ? Move{22,18} : Move{22,23});
    unsigned calls=0;
    const auto pred=predict_public(loaded,row.input,&calls);
    const auto before=predict_public(model,row.input);
    const auto baseline=linear.predict(row.input);
    expect_value(calls==1 && pred.policy==before.policy && pred.value==before.value &&
                 pred.probability(row.input,Move{128,128})==0.f &&
                 std::abs(std::accumulate(pred.policy.begin(),pred.policy.end(),0.f)-1.f)<1e-6f,
                 "reload, one encoding or policy mask failed");
    nonlinear_hits+=pred.policy[row.preferred]>pred.policy[other];
    linear_hits+=baseline[row.preferred]>baseline[other];
    squared_error+=std::pow(pred.value-row.team_return,2);
    linear_squared_error+=std::pow(linear.value(row.input)-row.team_return,2);
    std::cout << "PV expression " << index/2 << index%2 << " p=" << pred.policy[row.preferred]
              << " vs=" << pred.policy[other] << " v=" << pred.value
              << " linear_p=" << baseline[row.preferred] << " vs=" << baseline[other] << '\n';
  }
  expect_value(nonlinear_hits==4 && linear_hits<4,"held-out nonlinear interaction not established");
  std::cout << "PASS expression heldout_rank=" << nonlinear_hits << "/4 linear_rank=" << linear_hits
            << "/4 value_rmse=" << std::sqrt(squared_error/4)
            << " linear_value_rmse=" << std::sqrt(linear_squared_error/4) << '\n';
}

inline PublicExample pv_cutoff(Rules rules) {
  auto state=single_case_state();
  state.no_progress=0;
  rules.truncation_cap=state.ply+1;
  place_single_case_piece(state,1,22,Platoon);
  place_single_case_piece(state,26,30,Engineer);
  PublicExample row{encode_public_decision(observe(state,0),{},rules.ccw)};
  const Move quiet{22,17};
  row.preferred=pv_index(row.input,quiet);
  step(state,quiet,rules);
  const auto score=team_score(state.result,0);
  expect_value(state.result==Result::Truncated && !score,"expected an actual masked truncation");
  row.settled=score.has_value();
  return row;
}

inline void public_policy_value_cases(const Rules& rules, const std::filesystem::path& path) {
  const auto started=std::chrono::steady_clock::now();
  const auto tactical=pv_tactical_cases(rules);
  pv_collision_evidence(rules);
  constexpr uint64_t seed=41019;
  constexpr int epochs=4000;
  constexpr float rate=.01f;
  const std::array<PublicExample,4> train{pv_expression_case(0,0,false),pv_expression_case(0,1,false),
                                         pv_expression_case(1,0,false),pv_expression_case(1,1,false)};
  auto model=initialize_public_model(seed);
  const auto initial=model.weights;
  expect_value(!update_public_model(model,pv_cutoff(rules),rate) && initial==model.weights,
               "truncation updated weights");
  PVLinear linear;
  unsigned updates=0, terminal_updates=0;
  Rng order(seed);
  std::array<unsigned,4> indices{0,1,2,3};
  for (int epoch=0; epoch<epochs; ++epoch) {
    for (unsigned i=4; i>1; --i) std::swap(indices[i-1],indices[order.below(i)]);
    for (const auto index : indices) {
      updates+=update_public_model(model,train[index],rate);
      linear.update(train[index],rate);
    }
    if (epoch%16==0) for (const auto& row : tactical) {
      terminal_updates+=update_public_model(model,row,rate);
      linear.update(row,rate);
    }
  }
  for (const auto span : {std::make_pair(pv_w1,pv_w2),std::make_pair(pv_w2,pv_policy),
                          std::make_pair(pv_policy,pv_value),std::make_pair(pv_value,pv_parameters)})
    expect_value(!std::equal(initial.begin()+span.first,initial.begin()+span.second,
                 model.weights.begin()+span.first),"one parameter group did not update");
  const uint64_t rule_hash=rules_hash(rules);
  save_public_model(path,model,rule_hash);
  const auto loaded=load_public_model(path,rule_hash);
  expect_value(model.weights==loaded.weights,"float32 round-trip changed parameters");
  pv_check_bad_models(path,rule_hash);
  pv_write_delayed_overflow_model(path,rule_hash);
  pv_retirement_cases(rules,loaded);
  pv_hidden_worlds(loaded,rules);
  pv_evaluate(model,loaded,linear);
  const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  std::cout << "PASS public policy/value encoding=JQPV0001 seed=" << seed << " rate=" << rate
            << " params=" << pv_parameters << " bytes=" << std::filesystem::file_size(path)
            << " rules=" << rule_hash << " training_roots=6 heldout_roots=4 epochs=" << epochs
            << " updates=" << updates+terminal_updates << " terminal_updates=" << terminal_updates
            << " cutoff_updates=0 seconds=" << seconds << '\n';
}

}  // namespace jq
