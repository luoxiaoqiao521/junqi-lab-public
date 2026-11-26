#include <iostream>
#include <filesystem>
#include "header.h"
#include "decision.h"
#include "replay.h"
#include "archive_review.h"
#include "config.h"
#include "learning_probe.h"
#include "transfer_probe.h"
#include "opening_train.h"
#include "archive_curriculum.h"
#include "draw_branch.h"
#include "decision_branches.h"
#include "credit_window.h"
#include "camp_control_pairs.h"
#include "flag_interception.h"
#include "mine_evidence.h"
#include "flag_urgency.h"
#include "exchange_counterplay.h"
#include "exchange_counterexamples.h"
#include "flag_guard_transfer.h"
#include "mine_choice.h"
#include "mine_collision.h"
#include "goal_followthrough.h"
#include "regional_arrival.h"
#include "early_route.h"
#include "formation_pair.h"
#include "partner_divergence.h"
#include "parent_update_audit.h"
#include "public_policy_value_input.h"
#include "../core/layout.h"
#include "../strategy/random.h"
namespace jq {
void simulate(uint64_t seed, const std::string& file, const std::string& rules_file,
              const std::string& obs_file, const std::string& policy,
              MatchKind match, IdentityVisibility visibility) {
  const auto mode=simulation_policy(policy);
  Rules rules; std::string err, rt = read_text(rules_file), ot = read_text(obs_file);
  auto rules_document = parse_json(rt);
  if (match==MatchKind::SouthNorth) {
    rules_document.object["id"] = parse_json("\"single-dark-derived-1\"");
    rules_document.object["players"] = parse_json("2");
    rules_document.object["teams"] = parse_json("[[0],[2]]");
  }
  rt = dump_json(rules_document);
  if (!parse_rules(rt, rules, err)) throw std::runtime_error(err);
  ot = match_observation_config(ot,rules,visibility);
  if (std::filesystem::exists(file)) throw std::runtime_error("output exists; choose a new log path");
  Rng layout_rng(seed); int8_t layouts[kSeats][kPiecesPerSeat]{};
  int participants=0;
  // Draw layouts in the same order so the seed pairs the retained seats across modes.
  for (int seat=0; seat<kSeats; ++seat) {
    generate_layout(layouts[seat],layout_rng);
    if (participates(match,seat)) participants |= 1 << seat;
  }
  if (mode.heuristic_seats && !(mode.heuristic_seats & participants))
    throw std::runtime_error("policy selects no participant");
  int first=rules.random_first_seat ? int(layout_rng.below(kSeats)) : 0;
  while (!participates(match,first)) first=next_seat(first,rules);
  State s; init_state(s, layouts, first, rules, match); Rng policy_rng(mix_seed(seed, 1));
  std::ofstream log(file, std::ios::binary); if (!log) throw std::runtime_error("cannot create log");
  write_header(log, seed, rules, rt, ot, first, layouts, policy, match);
  log << frame_json(s,nullptr,visibility) << '\n';
  while (s.result == Result::Ongoing) {
    Move m;
    const auto observation = observe(s,s.to_move,visibility);
    const DecisionScorer scorer(observation,rules,mode.at(s.to_move));
    if (!scorer.choose(policy_rng,m) || !is_legal(s, m)) throw std::runtime_error("no valid action");
    auto event = step(s, m, rules);
    log << (mode.records_decision() ? decision_frame(s,event,decision_record(scorer,m),visibility,rules.eliminate_exhausted_movers)
                 : frame_json(s,&event,visibility,rules.eliminate_exhausted_movers)) << '\n';
  }
  log.flush(); if (!log) throw std::runtime_error("log write failed");
  std::cout << "seed=" << seed << " plies=" << s.ply << " result=" << int(s.result) << " log=" << file << '\n';
}
void run_research_probe(int argc, char** argv) {
  jq::Rules rules; std::string error;
  if (!jq::parse_rules(jq::read_text(argv[2]),rules,error)) throw std::runtime_error(error);
  ShortWeights weights{};
  const std::string command=argv[1];
  const int width = argc == 13 || (command=="resume-position" || command=="resume-evidence") ? 6 : 5;
  for (int i=0;i<width;++i) {
    size_t used=0; weights[i]=std::stod(argv[i+3],&used);
    if (used!=std::string(argv[i+3]).size() || !std::isfinite(weights[i]))
      throw std::runtime_error("invalid weight");
  }
  if (argc == 12 || argc == 13) {
    const std::string policy=argv[width+5], variant=argv[width+6];
    const std::string start=argv[width+3], count=argv[width+4];
    if (start.find_first_not_of("0123456789")!=std::string::npos ||
        count.find_first_not_of("0123456789")!=std::string::npos ||
        (policy!="short" && policy!="short-final-flag" && policy!="random" && policy!="heuristic" &&
         policy!="position" && policy!="evidence") ||
        ((policy=="position" || policy=="evidence")!=(width==6)) ||
        (variant!="0" && variant!="1" && variant!="2"))
      throw std::runtime_error("invalid transfer configuration");
    const auto first=std::stoull(start), n=std::stoull(count);
    if (n==0 || n>1024 || first>UINT64_MAX-n) throw std::runtime_error("transfer budget exceeded");
    jq::transfer_probe(rules,weights,first,n,policy,std::stoi(variant));
  } else if (command != "transfer-probe") {
    const std::string seed = argc == 8 ? "50000" : argv[width+3];
    if (seed.empty() || seed.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error("invalid training seed");
    const auto first = std::stoull(seed);
    if (first > UINT64_MAX - 511) throw std::runtime_error("training seed range overflow");
    jq::train_openings(rules,weights,first,command=="train-position" || command=="resume-position",
                       command=="train-position",command=="train-evidence" || command=="resume-evidence");
  } else jq::transfer_probe(rules,weights);
}
void run_crossplay(int argc, char** argv) {
  Rules rules; std::string error;
  if (!parse_rules(read_text(argv[2]),rules,error)) throw std::runtime_error(error);
  const std::string start=argv[3], count=argv[4];
  for (const auto& number : {start,count})
    if (number.empty() || number.find_first_not_of("0123456789")!=std::string::npos)
      throw std::runtime_error("invalid cross-play seed range");
  const auto first=std::stoull(start), n=std::stoull(count);
  if (n==0 || n>1024 || first>UINT64_MAX-n) throw std::runtime_error("cross-play budget exceeded");
  const size_t profiles=argc==29 ? 4 : 3;
  std::array<FrozenSeat,4> seats;
  for (size_t index=0; index<profiles; ++index) {
    auto& seat=seats[index];
    seat.policy=argv[5+index];
    if (seat.policy!="short" && seat.policy!="heuristic" && seat.policy!="random" &&
        !(index>0 && seat.policy=="contact") &&
        !(index==0 && seat.policy=="short-final-flag" && std::string(argv[1])=="cross-play"))
      throw std::runtime_error("invalid frozen seat policy");
    for (size_t j=0; j<5; ++j) {
      const std::string value=argv[5+profiles+5*index+j];
      size_t used=0;
      seat.weights[j]=std::stod(value,&used);
      if (used!=value.size() || !std::isfinite(seat.weights[j]))
        throw std::runtime_error("invalid frozen seat weight");
    }
  }
  if (std::string(argv[1])=="cross-play-divergence")
    partner_divergence(rules,first,n,seats);
  else
    transfer_probe(rules,seats[0].weights,first,n,seats[0].policy,0,&seats[1],&seats[2],
                   profiles==4 ? &seats[3] : nullptr);
}

void run_parent_audit(int argc,char** argv) {
  ParentAudit settings{};
  if (argc>=23) {
    settings.confirm_choice=std::stoi(argv[22]);
    if (settings.confirm_choice<1 || settings.confirm_choice>4)
      throw std::runtime_error("invalid fixed audit choice");
  }
  if (argc==24) {
    const auto value=std::stoi(argv[23]);
    if (std::string(argv[1])=="parent-candidate-hq-swap") {
      if (value<0 || value>=kNodes) throw std::runtime_error("invalid alternate HQ");
      settings.other_hq=value;
    } else {
      if (value<0 || value>=8*kAuditConfirmationPairs) throw std::runtime_error("invalid audit witness");
      settings.witness=value;
    }
  }
  std::string error;
  if (!parse_rules(read_text(argv[2]),settings.rules,error)) throw std::runtime_error(error);
  settings.seed=std::stoull(argv[3]);
  settings.learner=std::stoi(argv[4]);
  settings.anchor=std::stoi(argv[5]);
  const std::string partner=argv[6];
  if (settings.learner<0 || settings.learner>=kSeats || settings.anchor<0 || settings.anchor>50 ||
      (partner!="short" && partner!="random")) throw std::runtime_error("invalid audit profile");
  auto& parent=settings.seats[settings.learner];
  auto& opposite=settings.seats[(settings.learner+2)%4];
  for (auto& seat : settings.seats) seat.policy="random";
  parent.policy="short";
  opposite.policy=partner;
  for (size_t i=0;i<5;++i) {
    parent.weights[i]=std::stod(argv[7+i]);
    settings.child[i]=std::stod(argv[12+i]);
    opposite.weights[i]=std::stod(argv[17+i]);
    if (!std::isfinite(parent.weights[i]) || !std::isfinite(settings.child[i]) ||
        !std::isfinite(opposite.weights[i])) throw std::runtime_error("nonfinite audit weight");
  }
  parent_update_audit(settings);
}

void run_update_probe(int argc, char** argv) {
  Rules rules; std::string error;
  if (!parse_rules(read_text(argv[2]),rules,error)) throw std::runtime_error(error);
  const std::string seed_text=argv[3], rate_text=argv[4], baseline=argv[5], batch_text=argv[6];
  const std::string policy=argc>=9 ? argv[7] : "short", profile=argc>=9 ? argv[8] : "0";
  const bool continuation=argc==14;
  if (seed_text.empty() || seed_text.find_first_not_of("0123456789")!=std::string::npos ||
      (baseline!="0" && baseline!="1") || (policy!="short" && policy!="position" && policy!="evidence") ||
      (profile!="0" && profile!="3" && !(continuation && profile=="4")) ||
      (continuation && (policy!="short" || profile=="3")) ||
      (batch_text!="32" && batch_text!="256" && !(argc==9 && batch_text=="1024") &&
       !(continuation && batch_text=="64")))
    throw std::runtime_error("invalid update experiment");
  const auto seed=std::stoull(seed_text);
  size_t used=0; const double rate=std::stod(rate_text,&used);
  if (used!=rate_text.size() || !std::isfinite(rate) || rate<=0 || rate>.5 || seed>UINT64_MAX-16384)
    throw std::runtime_error("update experiment budget or rate invalid");
  ShortWeights weights{};
  if (continuation)
    for (int index=0; index<5; ++index) {
      const std::string value=argv[9+index];
      weights[index]=std::stod(value,&used);
      if (used!=value.size() || !std::isfinite(weights[index]))
        throw std::runtime_error("invalid continuation weight");
    }
  train_openings(rules,weights,seed,policy=="position",false,policy=="evidence",rate,
                 std::stoi(batch_text),baseline=="1",true,std::stoi(profile));
}
}
int main(int argc, char** argv) {
  try {
    if (argc==6 && std::string(argv[1])=="replay-public-policy-value") {
      const std::string budget=argv[5];
      if (budget.empty() || budget.size()>3 || budget.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("invalid public replay decision budget");
      jq::replay_public_model(argv[2],argv[3],argv[4],static_cast<unsigned>(std::stoul(budget)));
      return 0;
    }
    if (argc==4 && std::string(argv[1])=="archive-curriculum") {
      jq::Rules rules; std::string error;
      if (!jq::parse_rules(jq::read_text(argv[2]),rules,error)) throw std::runtime_error(error);
      jq::run_archive_curriculum(rules,argv[3]); return 0;
    }
    if ((argc==22 && std::string(argv[1])=="parent-update-audit") ||
        (argc==23 && std::string(argv[1])=="parent-candidate-confirm") ||
        (argc==24 && (std::string(argv[1])=="parent-candidate-exercise" ||
                      std::string(argv[1])=="parent-candidate-hq-swap"))) {
      jq::run_parent_audit(argc,argv); return 0;
    }
    if (((argc==23 || argc==29) && std::string(argv[1])=="cross-play") ||
        (argc==29 && std::string(argv[1])=="cross-play-divergence")) {
      jq::run_crossplay(argc,argv); return 0;
    }
    if ((argc == 12 && (std::string(argv[1]) == "camp-control-pair" || std::string(argv[1]) == "flag-urgency-pair")) ||
        (argc == 11 && (std::string(argv[1]) == "decision-branches" ||
                       std::string(argv[1]) == "credit-windows" ||
                       std::string(argv[1]) == "credit-windows-reseed" ||
                       std::string(argv[1]) == "credit-windows-live" ||
                       std::string(argv[1]) == "camp-control-scan" ||
                       std::string(argv[1]) == "flag-interception" ||
                       std::string(argv[1]) == "mine-evidence" ||
                       std::string(argv[1]) == "flag-urgency-scan" ||
                       std::string(argv[1]) == "exchange-counterplay" ||
                       std::string(argv[1]) == "exchange-response-confirm" ||
                       std::string(argv[1]) == "exchange-counterexamples" ||
                       std::string(argv[1]) == "flag-guard-transfer" ||
                       std::string(argv[1]) == "mine-choice" ||
                       std::string(argv[1]) == "mine-information-value" ||
                       std::string(argv[1]) == "mine-quiet-recheck" ||
                       std::string(argv[1]) == "mine-collision-evidence" ||
                       std::string(argv[1]) == "regional-arrival" ||
                       std::string(argv[1]) == "early-route" ||
                       std::string(argv[1]) == "partner-route-followthrough" ||
                       std::string(argv[1]) == "goal-followthrough" ||
                       std::string(argv[1]) == "formation-one-swap"))) {
      jq::Rules rules; std::string error;
      if (!jq::parse_rules(jq::read_text(argv[2]),rules,error)) throw std::runtime_error(error);
      const std::string seed=argv[3], seat=argv[4], variant=argv[5];
      if (seed.empty() || seed.find_first_not_of("0123456789")!=std::string::npos ||
          seat.size()!=1 || seat[0]<'0' || seat[0]>'3' ||
          variant.size()!=1 || variant[0]<'0' || variant[0]>'2')
        throw std::runtime_error("invalid decision review configuration");
      jq::ShortWeights weights{};
      for (int i=0; i<5; ++i) {
        size_t used=0;
        const std::string text=argv[6+i];
        weights[i]=std::stod(text,&used);
        if (used!=text.size() || !std::isfinite(weights[i])) throw std::runtime_error("invalid review weight");
      }
      if (std::string(argv[1])=="camp-control-pair") {
        const std::string kind=argv[11];
        if (kind!="0" && kind!="1") throw std::runtime_error("invalid camp contrast group");
        jq::camp_control_pairs(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights,size_t(kind[0]-'0'));
      } else if (std::string(argv[1])=="flag-urgency-pair") {
        const std::string kind=argv[11];
        if (kind!="0" && kind!="1") throw std::runtime_error("invalid urgency contrast group");
        jq::flag_urgency(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights,kind[0]-'0');
      } else if (std::string(argv[1])=="decision-branches")
        jq::decision_branches(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="camp-control-scan")
        jq::scan_camp_control(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="flag-interception")
        jq::flag_interception(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="mine-evidence")
        jq::mine_evidence_probe(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="flag-urgency-scan")
        jq::flag_urgency(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="exchange-counterplay" ||
               std::string(argv[1])=="exchange-response-confirm")
        jq::exchange_counterplay(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights,
                                 std::string(argv[1])=="exchange-response-confirm");
      else if (std::string(argv[1])=="exchange-counterexamples")
        jq::exchange_counterexamples(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="flag-guard-transfer")
        jq::flag_guard_transfer(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="mine-choice" || std::string(argv[1])=="mine-information-value" ||
               std::string(argv[1])=="mine-quiet-recheck") {
        auto study=jq::MineStudy::AttackPair;
        if (std::string(argv[1])=="mine-information-value") study=jq::MineStudy::InformationValue;
        if (std::string(argv[1])=="mine-quiet-recheck") study=jq::MineStudy::QuietRecheck;
        jq::mine_choice_probe(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights,study);
      }
      else if (std::string(argv[1])=="mine-collision-evidence")
        jq::mine_collision_evidence(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="regional-arrival")
        jq::regional_arrival(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="early-route" || std::string(argv[1])=="partner-route-followthrough")
        jq::early_route_probe(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights,
                              std::string(argv[1])=="partner-route-followthrough");
      else if (std::string(argv[1])=="formation-one-swap")
        jq::formation_one_swap(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else if (std::string(argv[1])=="goal-followthrough")
        jq::goal_followthrough(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights);
      else jq::credit_windows(rules,std::stoull(seed),seat[0]-'0',variant[0]-'0',weights,
                              std::string(argv[1])=="credit-windows-reseed",
                              std::string(argv[1])=="credit-windows-live");
      return 0;
    }
    if ((argc==7 && std::string(argv[1])=="train-tuned") ||
        ((argc==9 || argc==14) && std::string(argv[1])=="train-general")) {
      jq::run_update_probe(argc,argv); return 0;
    }
    if ((argc == 8 && (std::string(argv[1]) == "transfer-probe" ||
                       std::string(argv[1]) == "train-openings")) ||
        ((argc == 12 || argc == 13) && std::string(argv[1]) == "transfer-probe") ||
        (argc == 9 && (std::string(argv[1]) == "train-openings" ||
                       std::string(argv[1]) == "train-position" || std::string(argv[1]) == "train-evidence")) ||
        (argc == 10 && (std::string(argv[1]) == "resume-position" || std::string(argv[1]) == "resume-evidence"))) {
      jq::run_research_probe(argc,argv);
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "learning-probe") {
      const auto number = [](const std::string& value) {
        uint64_t n = 0;
        const auto end = value.data() + value.size();
        const auto parsed = std::from_chars(value.data(), end, n);
        if (parsed.ec != std::errc{} || parsed.ptr != end)
          throw std::runtime_error("invalid learning-probe seed or count");
        return n;
      };
      const auto first = number(argv[3]), count = number(argv[4]);
      if (!count || count > 1024 || first > UINT64_MAX - count)
        throw std::runtime_error("learning-probe range outside budget");
      jq::Rules rules;
      std::string error;
      if (!jq::parse_rules(jq::read_text(argv[2]), rules, error)) throw std::runtime_error(error);
      jq::learning_probe(rules, first, count);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "replay") {
      if (!jq::verify_archive_review(jq::read_text(argv[2]))) jq::replay(argv[2]);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "draw-branches") {
      jq::draw_branches(argv[2]); return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "draw-target") {
      jq::draw_branches(argv[2],true); return 0;
    }
    if (argc == 3 && (std::string(argv[1]) == "pressure-ranking" ||
                      std::string(argv[1]) == "pressure-opponents" ||
                      std::string(argv[1]) == "pressure-moved")) {
      jq::draw_branches(argv[2],false,true,std::string(argv[1])!="pressure-ranking",
                        std::string(argv[1])=="pressure-moved"); return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "replay-candidates") {
      jq::replay(argv[2], true); return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "replay-public-evidence") {
      const std::string seat = argv[3];
      jq::expect_value(seat.size() == 1 && seat[0] >= '0' && seat[0] <= '3', "evidence seat must be 0-3");
      jq::replay(argv[2], false, {}, seat[0] - '0'); return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "export") {
      const auto body=jq::read_text(argv[2]);
      if (!jq::verify_archive_review(body)) jq::replay(argv[2]);
      if (std::filesystem::exists(argv[3])) throw std::runtime_error("export exists");
      std::ofstream out(argv[3], std::ios::binary);
      out << "window.JUNQI_LOG=" << jq::quote_json(body) << ";\n";
      out.flush(); if (!out) throw std::runtime_error("export failed"); return 0;
    }
    if (argc == 4 && (std::string(argv[1]) == "review" || std::string(argv[1]) == "export-review")) {
      if (std::filesystem::exists(argv[3])) throw std::runtime_error("review output exists");
      auto body=jq::read_text(argv[2]);
      if (!jq::verify_archive_review(body)) {
        std::ostringstream review;
        jq::replay(argv[2],false,{},-1,&review);
        body=review.str();
      }
      if (body.size()>64000000) throw std::runtime_error("review output too large");
      std::ofstream out(argv[3],std::ios::binary);
      if (std::string(argv[1]) == "export-review") out << "window.JUNQI_LOG=" << jq::quote_json(body) << ";\n";
      else out << body;
      out.flush(); if (!out) throw std::runtime_error("review export failed"); return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "import-jgs") {
      jq::import_archive_review(argv[2],argv[3],argv[4],argv[5]); return 0;
    }
    const std::string command=argc>1 ? argv[1] : "";
    if ((argc != 6 && argc != 7) ||
        (command!="simulate" && command!="simulate-single" && command!="simulate-dual" && command!="simulate-full"))
      throw std::runtime_error("Usage: junqi simulate|simulate-single|simulate-dual|simulate-full SEED OUTPUT RULES OBSERVATION");
    uint64_t seed = 0; std::string n = argv[2]; auto c = std::from_chars(n.data(), n.data() + n.size(), seed);
    if (c.ec != std::errc{} || c.ptr != n.data() + n.size()) throw std::runtime_error("invalid seed");
    const auto match=command=="simulate-single" ? jq::MatchKind::SouthNorth : jq::MatchKind::FourSeats;
    auto visibility=jq::IdentityVisibility::OwnAndPublic;
    if (command=="simulate-dual") visibility=jq::IdentityVisibility::AllyAndPublic;
    if (command=="simulate-full") visibility=jq::IdentityVisibility::AllIdentities;
    jq::simulate(seed,argv[3],argv[4],argv[5],argc==7 ? argv[6] : "random",match,visibility); return 0;
  } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
