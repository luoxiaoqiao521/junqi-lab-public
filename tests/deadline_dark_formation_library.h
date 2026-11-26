#pragma once

#include <set>

using FormationTemplate=std::array<int8_t,kPiecesPerSeat>;

struct FormationHistory {
  const CoverageRoot& root;
  std::array<unsigned,kNodes> allowed;
  std::array<int,kPiecesPerSeat> nodes;
  int focal_slot;
};

struct FormationMatches {
  int scanned=0, attempts=0, nodes=0, matched=0, bombs=0, unresolved=0, rejected=0, replay_failed=0;
};

static bool formation_fits(const FormationTemplate& layout,const FormationHistory& history) {
  for (int i=0;i<kPiecesPerSeat;++i)
    if (!(history.allowed[history.nodes[i]]&(1u<<layout[i]))) return false;
  return true;
}

static void try_formation(const FormationTemplate& layout,const FormationHistory& history,
                           FormationMatches& report,uint64_t salt,std::array<CoverageTrial,2>* picked) {
  ++report.attempts;
  std::array<unsigned,kNodes> opening_pins{},no_current_pins{};
  std::array<int8_t,kNodes> opening{};
  for (int i=0;i<kPiecesPerSeat;++i) opening_pins[history.nodes[i]]=1u<<layout[i];
  Rng rng(salt);
  CaptureSearch work{384,&rng,&opening};
  work.initial_pins=&opening_pins;
  const auto status=history.root.solve(no_current_pins,work);
  report.nodes+=384-work.remaining;
  if (status==CaptureCompatibility::Unresolved) { ++report.unresolved; return; }
  if (status==CaptureCompatibility::Contradicted) { ++report.rejected; return; }
  PrefixReplayWork replay;
  const auto world=history.root.verify(opening,replay);
  if (!world) { ++report.replay_failed; return; }
  for (int i=0;i<kPiecesPerSeat;++i)
    single_case_check(opening[history.nodes[i]]==layout[i],"verified opening did not retain its formation pins");
  ++report.matched;
  const bool bomb=layout[history.focal_slot]==Bomb;
  report.bombs+=bomb;
  if (picked && (*picked)[bomb].worlds.size()<2) (*picked)[bomb].worlds.push_back(*world);
}

static std::optional<FormationTemplate> repair_formation(FormationTemplate layout,const FormationHistory& context) {
  int a=kNone,b=kNone;
  for (int i=0;i<kPiecesPerSeat;++i) if (!(context.allowed[context.nodes[i]]&(1u<<layout[i]))) {
    if (b!=kNone) return std::nullopt;
    if (a==kNone) a=i;
    else b=i;
  }
  if (a==kNone) return std::nullopt;
  for (int j=0;j<kPiecesPerSeat;++j) {
    if (j==a || (b!=kNone && j!=b) ||
        !(context.allowed[context.nodes[a]]&(1u<<layout[j])) ||
        !(context.allowed[context.nodes[j]]&(1u<<layout[a]))) continue;
    if (!deployment_rank_allowed(context.nodes[a],layout[j]) || !deployment_rank_allowed(context.nodes[j],layout[a]))
      continue;
    std::swap(layout[a],layout[j]);
    return layout;
  }
  return std::nullopt;
}

static void count_formations(const std::string& key,int source,const std::vector<FormationTemplate>& library,
                             const FormationHistory& history,uint64_t seed,FormationWorlds* picked) {
  std::array<FormationMatches,2> reports{};
  std::set<FormationTemplate> seen;
  const size_t order=seed%library.size();
  for (size_t i=0;i<library.size();++i) {
    const auto& layout=library[(order+i)%library.size()];
    const bool exact=formation_fits(layout,history);
    auto& result=reports[exact?0:1];
    ++result.scanned;
    if (result.attempts>=16) continue;
    const auto candidate=exact?std::optional<FormationTemplate>(layout):repair_formation(layout,history);
    if (candidate && seen.insert(*candidate).second)
      try_formation(*candidate,history,result,mix_seed(seed,result.attempts),picked?&(*picked)[exact?0:1]:nullptr);
  }
  for (int type=0;type<2;++type) {
    const auto& r=reports[type];
    std::cout<<"F "<<key<<' '<<source<<' '<<type<<' '<<r.scanned<<' '<<r.attempts<<' '<<r.nodes<<' '
             <<r.matched<<' '<<r.bombs<<' '<<r.unresolved<<' '<<r.rejected<<' '<<r.replay_failed<<'\n';
  }
}

static int formation_slot(int node,int seat) {
  for (int i=0;i<kPiecesPerSeat;++i) if (layout_slot_node(seat,i)==node) return i;
  throw std::runtime_error("focal piece has no identifiable opening slot");
}

static void emit_formation_matches(const ArchiveRows& raw,const Rules& rules,const std::string& key,int viewer,
    int cut,uint64_t seed,const std::array<std::vector<FormationTemplate>,2>& banks,FormationWorlds* picked) {
  const auto prefix=coverage_prefix(raw,raw.layouts,rules,viewer,cut);
  const auto d=seat_decision(prefix.game.state);
  const auto& v=d.observation;
  const auto& game=prefix.game;
  const auto& actor=game.marks[viewer];
  const auto evidence=game.reply_history.for_actor(v,actor);
  const auto focus=directed_contact(d,evidence,rules,0);
  if (focus.kind!=0) return;
  const CoverageRoot root{raw,rules,prefix,d,evidence};
  const int square=focus.move.to,seat=v.seat_at(square);
  FormationHistory info{root,game.reply_history.opening_support(v,actor,prefix.history.front()),{},
                        formation_slot(game.reply_history.opening_node(square),seat)};
  for (int i=0;i<kPiecesPerSeat;++i) info.nodes[i]=layout_slot_node(seat,i);
  single_case_check(game.state.id_at[square]%kPiecesPerSeat==info.focal_slot,
                    "visible piece history did not recover the opening slot");
  const auto action=early_action_key(focus.move);
  std::cout<<"X "<<key<<' '<<action<<' '<<int(evidence.moved[square])<<' '
           <<int(game.state.rank_at[square]==Bomb)<<'\n';
  for (int i=0;i<2;++i)
    count_formations(key,i,banks[i],info,mix_seed(seed,i),i==0?picked:nullptr);
}

static void run_dark_formation_library(const char* rules_path,bool compare_sources) {
  StudentNetwork student;
  single_case_check(student.load(std::cin,true),"invalid formation audit student");
  std::array<std::vector<FormationTemplate>,2> banks;
  for (auto& library:banks) {
    int n;
    single_case_check(bool(std::cin>>n) && n>0 && n<=40000,"invalid formation template count");
    for (int i=0;i<n;++i) {
      int8_t layout[kPiecesPerSeat];
      for (auto& rank:layout) {
        int r;
        single_case_check(bool(std::cin>>r) && r>=Flag && r<RankCount,"invalid formation rank");
        rank=int8_t(r);
      }
      if (!valid_layout(layout)) continue;
      FormationTemplate row;
      std::copy_n(layout,kPiecesPerSeat,row.begin());
      library.push_back(row);
    }
    single_case_check(!library.empty(),"empty legal formation bank");
    std::cout<<"L "<<n<<' '<<library.size()<<'\n';
  }
  std::string key,hex,error;
  int viewer,cut,direction;
  uint64_t seed;
  while (std::cin>>key>>viewer>>cut>>direction>>seed>>hex) {
    Rules rules;
    single_case_check(viewer>=0 && viewer<kSeats && cut>0 && (direction==0 || direction==1) &&
                      parse_rules(archive_review_rules(read_text(rules_path),bool(direction)),rules,error),
                      "invalid formation archive input");
    const ArchiveRows raw(decode_archive_tape(hex));
    FormationWorlds picked;
    auto* samples=compare_sources?&picked:nullptr;
    emit_formation_matches(raw,rules,key,viewer,cut,seed,banks,samples);
    dark_budget_root(raw,rules,key,viewer,cut,seed,student,samples);
  }
  single_case_check(std::cin.eof(),"incomplete formation archive request");
}
