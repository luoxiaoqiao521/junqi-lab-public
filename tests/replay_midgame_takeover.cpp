#include "../sim/archive_review.h"
#include "../sim/decision_branches.h"
#include "../sim/public_relay_screen.h"
#include "../sim/public_threat_evidence.h"
#include <map>
#include <array>
#include <iostream>

using namespace jq;

struct Timing {
  int contact=-1, enemy_loss=-1, enemy_exit=-1, own_exit=-1;
  int exchanges=0, quiet_run=0, max_quiet=0;
  void note(const StepInfo& event,int team,int elapsed) {
    if (event.outcome==Outcome::Move) {
      max_quiet=std::max(max_quiet,++quiet_run);
    } else {
      if (contact<0) contact=elapsed;
      ++exchanges;
      quiet_run=0;
    }
    const bool victim=event.outcome==Outcome::BothRemoved ||
        (event.outcome==Outcome::DefenderWins && team_of(event.seat)!=team) ||
        (event.outcome==Outcome::AttackerWins && team_of(event.seat)==team);
    if (victim && enemy_loss<0) enemy_loss=elapsed;
    for (int seat=0;seat<kSeats;++seat) if (event.eliminated_mask & (1u<<seat)) {
      auto& first=team_of(seat)==team?own_exit:enemy_exit;
      if (first<0) first=elapsed;
    }
  }
  void print() const {
    std::cout<<' '<<contact<<' '<<enemy_loss<<' '<<enemy_exit<<' '<<own_exit;
  }
};

struct Root {
  State state;
  Move human;
  Timing observed{};
  std::optional<OpeningMarks> history;
  std::vector<OpeningMarks> all_history{};
};

struct Compare {
  int decision=0, next=0, retired=0, revealed=0, clock=0, result=0;
  int first=0;
  unsigned external=0;
  void note(const State& trial,const State& archived,int sequence,bool legal) {
    const bool turn=trial.result==Result::Ongoing && archived.result==Result::Ongoing &&
                    trial.to_move!=archived.to_move;
    const bool exits=!std::equal(std::begin(trial.out),std::end(trial.out),std::begin(archived.out));
    const bool revealed_diff=!std::equal(std::begin(trial.flag_revealed),std::end(trial.flag_revealed),
                                         std::begin(archived.flag_revealed));
    const bool clocks=trial.no_progress!=archived.no_progress;
    const bool results=trial.result!=archived.result;
    if (!first && (!legal || turn || exits || revealed_diff || clocks || results)) first=sequence;
    decision+=!legal; next+=turn; retired+=exits; revealed+=revealed_diff; clock+=clocks; result+=results;
  }
  void print() const {
    std::cout<<' '<<decision<<' '<<next<<' '<<retired<<' '<<revealed<<' '<<clock<<' '<<result<<' '
             <<first<<' '<<external;
  }
};

struct Policies { ShortWeights parent{}, trained{}, peer{}; };

static Move select(const SeatDecision& d,int learner,int table,int mode,const Policies& models,Rng& rng) {
  if (d.observation.viewer==learner)
    return short_move(d.observation,d.actions,mode==0?models.parent:models.trained,rng);
  if (table==1) return short_move(d.observation,d.actions,models.peer,rng);
  return d.actions.at(rng.below(static_cast<uint32_t>(d.actions.size())));
}

#include "replay_curriculum_roots.h"
#include "replay_first_followthrough.h"
#include "replay_bomb_reply.h"
#include "replay_bomb_reserve.h"

static void trial(const std::string& key,const Root& root,Rules rules,int table,int rep,int mode,
                  const Policies& models,int cap) {
  auto state=root.state;
  const int learner=state.to_move;
  const int team=team_of(learner);
  auto rng=review_streams(mix_seed(std::stoull(key.substr(0,16),nullptr,16),
                                  uint64_t(8*state.ply+rep)));
  rules.truncation_cap=std::min(rules.truncation_cap,state.ply+cap);
  Timing trace;
  Move first{};
  EndReason ending=EndReason::None;
  int turns=0;
  while (state.result==Result::Ongoing) {
    const auto d=seat_decision(state);
    expect_value(!d.actions.empty(),"takeover generated a turn without legal actions");
    auto choice=select(d,learner,table,mode,models,rng[state.to_move]);
    if (state.to_move==learner) ++turns;
    if (state.ply==root.state.ply) {
      if (mode==2) choice=root.human;
      first=choice;
    }
    const auto e=step(state,choice,rules);
    ending=e.end_reason;
    trace.note(e,team,state.ply-root.state.ply);
  }
  std::cout<<"T "<<key<<' '<<root.state.ply<<' '<<table<<' '<<rep<<' '<<mode<<' '
           <<review_outcome(state.result,learner)<<' '<<state.ply-root.state.ply<<' '
           <<int(ending)<<' '<<int(first.from)<<' '<<int(first.to)<<' '
           <<int(root.state.seat_at[first.to]>=0)<<' '<<turns;
  trace.print();
  std::cout<<'\n';
  if (root.history)
    std::cout<<"Q "<<key<<' '<<root.state.ply<<' '<<table<<' '<<rep<<' '<<mode<<' '
             <<trace.exchanges<<' '<<trace.max_quiet<<'\n';
}

static void audit_history(const std::string& key,const Root& root,const Policies& models) {
  const auto d=seat_decision(root.state);
  const auto& marks=*root.history;
  int moved=0,restricted=0,singletons=0;
  for (int node=0;node<kNodes;++node) {
    if (d.observation.seat_at(node)<0 || d.observation.rank_at(node)!=Unknown) continue;
    const unsigned mask=marks.ranks[node];
    expect_value(mask && (mask & (1u<<root.state.rank_at[node])),
                 "public history excluded the actual hidden rank");
    moved+=marks.moved[node];
    restricted+=mask!=all_evidence_ranks;
    singletons+=(mask & (mask-1u))==0;
  }
  const auto human=std::find(d.actions.begin(),d.actions.end(),root.human);
  expect_value(human!=d.actions.end(),"recorded first move is missing from the legal candidates");
  const auto masses=review_learned(d,models.trained);
  const auto index=static_cast<size_t>(human-d.actions.begin());
  const auto human_features=short_features(d.observation,*human);
  int ties=0,better=0,relays=0;
  bool human_relay=false;
  std::map<ShortWeights,unsigned> quiet_roles;
  for (size_t i=0;i<d.actions.size();++i) {
    const auto action=d.actions[i];
    const auto features=short_features(d.observation,action);
    ties+=features==human_features;
    better+=masses[i]>masses[index];
    if (d.observation.seat_at(action.to)>=0) continue;
    const auto choices=public_relay_contacts(d,marks,action);
    const bool relay=std::any_of(choices.begin(),choices.end(),[](const auto& choice) {
      return choice.evidence==PublicRelayEvidence::NewContact;
    });
    relays+=relay;
    human_relay|=relay && i==index;
    quiet_roles[features]|=relay?2u:1u;
  }
  const auto aliases=std::count_if(quiet_roles.begin(),quiet_roles.end(),
                                   [](const auto& entry) { return entry.second==3u; });
  std::cout<<"P "<<key<<' '<<root.state.ply<<' '<<moved<<' '<<restricted<<' '<<singletons
           <<' '<<d.actions.size()<<' '<<std::llround(masses[index]*1000000)<<' '<<better<<' '
           <<ties<<' '<<relays<<' '<<aliases<<' '<<human_relay<<'\n';
}

static void run_root(const std::string& key,const Root& root,const Rules& rules,const Policies& models,int cap) {
  if (root.history) audit_history(key,root,models);
  const auto d=seat_decision(root.state);
  const auto contacts=std::count_if(d.actions.begin(),d.actions.end(),[&](Move m) {
    return d.observation.seat_at(m.to)>=0;
  });
  const auto known_flag=std::count_if(d.actions.begin(),d.actions.end(),[&](Move m) {
    return d.observation.rank_at(m.to)==Flag;
  });
  const auto masses=review_learned(d,models.trained);
  double contact_mass=0;
  for (size_t i=0;i<d.actions.size();++i) if (d.observation.seat_at(d.actions[i].to)>=0) contact_mass+=masses[i];
  std::cout<<"R "<<key<<' '<<root.state.ply<<' '<<int(root.state.to_move)<<' '
           <<root.state.no_progress<<' '<<d.actions.size()<<' '<<contacts<<' '<<known_flag<<' '
           <<int(root.human.from)<<' '<<int(root.human.to)<<' '
           <<int(root.state.seat_at[root.human.to]>=0)<<' '<<std::llround(contact_mass*1000000)<<' '
           <<std::count(std::begin(root.state.out),std::end(root.state.out),true)<<'\n';
  for (int table : {0,1}) for (int rep=0;rep<4;++rep) for (int mode : {0,1,2})
    trial(key,root,rules,table,rep,mode,models,cap);
}

static int nibble(char x) {
  if (x>='0' && x<='9') return x-'0';
  if (x>='a' && x<='f') return x-'a'+10;
  throw std::runtime_error("invalid tape hex");
}

static std::string tape_bytes(const std::string& tape) {
  expect_value(tape.size()%2==0,"invalid tape length");
  std::string input;
  for (size_t i=0;i<tape.size();i+=2)
    input+=static_cast<char>(16*nibble(tape[i])+nibble(tape[i+1]));
  return input;
}

static unsigned external_control(const ArchiveRows& raw,size_t row) {
  const int kind=raw.byte(row+1), cause=raw.byte(row+3);
  return kind==2 || kind==4 || (kind==3 && cause!=1 && cause!=2);
}

static void recorded(const std::string& key,const std::string& tape,Rules rules,Rules self,
                     const Policies& models,int cap,bool restore_history,bool curriculum,int followthrough,
                     int bomb_replies) {
  ArchiveReview review;
  BombReplies bombs;
  std::ostringstream bomb_study;
  std::vector<Root> roots;
  Compare archive_diff,self_diff;
  int seq=0, verified=0;
  bool failed=false, missing_end=false;
  try {
    ArchiveRows raw(tape_bytes(tape));
    init_state(review.state,raw.layouts,archive_first_seat(raw),rules);
    std::vector<OpeningMarks> marks;
    if (restore_history)
      for (int seat=0;seat<kSeats;++seat) marks.emplace_back(observe(review.state,seat),true);
    int index=0;
    while (index<raw.entries && raw.byte(raw.row(index))==245) {
      seq=index+1;
      review.control(raw,raw.row(index),seq,rules);
      ++index;
      ++archive_diff.external;
      ++self_diff.external;
    }
    while (index<raw.entries && !review.ended) {
      seq=index+1;
      const auto row=raw.row(index++);
      expect_value(raw.byte(row)==95,"expected recorded move");
      auto archived=review.state;
      const int from=raw.point(row+2), to=raw.point(row+4);
      expect_value(from>=0 && to>=0,"invalid archive move coordinates");
      const Move move{static_cast<uint8_t>(from),static_cast<uint8_t>(to)};
      review.play(raw,row,seq,rules);
      if (bomb_replies==1) bombs.take_turn(archived,move,models);
      if (bomb_replies>=2)
        bombs.take_turn(archived,move,models,[&](const BombWindow& w,const SeatDecision& d,int target) {
          bomb_reserve_root(bomb_study,key,archived,w,d,target,marks[d.observation.viewer],rules,models,cap,
                            bomb_replies==3?2:0);
        });
      review.frames.str("");
      if (!bomb_replies && (archived.ply==96 || archived.ply==144 || (curriculum && archived.ply==48)))
        roots.push_back({archived,move,{},restore_history?std::optional<OpeningMarks>(marks[archived.to_move]):std::nullopt,
                         curriculum?marks:std::vector<OpeningMarks>{}});
      unsigned external=0;
      while (index<raw.entries && raw.byte(raw.row(index))==245) {
        const auto control=raw.row(index);
        external+=external_control(raw,control);
        review.control(raw,control,++index,rules);
      }
      if (bomb_replies) bombs.after(archived,move,review.last.outcome);
      for (int seat=0;restore_history && seat<kSeats;++seat)
        advance_contact_ranks(marks[seat],observe(archived,seat),observe(review.state,seat),
                              move,review.last.outcome,rules);
      for (auto& root : roots)
        root.observed.note(review.last,team_of(root.state.to_move),review.state.ply-root.state.ply);
      auto alternate=archived;
      const bool legal=is_legal(alternate,move);
      if (legal) {
        step(archived,move,rules);
        step(alternate,move,self);
      }
      archive_diff.note(archived,review.state,seq,legal);
      self_diff.note(alternate,review.state,seq,legal);
      archive_diff.external+=external;
      self_diff.external+=external;
      ++verified;
    }
    missing_end=!review.ended;
    expect_value(review.captured_flag==kNone,"missing flag control");
    if (bomb_replies==1 && !missing_end) {
      bombs.finish(review.state,key,0);
      for (int source:{1,2}) simulated_bomb_replies(raw,key,rules,models,source);
    }
    if (bomb_replies>=2 && !missing_end) std::cout<<bomb_study.str();
  } catch (const std::exception& error) {
    failed=true;
    std::cerr<<key<<' '<<seq<<' '<<error.what()<<'\n';
  }
  std::cout<<"D "<<key<<' '<<verified<<' '<<seq<<' '<<failed<<' '<<missing_end;
  archive_diff.print();
  self_diff.print();
  std::cout<<'\n';
  for (const auto& root : roots) {
    const auto result=failed || missing_end ? Result::Truncated : review.state.result;
    if (curriculum) {
      try {
        curriculum_root(key,root,rules,models,review_outcome(result,root.state.to_move));
      } catch (const std::exception& error) {
        std::cout<<"X "<<key<<' '<<root.state.ply<<'\n';
        std::cerr<<key<<' '<<root.state.ply<<' '<<error.what()<<'\n';
        continue;
      }
      if (followthrough)
        for (int table=followthrough-1;table<=followthrough;++table)
          for (int rep:{0,1}) for (bool human:{false,true}) for (bool alternate:{false,true})
          replay_follow_case(key,root,rules,models,cap,table,rep,human,alternate);
      continue;
    }
    std::cout<<"H "<<key<<' '<<root.state.ply<<' '<<review_outcome(result,root.state.to_move)<<' '
             <<verified-root.state.ply;
    root.observed.print();
    std::cout<<'\n';
    if (root.history)
      std::cout<<"Q "<<key<<' '<<root.state.ply<<" -1 0 0 "<<root.observed.exchanges<<' '
               <<root.observed.max_quiet<<'\n';
    run_root(key,root,rules,models,cap);
  }
}

int main(int argc,char** argv) {
  try {
    if (argc<3 || argc>5) return 2;
    const std::string mode=argc==5 ? argv[4] : "";
    const int followthrough=mode=="curriculum-pressure"?2:mode=="curriculum-followthrough"?1:0;
    const bool curriculum=followthrough || mode=="curriculum-roots";
    const int bomb_replies=mode=="bomb-reserve-resample"?3:mode=="bomb-reserve"?2:mode=="bomb-replies"?1:0;
    const bool restore_history=curriculum || mode=="public-history" || bomb_replies>=2;
    if (argc==5 && !restore_history && !bomb_replies) return 2;
    const int cap=argc==3?240:std::stoi(argv[3]);
    if (cap!=240 && cap!=960) return 2;
    Rules archive,self;
    std::string error;
    if (!parse_rules(archive_review_rules(read_text(argv[1]),false),archive,error) ||
        !parse_rules(read_text(argv[2]),self,error)) throw std::runtime_error(error);
    self.ccw=archive.ccw;
    Policies models;
    for (auto weights : {&models.parent,&models.trained,&models.peer})
      for (size_t i=0;i<5;++i) if (!(std::cin>>(*weights)[i]) || !std::isfinite((*weights)[i])) return 2;
    std::string key,tape;
    while (std::cin>>key>>tape) recorded(key,tape,archive,self,models,cap,restore_history,curriculum,followthrough,bomb_replies);
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
