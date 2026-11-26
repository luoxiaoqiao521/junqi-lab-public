"""Reproduce stratified takeovers and test a single public decision at a time."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
import random
from statistics import median
import sys

from cross_partner_deadline_student import encode, persist
from measure_deadline_opening import ROOT, SCORE, build, model_roster, require
from measure_human_winner_takeover import CUTS, END_FIELDS, MODEL, source_frames
from train_early_deadline_student import model_arguments, native

PRIOR = (ROOT/"build/human-winner-takeover/measure-human-winner-takeover-cb07059a3f34683e3c18"
         /"summary.json")
SEED = 72411
ROLES = ("visible_finish", "forced_guard", "quiet_forward")
TURN = ("at quiet move pressure proof missed threat guard advance quiet_move reversal enemies").split()
END = ("plies outcome trace first_enemy last_enemy max_quiet reverse").split()
RESULT = {-1:"cutoff",0:"loss",1:"draw",2:"win"}


def select(previous):
    randomized=random.Random(SEED)
    pool=[]
    for r in previous["cases"]:
        family=r["source"]
        if previous["manifest"][family]["pilot"]:
            continue
        for k,b in r["runs"].items():
            rep, arm=map(int,k.split("/"))
            if arm!=0:
                continue
            phase=r["cut"]
            gap=b["plies"]-r["human_left"]
            bucket=("comeback" if phase=="comeback" else
                    "loss" if b["outcome"]==0 else "draw" if b["outcome"]==1 else
                    "slow" if b["outcome"]==2 and gap>=40 else
                    "fast" if b["outcome"]==2 and gap<=0 else "other")
            pool.append(dict(source=family, phase=phase, rep=rep, bucket=bucket,
                             at=r["ply"], human=r["human_left"], recorded=b))
    randomized.shuffle(pool)
    selected={}
    chosen_families=set()
    strata=[("fast",phase,2) for phase in CUTS[1:3]]
    strata += [("slow",phase,3) for phase in CUTS[:3]]
    strata += [(bucket,phase,1) for bucket in ("loss","draw") for phase in CUTS[:3]]
    strata += [("comeback","comeback",4)]
    coverage={}
    for bucket,phase,budget in strata:
        counts=0
        for row in pool:
            if row["bucket"]!=bucket or row["phase"]!=phase:
                continue
            source=row["source"]
            family=previous["manifest"][source]["family"]
            if family in chosen_families:
                continue
            selected[source]=row
            chosen_families.add(family)
            counts+=1
            if counts==budget:
                break
        coverage[f"{bucket}/{phase}"]=[counts,budget]
    require(sum(r["bucket"]=="slow" for r in selected.values())>=6,"too few distinct slow games")
    return selected,coverage


def parse(stdout, selected):
    cases=defaultdict(lambda:dict(turns=[],forks={}))
    for line in stdout.splitlines():
        tag,key,*values=line.split()
        require(key in selected,"unexpected sampled source")
        values=list(map(int,values))
        row=cases[key]
        if tag=="T":
            row["turns"].append(dict(zip(TURN,values,strict=True)))
        elif tag=="E":
            require("end" not in row,"duplicate baseline")
            row["end"]=dict(zip(END,values,strict=True))
        elif tag=="P":
            kind,at,a,b,listed=values
            require(kind in range(len(ROLES)) and ROLES[kind] not in row["forks"] and a!=b,
                    "invalid single decision")
            row["forks"][ROLES[kind]]=dict(at=at,base=a,new=b,listed=listed,branches={})
        else:
            kind,rep,arm,*outcome=values
            branches=row["forks"][ROLES[kind]]["branches"]
            require(tag=="V" and f"{rep}/{arm}" not in branches,"duplicate branch")
            branches[f"{rep}/{arm}"]=dict(zip(END_FIELDS,outcome,strict=True))
    require(cases.keys()==selected.keys(),"a selected game was not replayed")
    for key,row in cases.items():
        validate(row,selected[key]["recorded"])
    return dict(cases)


def validate(row, recorded):
    original=row["end"]
    require(all(original[k]==recorded[k] for k in END),"historical replay changed")
    turns={t["at"]:t for t in row["turns"]}
    require(0 in turns and len(turns)==len(row["turns"]) and max(turns)<original["plies"],
            "invalid or incomplete turn trace")
    for name,fork in row["forks"].items():
        t=turns[fork["at"]]
        require(t["move"]==fork["base"],"fork did not preserve the actual baseline choice")
        arms=fork["branches"]
        require(set(arms)=={f"{i}/{j}" for i in range(3) for j in range(2)},"missing counterfactual")
        base=arms["0/0"]
        require(base["trace"]==original["trace"] and base["outcome"]==original["outcome"]
                and base["plies"]+fork["at"]==original["plies"],"prefix plus control failed exact replay")
        for i in range(3):
            a,b=arms[f"{i}/0"],arms[f"{i}/1"]
            require(a["first"]==fork["base"] and b["first"]==fork["new"] and a["plies"]>0 and b["plies"]>0
                    and a["outcome"] in SCORE and b["outcome"] in SCORE,"counterfactual not played")
            if name=="visible_finish":
                require(b["outcome"]==2 and b["plies"]==1 and t["missed"]==1,"public terminal proof failed")


def describe_case(row, meta):
    turns=row["turns"]
    end=row["end"]
    wins=[t["at"]+1 for t in turns if t["proof"]]
    first_win=wins[0] if wins else None
    result=dict(bucket=meta["bucket"],phase=meta["phase"],rep=meta["rep"],human=meta["human"],
                at=meta["at"],baseline=end["plies"],outcome=RESULT[end["outcome"]],
                first_enemy=end["first_enemy"],last_enemy=end["last_enemy"],
                first_visible_win=first_win,missed=sum(t["missed"] for t in turns),
                guard=sum(t["guard"] for t in turns),reroute=sum(t["advance"] for t in turns),
                quiet=max(t["quiet"] for t in turns),own_reverse=sum(t["reversal"] for t in turns),
                second_enemy_after_first=(end["last_enemy"]-end["first_enemy"]
                                          if 0<end["first_enemy"]<end["last_enemy"] else None))
    result["avoidably_delayed"]=end["plies"]-first_win if first_win and end["outcome"]==2 else None
    return result


def describe_branches(cases):
    result={}
    for role in ROLES:
        selected=[(k,row["forks"][role]) for k,row in cases.items() if role in row["forks"]]
        tally=Counter()
        win_speed=[]
        for key,fork in selected:
            for rep in range(3):
                a,b=(fork["branches"][f"{rep}/{arm}"] for arm in range(2))
                tally[f'{RESULT[a["outcome"]]}->{RESULT[b["outcome"]]}']+=1
                if a["outcome"]==b["outcome"]==2:
                    win_speed.append(b["plies"]-a["plies"])
        result[role]=dict(sources=len(selected),listed=sum(x["listed"] for _,x in selected),
                          transitions=dict(tally),paired_win_median=median(win_speed) if win_speed else None)
    return result


def main(audit):
    directory=ROOT/"build/takeover-delay"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    previous=json.loads(PRIOR.read_bytes())
    selected,coverage=select(previous)
    sources,_=source_frames()
    roster=model_roster()
    weights=[w for n in ("candidate","historical") for w in roster[n]["weights"]]
    params=" ".join(map(str,(1,1,*weights)))+" "+model_arguments(json.loads(MODEL.read_bytes()),"replay_control")
    payload=[params]
    for key,row in sorted(selected.items()):
        tape=sources[key]["tape"]
        require(sha256(bytes.fromhex(tape)).hexdigest()==previous["manifest"][key]["tape_sha256"],
                "source tape changed")
        payload.append(f'{key} {CUTS.index(row["phase"])} {row["rep"]} {tape}')
    cases=parse(native(directory,"winner-delay-audit","\n".join(payload)),selected)
    summaries={key:describe_case(row,selected[key]) for key,row in cases.items()}
    forks=describe_branches(cases)
    result=dict(seed=SEED,coverage=coverage,cases=cases,summaries=summaries,forks=forks)
    persist(directory/"summary.json",encode(result),audit)
    print(json.dumps(dict(coverage=coverage,forks=forks),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
