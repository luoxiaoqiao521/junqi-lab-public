"""Reuse existing seat-order nominations and compare one additional shortlist slot."""
from collections import Counter
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation, model_wire
from serial_dark_partner_support import SOURCE
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


PRIOR=ROOT/"build/dark-partner-evidence/audit-four-dark-partner-resource-consumption-verified-72c1c7a0a84803011017"
ARMS=("four","static","seat-order","public-contact")
INPUTS=("parent.json","grid.json","expected.json","engine/deadline_opening.exe","engine/rules.json")


def prepare(scope,action_level,force_contact):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(SOURCE/"original.json",folder/"parent.json")
    verified=read(PRIOR/"grid.json")["heldout"]
    prior=read(PRIOR/"heldout.json")["rows"]
    known=[]
    expected=[]
    for key,row in zip(verified,prior,strict=True):
        require(list(key[:3])==[row[n] for n in ("seed","seat","table")],"historical root receipts disagree")
        wins=[a for a in row["actions"] if a["label"]==1]
        losses=[a for a in row["actions"] if a["label"]==0]
        if wins and losses:
            require(not wins[0]["listed"],"positive historical check must be a missed candidate")
            known.append(key)
            expected.append(dict(seed=row["seed"],ply=row["ply"],base=row["before"],target=wins[0]["move"]))
    require(len(known)==2,"historical candidate checks changed")
    start=24169000 if force_contact else 24168000 if action_level else 24167000
    grid=dict(known=known,fresh=[(start+i,i%4,(i//4)%2,(24,48,72,96)[(i//8)%4])
                                for i in range(96 if force_contact else 64)])
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"expected.json",expected)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def decode(raw,grid,force_contact):
    out=[]
    for tag,*values in map(str.split,raw.splitlines()):
        data=list(map(int,values))
        if tag=="F":
            seed,seat,table,ply,valid=data
            out.append(dict(seed=seed,seat=seat,table=table,ply=ply,valid=valid,
                            lists=[],choices=[],contacts=[],games={}))
        elif tag=="W":
            steps,worlds,rejected,isolation=data
            out[-1].update(steps=steps,worlds=worlds,rejected=rejected,isolation=isolation)
        elif tag=="A":
            arm,choice,width,*moves=data
            require(arm==len(out[-1]["lists"]) and width==len(set(moves)) and choice in moves,
                    "invalid shortlist or selected action")
            out[-1]["lists"].append(moves)
            out[-1]["choices"].append(choice)
        elif tag=="C":
            out[-1]["contacts"].append(data[0])
        else:
            require(tag=="G","unknown nomination receipt")
            move,desk,rep,result,spent=data
            out[-1]["games"][move,desk,rep]=[result,spent]
    require([(r["seed"],r["seat"],r["table"]) for r in out]==[tuple(k[:3]) for k in grid],
            "nomination assignments changed")
    for r in out:
        if not r["valid"]:
            require(not r["lists"] and not r["games"],"unrestored nomination performed work")
            continue
        a,b,c,d=r["lists"]
        require(all(x[:len(a)]==a for x in (b,c,d)) and len(a)<=4 and
                len(b)==len(c)==len(d)<=len(a)+1,"historical reuse changed the original shortlist or added extra slots")
        choices={r["choices"][i] for i in (0,3)} if force_contact else set(r["choices"])
        reps=8 if force_contact else 3
        expected=set(product(choices,(r["table"],3),range(reps))) if len(choices)>1 else set()
        require(r["games"].keys()==expected and all(res in (-1,0,1,2) and n>0 for res,n in r["games"].values()),
                "same-root games are incomplete")
    return out


def outcomes(r,old,new,table,reps=3):
    if old==new:
        return (0,0)
    differences=[score_difference(r["games"][old,table,i][0],r["games"][new,table,i][0]) for i in range(reps)]
    return tuple(map(mean,zip(*differences,strict=True)))


def evaluate(rows):
    counts=Counter()
    cases={n:Counter() for n in ARMS}
    scores={n:[] for n in ARMS[1:]}
    rivals={n:[] for n in ARMS[1:]}
    for r in rows:
        counts["assigned"]+=1
        if not r["valid"]:
            for n in ARMS[1:]:
                scores[n].append((0,0))
                rivals[n].append((0,0))
            continue
        counts["restored"]+=1
        counts["isolation"]+=r["isolation"]
        counts["covered"]+=r["worlds"]==2
        counts["steps"]+=r["steps"]
        counts["rejected"]+=r["rejected"]
        contacts=set(r["contacts"])
        counts["known_contact_roots"]+=bool(contacts)
        counts["legal_proved_contacts"]+=len(contacts)
        missing=contacts-set(r["lists"][0])
        counts["missing_contact_roots"]+=bool(contacts) and not (contacts&set(r["lists"][0]))
        old=r["choices"][0]
        for n,shortlist,pick in zip(ARMS,r["lists"],r["choices"],strict=True):
            cases[n]["listed"]+=bool(contacts&set(shortlist))
            cases[n]["actions_covered"]+=len(contacts&set(shortlist))
            cases[n]["new_proved_actions"]+=len(missing&set(shortlist))
            cases[n]["new_proved_action_selected"]+=pick in missing
            cases[n]["picked"]+=pick in contacts
            cases[n]["changed"]+=pick!=old
            if n!="four":
                scores[n].append(outcomes(r,old,pick,r["table"]))
                rivals[n].append(outcomes(r,old,pick,3))
    return dict(counts=counts,arms=cases,same_table={n:contrast(v) for n,v in scores.items()},
                new_enemies={n:contrast(v) for n,v in rivals.items()})


def evaluate_forced(rows):
    counts=Counter()
    same,changed,two=[],[],[]
    for r in rows:
        counts["assigned"]+=1
        counts["restored"]+=r["valid"]
        counts["covered"]+=r.get("worlds",0)==2
        counts["isolation"]+=r.get("isolation",0)
        if not r["valid"] or r["choices"][0]==r["choices"][3]:
            same.append((0,0))
            changed.append((0,0))
            continue
        a,b=r["choices"][0],r["choices"][3]
        require(b in r["contacts"] and b not in r["lists"][0],"forced move is not an omitted proved collision")
        counts["forced"]+=1
        comparisons=[outcomes(r,a,b,t,8) for t in (r["table"],3)]
        same.append(comparisons[0])
        changed.append(comparisons[1])
        two.append(dict(seed=r["seed"],old=a,proposed=b,same=comparisons[0],different=comparisons[1]))
        counts["wins_both_tables"]+=all(lo>0 for lo,hi in comparisons)
        counts["loses_both_tables"]+=all(hi<0 for lo,hi in comparisons)
        counts["direction_reversal"]+=any(lo>0 for lo,hi in comparisons) and any(hi<0 for lo,hi in comparisons)
        for desk in (r["table"],3):
            for i in range(8):
                x=r["games"][a,desk,i][0]
                y=r["games"][b,desk,i][0]
                counts["from_win_to_loss"]+=x==2 and y==0
                counts["from_loss_to_win"]+=x==0 and y==2
    return dict(counts=counts,same_table=contrast(same),new_enemies=contrast(changed),examples=two)


def verify_known(folder,rows):
    for e,r in zip(read(folder/"expected.json"),rows,strict=True):
        require(r["seed"]==e["seed"] and r["ply"]==e["ply"] and e["target"] in r["contacts"]
                and r["choices"][0]==e["base"] and e["target"] not in r["lists"][0],
                "historical missed nomination did not reproduce")


def main(audit,action_level=False,force_contact=False):
    scope=FullModeRun(ROOT/"build/dark-heritage-nominations"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope,action_level,force_contact)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"heritage inputs changed")
    phase="fresh" if audit else "known"
    grid=read(folder/"grid.json")[phase]
    model=read(folder/"parent.json")
    started=perf_counter()
    rows=[]
    for i in range(0,len(grid),8):
        batch=grid[i:i+8]
        wire=[model_wire(model),*(" ".join(map(str,x)) for x in batch)]
        with isolated(scope,ROOT/"build"):
            mode="dark-heritage-forced-contact" if force_contact else (
                "dark-heritage-action-nominations" if action_level else "dark-heritage-nominations")
            raw=scope.run(mode,"\n".join(wire)+"\n")
        rows+=decode(raw,batch,force_contact)
    result=evaluate_forced(rows) if force_contact else evaluate(rows)
    if audit:
        require(result["counts"]["covered"] and result["counts"]["isolation"],
                "actual historical nomination did not run with a hidden swap control")
        which="public-contact" if action_level else "seat-order"
        panel=lambda n:result[n] if force_contact else result[n][which]
        interval=panel("new_enemies")["interval"]
        actual=panel("same_table")["interval"]
        verdict=("supported" if min(interval[0],actual[0])>0 else "refuted" if max(interval[1],actual[1])<0
                 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    else:
        verify_known(folder,rows)
        expected=read(folder/"expected.json")
        result["exact_historical_targets"]={n:sum(e["target"] in r["lists"][i]
            for e,r in zip(expected,rows,strict=True)) for i,n in enumerate(ARMS)}
        result["exact_historical_selections"]={n:sum(e["target"]==r["choices"][i]
            for e,r in zip(expected,rows,strict=True)) for i,n in enumerate(ARMS)}
    serial=[dict(r,games=[(*k,*v) for k,v in r["games"].items()]) for r in rows]
    write_receipt(folder/(phase+".json"),dict(summary=result,rows=serial,seconds=perf_counter()-started))
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected historical nomination stage")
    main(bool(sys.argv[1:]))
