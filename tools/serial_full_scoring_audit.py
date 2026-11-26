"""Audit the actual full-mode selector and test a bounded reference search on matched roots."""
from collections import Counter
from itertools import product
import json
import os
from random import Random
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_full_mode_inheritance_confirm import STARTS, frozen_inputs
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

PRIOR=ROOT/"build/full-mode-inheritance-confirmation/confirm-third-inherited-student-on-fresh-full-openings-f2a6cbac447fbc55ab2a"
ROOT_KEYS="cut hash seat path reset inherited old_result new_result old_final new_final legal".split()
ACTION_KEYS="order objective position exposure terminal collision next".split()
ORDINARY=range(2862100,2862108)


def manifest():
    bad,controls=[],[]
    for start in STARTS:
        data=read(PRIOR/f"panel-{start}.json")
        get=lambda name:{r["seed"]:r for r in data[name]["rows"] if r["arm"]==4 and r["table"]==2}
        old,new=get("reset"),get("inherited")
        for seed,a in old.items():
            b=new[seed]
            expect=dict(seed=seed,cut=0,table=2,old_result=a["result"],new_result=b["result"],
                        old_final=a["final_hash"],new_final=b["final_hash"])
            if b["result"]<a["result"]:
                bad.append(expect)
            if a["result"]==b["result"]==2 and a["final_hash"]!=b["final_hash"]:
                controls.append(expect)
    rng=Random(624)
    ordinary=[dict(seed=seed,cut=(1,17,32,51)[i%4],table=2) for i,seed in enumerate(ORDINARY)]
    return [dict(x,group=name) for name,items in (("decline",rng.sample(bad,8)),
            ("kept-win",rng.sample(controls,8)),("ordinary",ordinary)) for x in items]


def verified_scope(folder):
    scope=FullModeRun(folder)
    stored=read(folder/"source.json")
    for file,digest in stored.items():
        require(sha(folder/file)==digest,"pinned scoring source changed")
    bundle=read(folder/"bundle.json")
    return scope," ".join(bundle["wire"].split()[:6])+" "+model_wire(read(folder/"reset.json"),"window")+" "+model_wire(
        read(folder/"inherited.json"),"window")


def prepare(folder):
    scope=FullModeRun(folder)
    frozen_inputs(scope)
    build(scope.engine)
    for name in ("reset.json","inherited.json","bundle.json"):
        require(sha(folder/name)==sha(PRIOR/name),"unrecognized reference weights")
    inputs=("engine/rules.json","engine/deadline_opening.exe","reset.json","inherited.json","bundle.json")
    write_receipt(folder/"source.json",{name:sha(folder/name) for name in inputs})
    write_receipt(folder/"manifest.json",manifest())


def run(scope,wire,tape):
    with isolated(scope,ROOT/"build"):
        return scope.run("full-scoring-audit",wire+"\n"+tape)


def student_rank(actions,order,fallback,model):
    if actions[fallback]["terminal"]==2:
        return fallback
    choice=fallback
    best=-1e9
    for action in order:
        score=actions[action][model]
        if score>best+1e-12:
            best,choice=score,action
    return choice


def collect(scope,wire,requests):
    roots={x["seed"]:dict(x,actions={},ablations={}) for x in requests}
    tape="\n".join(f"{x['seed']} {x['table']} {x['cut']} 0" for x in requests)
    for tag,seed,*v in map(str.split,run(scope,wire,tape).splitlines()):
        root=roots[int(seed)]
        if tag=="R":
            row=dict(zip(ROOT_KEYS,map(int,v),strict=True))
            for key in ("old_result","new_result","old_final","new_final"):
                require(key not in root or root[key]==row[key],"actual paired episode differs after reloading")
            root.update(row,request_cut=root["cut"])
        elif tag=="X":
            kind,choice,*members=map(int,v)
            root["ablations"][kind]=dict(choice=choice,list=members)
        elif tag=="A":
            action=int(v[0])
            row=dict(zip(ACTION_KEYS,map(int,v[1:8]),strict=True))
            row.update(zip(("reset","inherited","six","plies"),map(float,v[8:]),strict=True))
            root["actions"][action]=row
        else:
            require(tag=="F","unknown scoring output")
            action=int(v[0])
            require(len(v)==59,"wrong encoding width in live score trace")
            root["actions"][action].update(logits=list(map(float,v[1:7])),x=list(map(float,v[7:])))
    for root in roots.values():
        actions=root["actions"]
        require(len(actions)==root["legal"] and len(root["ablations"])==3 and
                all(k in actions and actions[k]["order"]>=0 for k in (root["reset"],root["inherited"])),
                "incomplete legal or shortlist attribution")
        require(root["request_cut"] or root["reset"]!=root["inherited"],"registered split did not diverge")
        shortlist=sorted((a for a in actions if actions[a]["order"]>=0),key=lambda a:actions[a]["order"])
        root["six"]=max(shortlist,key=lambda a:actions[a]["six"])
        for model in ("reset","inherited"):
            require(student_rank(actions,shortlist,root["path"],model)==root[model],
                    "numeric scoring trace no longer explains the actual shortlist ranking")
        root["all_student"]=student_rank(actions,shortlist+[a for a in actions if a not in shortlist],root["path"],"inherited")
    return roots


def branches(scope,wire,roots,choices,tables,streams,use_reset=False):
    rows=[]
    expected=set()
    for seed,actions in choices.items():
        r=roots[seed]
        requests=list(product(actions,tables,streams))
        expected.update((seed,*v) for v in requests)
        pairs=" ".join(f"{a} {t} {s}" for a,t,s in requests)
        rows.append(f"{seed} {r['table']} {r['request_cut']} {2 if use_reset else 1} {len(requests)} {pairs}")
    result={}
    steps=0
    for tag,seed,hash_,action,table,stream,outcome,plies in map(str.split,run(scope,wire,"\n".join(rows)).splitlines()):
        seed,action,table,stream,outcome,plies=map(int,(seed,action,table,stream,outcome,plies))
        key=(seed,action,table,stream)
        require(tag=="B" and key in expected and key not in result and int(hash_)==roots[seed]["hash"] and
                outcome in (-1,0,1,2) and plies>0,"branch trace drift")
        result[key]=outcome
        steps+=plies
    require(result.keys()==expected,"missing branch traces")
    return result,steps


def complete(outcomes,seed,action,table,streams):
    return mean(outcomes[seed,action,table,s] if outcomes[seed,action,table,s]>=0 else 1 for s in streams)


def discover(scope,wire,roots):
    early_streams=(19100,19101)
    deep_streams=range(19110,19114)
    universe={s:list(r["actions"]) for s,r in roots.items()}
    rows,work=branches(scope,wire,roots,universe,(2,),early_streams)
    finalist={}
    for s,r in roots.items():
        def rank(a):
            item=r["actions"][a]
            return item["terminal"]==2,complete(rows,s,a,2,early_streams),item["inherited"]
        ranked=sorted(universe[s],key=rank,reverse=True)
        r["flat_full"]=ranked[0]
        finalist[s]=list(dict.fromkeys([*ranked[:4],r["reset"],r["inherited"],r["six"]]))
    further,more=branches(scope,wire,roots,finalist,(2,),deep_streams)
    for s,r in roots.items():
        def rank(a):
            return (r["actions"][a]["terminal"]==2,complete(further,s,a,2,deep_streams),
                    r["actions"][a]["inherited"])
        r["reference"]=max(finalist[s],key=rank)
        r["reference_went_wide"]=r["actions"][r["reference"]]["order"]==-1
    return {"all_action_plies":work,"refinement_plies":more,"screening_runs":len(rows),"refinement_runs":len(further)}


def analyze(roots):
    groups={}
    for group in ("decline","kept-win","ordinary"):
        rows=[x for x in roots.values() if x["group"]==group]
        groups[group]=dict(
            n=len(rows),cut=[r["cut"] for r in rows],legal=sum(r["legal"] for r in rows),
            exact_win_miss=sum(any(a["terminal"]==2 for a in r["actions"].values()) and
                               r["actions"][r["inherited"]]["terminal"]!=2 for r in rows),
            terminal_loss_chosen=sum(r["actions"][r["inherited"]]["terminal"]==0 for r in rows),
            exposure_nonzero=sum(a["exposure"]!=0 for r in rows for a in r["actions"].values()),
            shortlist_changed=[sum(set(r["ablations"][i]["list"])!={a for a,x in r["actions"].items() if x["order"]>=0}
                                   for r in rows) for i in range(3)],
            ablation_changed=[sum(r["ablations"][i]["choice"]!=r["inherited"] for r in rows) for i in range(3)],
            wide_rank_changed=sum(r["all_student"]!=r["inherited"] for r in rows),
            reference_changed=sum(r["reference"]!=r["inherited"] for r in rows),
            reference_outside=sum(r["reference_went_wide"] for r in rows))
    return groups


def confirmation(scope,wire,old):
    roots=collect(scope,wire,[dict(row) for row in read(scope.folder/"manifest.json")])
    for seed,root in roots.items():
        previous=old[str(seed)]
        require(root["hash"]==previous["hash"] and root["cut"]==previous["cut"],"recheck root drift")
        for key in ("reference","flat_full","all_student","six"):
            root[key]=previous[key]
        for i in range(3):
            root[f"omit_{i}"]=root["ablations"][i]["choice"]
    arms=("reset","reference","flat_full","six","all_student","omit_0","omit_1","omit_2")
    choices={s:list(dict.fromkeys(r[k] for k in (*arms,"inherited"))) for s,r in roots.items()}
    streams=range(19200,19208)
    outcome,work=branches(scope,wire,roots,choices,(2,3),streams)
    paired,extra=branches(scope,wire,roots,choices,(2,),streams,use_reset=True)
    panels={}
    for name,data,table in (("natural",outcome,2),("changed_partner",outcome,3),("reset_continuation",paired,2)):
        panels[name]={}
        for group in ("all","decline","kept-win","ordinary"):
            subset={s:r for s,r in roots.items() if group=="all" or r["group"]==group}
            panels[name][group]={}
            for k in arms:
                differences=[]
                for s,r in subset.items():
                    rows=[(0.,0.) if r[k]==r["inherited"] else score_difference(data[s,r["inherited"],table,j],data[s,r[k],table,j])
                          for j in streams]
                    differences.append(tuple(map(mean,zip(*rows,strict=True))))
                panels[name][group][k]=contrast(differences)
    raw=[["path",*k,v] for k,v in outcome.items()]+[["reset",*k,v] for k,v in paired.items()]
    return dict(panels=panels,natural_runs=len(raw),work=work+extra,truncated=sum(r[-1]<0 for r in raw),outcomes=raw)


def main(audit):
    folder=ROOT/"build/full-scoring-audit"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        prepare(folder)
    scope,wire=verified_scope(folder)
    development=folder/"development.json"
    if audit:
        d=read(development)
        result=confirmation(scope,wire,d["roots"])
        write_receipt(folder/"confirmation.json",result)
        tables=[result["panels"][desk]["all"]["reference"]["interval"] for desk in ("natural","changed_partner","reset_continuation")]
        verdict="supported" if all(lo>0 for lo,_ in tables) else "refuted" if all(hi<0 for _,hi in tables) else "inconclusive"
        print(json.dumps(dict(intervals=tables,work=result["work"],branches=result["natural_runs"],
                              truncations=result["truncated"])))
        print("JUNQI_RESEARCH "+verdict)
    else:
        roots=collect(scope,wire,read(folder/"manifest.json"))
        search=discover(scope,wire,roots)
        groups=analyze(roots)
        write_receipt(development,dict(roots=roots,search=search,groups=groups))
        print(json.dumps(dict(search=search,groups=groups)))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
