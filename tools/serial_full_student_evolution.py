"""Train one frozen-parent successor, then compare both takeover schedules."""
from collections import Counter
import hashlib
from itertools import product
import json
import os
import sys

from compare_book_learning import OUTPUTS, fit, np
from serial_dynamic_reply import PRIOR, ROOT, build, frozen_wire, native, require
from serial_formation_matrix import contrast, score_difference
from serial_reply_window import examples, judge, model_wire, split
from serial_worker import read, write_receipt

CASE = ROOT/"build/student-candidate-coverage/compare-bounded-student-candidate-coverage-a130c7d9bd147ba326f7/confirmation.json"
STARTS = (2823000, 2825000)
GAME_KEYS = (
    "seed table arm plies result turns used changed steps micros first_change engineer_move engineer_exit "
    "engineer_loss first_exit exit_seat exit_reason first_enemy support_open support_used final_hash"
).split()
COMPARISONS = {"widen_parent_schedule": (1,3), "learn_in_windows": (1,2),
               "learn_continuously": (3,4), "child_vs_route": (0,4)}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def classroom(folder, wire):
    previous = read(CASE)["rows"]
    manifest = [(s,(1,32,70,112)[i%4],i%2,0,-1,-1) for i,s in enumerate(range(2821000,2821030))]
    for seed,table in ((2817807,1),(2817821,0)):
        old,new = (next(r for r in previous if (r["seed"],r["table"],r["arm"])==(seed,table,a))
                   for a in (1,6))
        require((old["first"],old["hash"])==(new["first"],new["hash"]), "registered comparison is not matched")
        require(old["new"]!=new["new"], "registered substitutes agree")
        manifest.append((seed,old["first"],table,old["hash"],old["new"],new["new"]))
    payload = "\n".join(" ".join(map(str,row)) for row in manifest)
    trace = native(folder,"ordinary-reply-classroom",wire+"\n"+payload)
    roots = {}
    missed = {}
    for tag,key,*parts in map(str.split,trace.splitlines()):
        if tag=="M":
            missed[key] = int(parts[0])
        elif tag=="R":
            root = dict(zip(("cut","hash","original","alternative","student","table","count"),map(int,parts),strict=True))
            roots[key] = dict(**root, features={}, games=[])
        elif tag=="F":
            action,*features = parts
            x = list(map(float,features))
            require(len(x)==52 and np.isfinite(x).all(), "incomplete visible features")
            roots[key]["features"][action] = x
        else:
            require(tag=="G","invalid classroom output")
            row=dict(zip(("cut","arm","table","stream","action","result","plies","own_flag","hash"),map(int,parts),strict=True))
            roots[key]["games"].append(row)
    require(set(roots)|set(missed)=={str(row[0]) for row in manifest}, "incomplete decision manifest")
    for r in roots.values():
        expected = {(int(a),t,s) for a,t,s in product(r["features"],range(2),range(6))}
        require(len(r["features"])==r["count"] and len(r["games"])==len(expected) and
                {(g["action"],g["table"],g["stream"]) for g in r["games"]}==expected and
                all(g["cut"]==r["cut"] and g["result"] in (-1,0,1,2) for g in r["games"]), "incomplete common-stream branch matrix")
        if r["original"]!=r["alternative"]:
            pair={(g["action"],g["table"],g["stream"]):g["result"] for g in r["games"]}
            r["registered_contrast"]={str(t):contrast([score_difference(pair[r["original"],t,s],pair[r["alternative"],t,s])
                                                       for s in range(6)]) for t in range(2)}
    return dict(roots=roots,missed=missed,trace_sha256=digest(trace),manifest=manifest)


def train(classroom_data, anchors, parent):
    rows = []
    admitted = []
    for key,root in classroom_data["roots"].items():
        if root["count"]<2 or any(g["result"]<0 for g in root["games"]):
            continue
        for action,x in root["features"].items():
            outcomes = Counter(g["result"] for g in root["games"] if g["action"]==int(action))
            rows.append((x,[outcomes[i]/12 for i in range(3)]))
        admitted.append(key)
    require(len(admitted)>=16,"insufficient development roots with complete outcomes")
    ordinary_rows = len(rows)
    for lesson in anchors:
        for x,label in zip(lesson["features"],lesson["scores"],strict=True):
            rows.append((x,[int(i==label) for i in range(3)]))
    x = np.asarray([features for features,_ in rows])
    x = np.clip((x-parent["mean"])/parent["scale"],-6,6)
    y = np.full((len(rows),len(OUTPUTS)),np.nan)
    y[:,:3] = [labels for _,labels in rows]
    parameters = fit(x,y,47,24,parent["parameters"],updates=60,rate=.04)
    return dict(mean=parent["mean"],scale=parent["scale"],parameters=parameters),dict(
        updates=60,learning_rate=.04,development_roots=admitted,ordinary_action_rows=ordinary_rows,anchor_pairs=len(anchors))


def games(folder,parent,child,start):
    payload = f"{start} 16 "+model_wire(parent,"window")+" "+model_wire(child,"window")
    rows = []
    for tag,*fields in map(str.split,native(folder,"full-student-evolution-games",payload).splitlines()):
        require(tag=="G","invalid full opening output")
        rows.append(dict(zip(GAME_KEYS,map(int,fields),strict=True)))
    require([(r["seed"],r["table"],r["arm"]) for r in rows]==
            list(product(range(start,start+16),range(3),range(5))),"incomplete full opening grid")
    by_game = {(r["seed"],r["table"],r["arm"]):r for r in rows}
    for r in rows:
        arm = r["arm"]
        ref = by_game[r["seed"],r["table"],0]
        require(0<=r["changed"]<=r["used"]<=r["turns"] and r["result"] in (-1,0,1,2) and
                (r["used"]==r["turns"] if arm>=3 else r["used"]<=(2 if arm else 0)),"bad learner coverage")
        require(r["changed"] or r["final_hash"]==ref["final_hash"],"no-change episode diverged")
    cells = {}
    differences = {}
    for table in range(3):
        for arm in range(5):
            games = [r for r in rows if r["table"]==table and r["arm"]==arm]
            cells[f"{table}/{arm}"] = dict(
                outcomes=dict(Counter(str(r["result"]) for r in games)),
                mean_plies=sum(r["plies"] for r in games)/len(games),
                wins100=sum(r["result"]==2 and r["plies"]<=100 for r in games),
                wins140=sum(r["result"]==2 and r["plies"]<=140 for r in games),
                losses140=sum(r["result"]==0 and r["plies"]<=140 for r in games),
                **{k:sum(r[k] for r in games) for k in ("turns","used","changed","micros","support_open","support_used")})
        for key,(old,new) in COMPARISONS.items():
            pairs=[]
            for seed in range(start,start+16):
                a,b = (by_game[seed,table,arm] for arm in (old,new))
                pairs.append((0.,0.) if a["final_hash"]==b["final_hash"] else score_difference(a["result"],b["result"]))
            differences[f"{table}/{key}"] = dict(**contrast(pairs),better=sum(lo>0 for lo,_ in pairs),
                                                  worse=sum(hi<0 for _,hi in pairs))
    return dict(rows=rows,cells=cells,differences=differences)


def main(audit):
    folder=ROOT/"build/full-student-evolution"/os.environ["JUNQI_SERIAL_KEY"]
    frozen=read(PRIOR/"students.json")
    parent=frozen["models"]["window"]
    wire,peer=frozen_wire()
    parent_hash=digest(model_wire(parent,"window"))
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
        data=classroom(folder,wire)
        write_receipt(folder/"classroom.json",data)
        cases,_,source=examples(folder)
        anchors,old_suite=split(cases)
        child,training=train(data,anchors,parent)
        artifact=dict(models={"board":frozen["models"]["board"],"window":child},
                      training=training,parent_sha256=parent_hash,peer=peer)
        write_receipt(folder/"student.json",artifact)
        reloaded=read(folder/"student.json")
        require(reloaded==artifact,"saved learner did not round-trip")
        old=judge(folder,frozen,source,old_suite)
        new=judge(folder,reloaded,source,old_suite)
        write_receipt(folder/"existing-skill-regression.json",dict(parent=old,child=new))
    else:
        reloaded=read(folder/"student.json")
        require(reloaded["parent_sha256"]==parent_hash and reloaded["peer"]==peer,"frozen inputs drifted")
    child=reloaded["models"]["window"]
    child_hash=digest(model_wire(child,"window"))
    require(child_hash!=parent_hash,"no student parameters learned")
    result=games(folder,parent,child,STARTS[audit])
    result.update(parent_sha256=parent_hash,child_sha256=child_hash,peer=peer)
    if audit:
        dev=read(folder/"development.json")
        require(dev["child_sha256"]==child_hash and
                {r["seed"] for r in dev["rows"]}.isdisjoint(r["seed"] for r in result["rows"]),
                "confirmation reused development or changed weights")
    write_receipt(folder/("confirmation.json" if audit else "development.json"),result)
    print(json.dumps(dict(training=reloaded["training"],differences=result["differences"])))
    if audit:
        direction = [panel["differences"][f"{t}/learn_continuously"]["interval"] for panel in (dev,result) for t in range(3)]
        print("JUNQI_RESEARCH "+("supported" if all(lo>0 for lo,_ in direction) else
                                "refuted" if all(hi<0 for _,hi in direction) else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
