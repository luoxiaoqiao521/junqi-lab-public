"""Train on natural post-retirement decisions, then test from untouched openings."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from statistics import median
import sys

from confirm_public_finish_dispatch import params
from export_deadline_learning_pairs import NODES, action_label, identities
from measure_deadline_opening import measure, paired
from measure_human_winner_takeover import MODEL, ROOT, encode, native, persist, require
from train_early_deadline_student import WIDTH, fit_student, model_arguments


TRAIN=(1097100,24)
CONFIRM=(1098100,40)
COLS="seed seat partner table arm first trigger action plies outcome ending friendly_retired enemy_retired prefix trace".split()


def parse_training(wire):
    roots, coverage, samples={}, {},defaultdict(lambda:defaultdict(dict))
    for line in wire.splitlines():
        tag,*values=line.split()
        seed,seat,context,*data=map(int,values)
        key=(seed,seat,context)
        if tag=="P":
            require(key not in roots,"duplicate public teaching root")
            roots[key]=data
        elif tag=="T":
            require(key not in coverage and len(data)==1,"duplicate opening coverage")
            coverage[key]=data[0]
        else:
            rep,prefix,action,plies,outcome,ending=data
            require(tag=="A" and rep in (0,1) and rep not in samples[key][action] and 1<=ending<=5,
                    "invalid continuation sample")
            samples[key][action][rep]=dict(prefix=prefix,plies=plies,outcome=outcome,ending=ending)
    expected={(seed,(seed%4+2*table+partner-1)%4,2*partner+table)
              for seed in range(TRAIN[0],sum(TRAIN)) for partner in (1,2) for table in (0,1)}
    require(set(coverage)==expected and set(roots)==set(samples)=={k for k,v in coverage.items() if v>=0},
            "missing natural opening or public root")
    rows=[]
    for key,frame in roots.items():
        trace,ply,actor,quiet,*nodes=frame
        board=[nodes[i:i+4] for i in range(0,4*NODES,4)]
        legal=nodes[4*NODES:]
        identities(board,actor)
        available={p[0] for p in board if p[0]>=0}
        require(set(legal)>=samples[key].keys() and 2<=len(samples[key])<=6 and coverage[key] in samples[key]
                and actor==key[1] and len({s for s in available if (s-actor)%2})==1 and 0<=quiet<70,
                "invalid post-retirement snapshot or shortlist")
        labels={}
        for move,reps in samples[key].items():
            require(set(reps)=={0,1} and all(row["prefix"]==trace for row in reps.values()),"unpaired sample")
            labels[move] = {"reply_outcomes":[action_label(row,ply) for row in reps.values()],"replays":reps}
        rows.append(dict(family=key[0],seat=actor,context=key[2],prefix=trace,
                         observation=dict(actor=actor,ply=ply,static_plies=quiet,board=board,legal=legal),
                         branches=labels))
    return rows,coverage


def fit_roots(directory, training, parent):
    return fit_student(directory,json.loads(training),(("post_retirement",WIDTH),),sha256(training).hexdigest(),
                       parent=(parent,"replay_control"))


def train(directory, parent):
    require((directory/"finish.json").is_file(),"public proof supervisor must accept before training")
    wire=native(directory,"post-retirement-labels",params(*TRAIN,parent))
    rows,coverage=parse_training(wire)
    roots={r["family"] for r in rows}
    result=dict(openings=len(coverage),roots=len(rows),families=len(roots))
    if len(roots)>=8:
        training=encode(rows)
        persist(directory/"post-roots.json",training,False)
        fitted=fit_roots(directory,training,parent)
        require(fitted["models"]["post_retirement"]!=parent["models"]["replay_control"],"parameters did not update")
        persist(directory/"post-model.json",encode(fitted),False)
        result.update(ply_median=median(r["observation"]["ply"] for r in rows),labels=fitted["labels"],
                      outcomes=dict(Counter(str(y["final"]) for r in rows for z in r["branches"].values()
                                            for y in z["reply_outcomes"])))
    result["coverage"]=[dict(seed=k[0],seat=k[1],context=k[2],baseline=v) for k,v in coverage.items()]
    persist(directory/"training.json",encode(result),False)


def parse_openings(wire):
    rows=defaultdict(dict)
    for line in wire.splitlines():
        tag,*values=line.split()
        row=dict(zip(COLS,map(int,values),strict=True))
        key=(row["seed"],row["seat"],row["partner"],row["table"])
        require(tag=="G" and row["arm"] not in rows[key] and row["outcome"] in (-1,0,1,2)
                and 1<=row["ending"]<=5 and 0<row["plies"]<=3000,"invalid opening result")
        rows[key][row["arm"]]=row
    expected={(seed,(seed%4+2*t+p-1)%4,p,t) for seed in range(CONFIRM[0],sum(CONFIRM))
              for p in (1,3) for t in (0,1)}
    require(set(rows)==expected,"incomplete holdout opening roster")
    for game in rows.values():
        require(set(game)=={0,1,2},"incomplete matched policies")
        a,b,c=(game[i] for i in range(3))
        require(all(a[k]==b[k]==c[k] for k in ("prefix","trigger","first","enemy_retired")),"changed matched prefix")
        require(a["trigger"]==0 or 0<a["enemy_retired"]<a["trigger"],"intervention preceded the first enemy exit")
        require(a["trigger"]>140 or b["action"]==c["action"],"changed time weights before deadline")
        for x,y in ((a,b),(b,c)):
            if x["action"]==y["action"]:
                require(all(x[k]==y[k] for k in ("outcome","trace","plies")),"same action produced different result")
        for row in game.values():
            changed=int(bool(row["trigger"] and row["action"]!=a["action"]))
            row.update(changed=changed,changed100=changed*(row["trigger"]<=100),changed140=changed*(row["trigger"]<=140))
    return rows


def panels(rows):
    result={}
    for partner in (None,1,3):
        groups=[[arms[i] for key,arms in sorted(rows.items()) if partner is None or key[2]==partner] for i in range(3)]
        measurements=list(map(measure,groups))
        comparisons={}
        for left,right in ((0,1),(1,2),(0,2)):
            a,b=groups[left],groups[right]
            report=paired(a,b,measurements[left],measurements[right])
            wins=[(x,y) for x,y in zip(a,b,strict=True) if x["outcome"]==y["outcome"]==2]
            conversions=[y["plies"]-x["plies"] for x,y in wins if x["enemy_retired"] and y["enemy_retired"]]
            report.update(changes=sum(x["action"]!=y["action"] for x,y in zip(a,b,strict=True)),
                          win_to_loss=sum(x["outcome"]==2 and y["outcome"]==0 for x,y in zip(a,b,strict=True)),
                          win_to_draw=sum(x["outcome"]==2 and y["outcome"]==1 for x,y in zip(a,b,strict=True)),
                          paired_wins=len(wins),conversion_median=median(conversions) if conversions else None)
            comparisons[f"{left}/{right}"]=report
        result[str(partner)]=dict(triggers=sum(bool(r["trigger"]) for r in groups[0]),
                                  triggers_140=sum(0<r["trigger"]<=140 for r in groups[0]),
                                  policies=measurements,comparisons=comparisons)
    return result


def confirm(directory,parent):
    record=json.loads((directory/"training.json").read_bytes())
    summary=dict(decision="insufficient_training_coverage",training_families=record["families"])
    if record["families"]>=8:
        weight_bytes=(directory/"post-model.json").read_bytes()
        fitted=json.loads(weight_bytes)
        model=model_arguments(fitted,"post_retirement")
        training=(directory/"post-roots.json").read_bytes()
        require(fitted["source"]==sha256(training).hexdigest(),"training inputs changed")
        require(encode(fit_roots(directory,training,parent))==weight_bytes,"saved training data changed rebuilt weights")
        wire=params(*CONFIRM,parent)+" "+model
        rows=parse_openings(native(directory,"post-retirement-openings",wire))
        require(parse_openings(native(directory,"post-retirement-openings",wire))==rows,"held-out replay changed")
        summaries=panels(rows)
        comparison=summaries["None"]["comparisons"]["0/1"]
        low,high=comparison["score_outer_interval"]
        if low>0 and comparison["exploratory_budget_ok"]:
            decision="keep_for_independent_expansion"
        elif high<-.025:
            decision="negative"
        else:
            decision="inconclusive_keep_checkpoint"
        summary.update(decision=decision,panels=summaries,rows=[arms for _,arms in sorted(rows.items())])
    persist(directory/"opening.json",encode(summary),False)


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    directory=ROOT/"build/finish-and-conversion"/os.environ["JUNQI_SERIAL_KEY"]
    (confirm if sys.argv[1:] else train)(directory,json.loads(MODEL.read_bytes()))
    print("JUNQI_VERDICT accepted")
