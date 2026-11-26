"""Compare WDL supervision with same-position preferences under shared retention."""
from collections import Counter
from itertools import product
import json
import os
import sys

from compare_book_learning import np
from serial_dynamic_reply import ROOT, PRIOR, build, frozen_wire, native, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import GAME_KEYS
from serial_local_lesson_cycle import independent_review
from serial_quiet_options import FROZEN
from serial_reply_window import examples, judge, model_wire, split
from serial_student_lesson_credit import PREVIOUS, measure_root_scores
from serial_visited_reply_value import collect, manifest
from serial_worker import read, write_receipt

MODELS=("parent","wdl","pair")
PAYOFF=np.array([0.,.5,1.])
PANEL_START=2842000
PANEL_COUNT=24


def wdl(root,action,pressure):
    counts=Counter(root["trials"][f"{action}/{t}/{pressure}/{s}"]["result"] for t,s in product(range(3),range(8)))
    require(sum(counts[k] for k in range(3))==24,"truncated branch cannot supply a lesson")
    return np.array([counts[k]/24 for k in range(3)])


def training_rows(screen,old):
    x,y,pairs,retain=[],[],[],[]
    for key,decision in screen["decisions"].items():
        root=screen["review"][key]
        if decision["admitted"]:
            a,b=decision["base"],decision["candidate"]
            # Ambiguous pressure branches temper both objectives by the same mixture.
            labels=[.75*wdl(root,action,0)+.25*wdl(root,action,1) for action in (a,b)]
            gap=float((labels[1]-labels[0])@PAYOFF)
            if gap<=0:
                continue
            pairs.append((len(x),len(x)+1,.5+.5*gap))
            x.extend(root["x"][a] for a in (a,b))
            y.extend(labels)
        elif all(hi<=0 for _,hi in decision["natural"]):
            # Use a restricted retention set where the reviewed alternative offered no better table.
            for action in root["x"]:
                retain.append(len(x))
                x.append(root["x"][action])
                y.append([np.nan]*3)
    for anchor in old:
        for features,result in zip(anchor["features"],anchor["scores"],strict=True):
            x.append(features)
            y.append(np.eye(3)[result])
    require(pairs and retain,"comparison needs both lesson and retention rows")
    return np.array(x),np.array(y),pairs,retain


def team_log_odds(logits):
    half_draw=logits[:,1]-np.log(2.)
    gain=np.logaddexp(logits[:,2],half_draw)
    loss=np.logaddexp(logits[:,0],half_draw)
    derivative=np.column_stack((-np.exp(logits[:,0]-loss),
                                np.exp(half_draw-gain)-np.exp(half_draw-loss),
                                np.exp(logits[:,2]-gain)))
    return gain-loss,derivative


def train(parent,course,use_pairs,*,steps=40,rate=.03,decay=.002,log_odds=False,trace=None,step_norms=None,
          advantage=False):
    raw,y,pairs,retained=course
    require(not (advantage and log_odds),"advantage targets use team score units")
    normalized=(raw-parent["mean"])/parent["scale"]
    x=np.clip(normalized,-6,6)
    w,v=(np.array(a,dtype=float) for a in parent["parameters"]["weights"])
    b,c=(np.array(a,dtype=float) for a in parent["parameters"]["bias"])

    def forward():
        h=np.tanh(x@w+b)
        z=(h@v+c)[:,:3]
        p=np.exp(z-z.max(axis=1,keepdims=True))
        p/=p.sum(axis=1,keepdims=True)
        return h,p,p@PAYOFF,z

    _,_,frozen,_=forward()
    active=np.isfinite(y[:,0])
    if use_pairs:
        active[np.array([(a,b) for a,b,_ in pairs]).ravel()]=False
    if trace is not None:
        trace["clipped_features"]=int(np.count_nonzero(normalized!=x))
        trace["trajectory"]=[]
    require(step_norms is None or len(step_norms)==steps,"update schedule length differs")
    for step in range(steps):
        h,p,value,logits=forward()
        gradient=np.zeros((len(x),len(c)))
        gradient[active,:3]=p[active]-y[active]
        direction=p*(PAYOFF-value[:,None])
        gradient[retained,:3]=2*(value[retained]-frozen[retained])[:,None]*direction[retained]
        preference,slope=team_log_odds(logits) if log_odds else (value,direction)
        if use_pairs:
            for a,z,target in pairs:
                margin=preference[z]-preference[a]
                if advantage:
                    delta=4*np.clip(margin-target,-.25,.25)
                else:
                    probability=np.exp(-np.logaddexp(0.,-4*margin)) if log_odds else 1/(1+np.exp(-4*margin))
                    delta=4*(probability-target)
                gradient[a,:3]-=delta*slope[a]
                gradient[z,:3]+=delta*slope[z]
        gradient/=len(x)
        hidden=(gradient@v.T)*(1-h*h)
        updates=(x.T@hidden+decay*w,hidden.sum(axis=0),h.T@gradient+decay*v,gradient.sum(axis=0))
        update_rate=rate
        if trace is not None or step_norms is not None:
            norm=float(np.sqrt(sum(np.sum(g*g) for g in updates)))
            if step_norms is not None:
                require(norm>0 or step_norms[step]==0,"nonzero requested step with no gradient")
                update_rate=step_norms[step]/norm if norm else 0.
        if trace is not None:
            trace["trajectory"].append(dict(step=step,
                pairs=[dict(base_logits=logits[a].tolist(), alternative_logits=logits[z].tolist(),
                            ev_gap=float(value[z]-value[a]), preference_gap=float(preference[z]-preference[a]),
                            gradient_norm=float(np.linalg.norm(gradient[[a,z]]))) for a,z,_ in pairs],
                parameter_step_l2=update_rate*norm))
        for parameter,increment in zip((w,b,v,c),updates,strict=True):
            parameter-=update_rate*increment
    require(all(np.isfinite(k).all() for k in (w,v,b,c)),"nonfinite preference model")
    return dict(mean=parent["mean"],scale=parent["scale"],
                parameters=dict(weights=[w.tolist(),v.tolist()],bias=[b.tolist(),c.tolist()]))


def run_games(folder,models):
    payload=f"{PANEL_START} {PANEL_COUNT} "+" ".join(model_wire(models[k],"window") for k in MODELS)
    games=[]
    for tag,*values in map(str.split,native(folder,"full-student-preference-games",payload).splitlines()):
        require(tag=="G","invalid frozen game output")
        games.append(dict(zip(GAME_KEYS,map(int,values),strict=True)))
    keys=list(product(range(PANEL_START,PANEL_START+PANEL_COUNT),range(4),range(3)))
    require([(g["seed"],g["table"],g["arm"]) for g in games]==keys and
            all(g["used"]==g["turns"] and g["result"] in (-1,0,1,2) for g in games),"invalid matched full policy")
    summary={}
    for table in range(4):
        rows=[g for g in games if g["table"]==table]
        deltas={}
        for a,b in ((0,1),(0,2),(1,2)):
            d=[(0.,0.) if x["final_hash"]==y["final_hash"] else score_difference(x["result"],y["result"])
               for x,y in zip(rows[a::3],rows[b::3],strict=True)]
            deltas[f"{MODELS[b]}_vs_{MODELS[a]}"]=contrast(d)
        summary[table]=dict(deltas=deltas,outcomes={name:Counter(g["result"] for g in rows[i::3])
                                                   for i,name in enumerate(MODELS)})
    return dict(rows=games,summary=summary)


def retention_panel(folder,models):
    wire,_=frozen_wire()
    wire+=" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    sources=manifest(True,seed_offset=2200,archive_offset=12,include_witness=False)
    search=collect(folder,wire,True,sources)
    review=independent_review(folder,wire,sources,search)
    root_keys=[k for k in search["roots"]]
    probe=dict(review=review,decisions={k:dict(admitted=True,base=review[k]["base"],candidate=review[k]["base"])
                                       for k in root_keys})
    actions={m:measure_root_scores(folder,v,probe) for m,v in models.items()}
    differences={}
    for m in MODELS[1:]:
        table={}
        for t in range(3):
            diffs=[]
            for k in root_keys:
                a,b=(actions[n][k]["chosen"] for n in ("parent",m))
                r=review[k]["trials"]
                ds=[(0.,0.) if a==b else score_difference(r[f"{a}/{t}/0/{s}"]["result"],r[f"{b}/{t}/0/{s}"]["result"])
                    for s in range(8)]
                diffs.append(tuple(sum(x)/8 for x in zip(*ds,strict=True)))
            table[t]=contrast(diffs)
        differences[m]=dict(changes=sum(actions["parent"][k]["chosen"]!=v["chosen"] for k,v in actions[m].items()),
                            tables=table)
    return dict(actions=actions,differences=differences,review=review,manifest=sources[0])


def main(audit):
    folder=ROOT/"build/preference-retention"/os.environ["JUNQI_SERIAL_KEY"]
    frozen=read(PRIOR/"students.json")
    parent=frozen["models"]["window"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
        cases,_,source=examples(folder)
        old,test=split(cases)
        screen=read(PREVIOUS/"screen.json")
        course=training_rows(screen,old)
        models=dict(parent=parent,wdl=train(parent,course,False),pair=train(parent,course,True))
        require(models["wdl"]!=models["pair"] and all(models[m]!=parent for m in MODELS[1:]),"weights did not update")
        write_receipt(folder/"models.json",models)
        diagnostic={}
        for name,model in read(folder/"models.json").items():
            artifact=dict(models={"board":frozen["models"]["board"],"window":model})
            diagnostic[name]=dict(lessons=measure_root_scores(folder,model,screen),
                                  old_correct=judge(folder,artifact,source,test)["errors"]["window"]["counts"]["correct"])
        write_receipt(folder/"development.json",dict(diagnostic=diagnostic,pairs=len(course[2]),retained=len(course[3]),
                                                     update_count=40))
        print(json.dumps({k:[r["chosen"]==r["lesson"] for r in d["lessons"].values()] for k,d in diagnostic.items()}))
    else:
        models=read(folder/"models.json")
        require(parent==models["parent"],"frozen parent changed")
        panel=run_games(folder,models)
        retention=retention_panel(folder,models)
        write_receipt(folder/"confirmation.json",dict(panel=panel,retention=retention))
        print(json.dumps(dict(panel=panel["summary"],retention=retention["differences"])))
        rows=[v["deltas"]["pair_vs_wdl"]["interval"] for v in panel["summary"].values()]
        print("JUNQI_RESEARCH "+("supported" if all(lo>0 for lo,_ in rows) else
                                "refuted" if all(hi<0 for _,hi in rows) else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
