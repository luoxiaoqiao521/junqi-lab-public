"""Fit independent WDL estimators and rerank exactly the same full-mode search traces."""
from collections import Counter
from itertools import product
import json
import os
from time import perf_counter
import sys

from compare_book_learning import np
from serial_dynamic_reply import ROOT, build, frozen_wire, native, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import digest
from serial_quiet_options import FROZEN
from serial_reply_window import model_wire
from serial_visited_reply_value import choose, manifest
from serial_worker import read, write_receipt

SCHEMA="full-visible-four-seat-leaf-wdl-v1"
WIDTH=36
PAYOFF=np.array([0.,.5,1.])
TEACHERS=("legacy","calibrated","state")


def collect(folder,wire,audit,*,sources=None,recheck=False):
    spec,tape=sources if sources is not None else manifest(
        audit,seed_offset=4900 if audit else 3800,archive_offset=18 if audit else 12,
        include_witness=False,cuts=(1,21,52,76))
    roots,missing={},{}
    mode="full-leaf-teacher-recheck" if recheck else "full-leaf-teacher-data"
    for tag,key,*fields in map(str.split,native(folder,mode,wire+"\n"+tape).splitlines()):
        require(key in spec,"unexpected leaf root")
        if tag=="M":
            missing[key]=fields
        elif tag=="R":
            root=dict(zip(("cut","hash","seat","actions","legal"),map(int,fields),strict=True))
            roots[key]=dict(root,leaves={},review={})
        elif tag=="L":
            action,stream,plies,h,outcome,forced,old,micros,*raw=fields
            r=dict(plies=int(plies),hash=int(h),outcome=int(outcome),forced=int(forced),
                   old=float(old),feature_micros=int(micros),x=list(map(float,raw)),rollouts={})
            require(len(r["x"])==WIDTH and np.isfinite(r["x"]).all() and
                    r["outcome"] in (-2,-1,0,1,2) and 1<=r["plies"]<=6,"invalid frozen leaf")
            roots[key]["leaves"].setdefault(action,{})[stream]=r
        else:
            require(tag in ("Y","G"),"unexpected frozen continuation")
            action,*values=fields
            t,i,y=values[-3:]
            score=int(y)
            require(score in (-1,0,1,2),"invalid fixed continuation outcome")
            r=roots[key]["leaves"][action][values[0]]["rollouts"] if tag=="Y" else roots[key]["review"].setdefault(action,{})
            require(f"{t}/{i}" not in r,"duplicated continuation")
            r[f"{t}/{i}"]=score
    require(set(missing)|set(roots)==set(spec) and len(roots)>=16,"source admission changed")
    for r in roots.values():
        require(len(r["leaves"])==r["actions"]<=min(4,r["legal"]) and r["leaves"].keys()==r["review"].keys(),
                "candidate coverage changed")
        for a,leaves in r["leaves"].items():
            require(set(leaves)==set("0123") and set(r["review"][a])=={f"{t}/{i}" for t,i in product(range(4),range(8))},
                    "incomplete root review")
            for leaf in leaves.values():
                y=leaf["rollouts"]
                require(set(y)=={f"{t}/{i}" for t,i in product(range(4),range(3))} and
                        (leaf["outcome"]==-2 or set(y.values())=={leaf["outcome"]}),"terminal label mismatch")
    return dict(roots=roots,missing=missing,spec=spec)


def observed(leaf,tables):
    values=[leaf["rollouts"][f"{t}/{s}"] for t,s in product(tables,range(3))]
    if -1 in values:
        return None
    counts=Counter(values)
    return np.array([counts[k]/len(values) for k in range(3)])


def probability(matrix,model):
    x=np.clip((matrix[:,model["cols"]]-model["mean"])/model["scale"],-6,6)
    z=x@model["weights"]+model["bias"]
    p=np.exp(z-z.max(axis=1,keepdims=True))
    return p/p.sum(axis=1,keepdims=True)


def train(data):
    leaves=[l for r in data["roots"].values() for a in r["leaves"].values() for l in a.values()
            if l["outcome"]==-2 and observed(l,range(3)) is not None]
    require(len(leaves)>=200,"insufficient independent cut states")
    x=np.array([l["x"] for l in leaves])
    y=np.array([observed(l,range(3)) for l in leaves])
    models={}
    for label,cols in (("calibrated",[0]),("state",list(range(WIDTH-1)))):
        features=x[:,cols]
        mean=features.mean(axis=0)
        scale=np.maximum(features.std(axis=0),.05)
        norm=np.clip((features-mean)/scale,-6,6)
        weights=np.zeros((len(cols),3))
        bias=np.zeros(3)
        for _ in range(160):
            z=norm@weights+bias
            p=np.exp(z-z.max(axis=1,keepdims=True))
            residual=(p/p.sum(axis=1,keepdims=True)-y)/len(x)
            weights-=.08*(norm.T@residual+.02*weights)
            bias-=.08*residual.sum(axis=0)
        require(np.isfinite(weights).all() and np.linalg.norm(weights)>0,"value did not update")
        models[label]=dict(cols=cols,weights=weights.tolist(),bias=bias.tolist(),
                           scale=scale.tolist(),mean=mean.tolist())
    return dict(schema=SCHEMA,models=models,training_leaves=len(x),training_tables=[0,1,2])


def value_rows(data,artifact):
    begin=perf_counter()
    require(artifact["schema"]==SCHEMA and artifact["training_tables"]==[0,1,2],"unsupported value artifact")
    forecasts={}
    names=(*TEACHERS,"residual") if "residual" in artifact else TEACHERS
    for key,r in data["roots"].items():
        choices={teacher:{} for teacher in names}
        explicit=[]
        for action,leaves in r["leaves"].items():
            rows=list(leaves.values())
            if rows[0]["forced"]==2:
                explicit.append(action)
            choices["legacy"][action]=dict(score=sum(l["old"] for l in rows))
            matrix=np.array([l["x"] for l in rows])
            values={name:probability(matrix,model) for name,model in artifact["models"].items()}
            if "residual" in artifact:
                values["residual"]=residual_probability(matrix,values["calibrated"],artifact["residual"])
            for name,pred in values.items():
                for j,l in enumerate(rows):
                    if l["outcome"] in (0,1,2):
                        pred[j]=np.eye(3)[l["outcome"]]
                choices[name][action]=dict(score=float((pred@PAYOFF).mean()),probabilities=pred.tolist())
        picks={name:explicit[0] if explicit else choose(list(r["leaves"]),scores) for name,scores in choices.items()}
        truncated=any(l["outcome"]==-1 for a in r["leaves"].values() for l in a.values())
        if truncated:
            picks={name:picks["legacy"] for name in names}
        forecasts[key]=dict(scores=choices,choice=picks,forced=bool(explicit),truncated=truncated)
    return forecasts,perf_counter()-begin


def residual_features(x):
    friendly=x[:,7:14]+x[:,21:28]
    opposing=x[:,14:21]+x[:,28:35]
    return np.column_stack((x[:,1],x[:,2],(friendly-opposing)[:,[0,1,2,4,5]],
                            x[:,3]+x[:,5]))


def fit_residual(data,base,alpha=1.):
    x,y,weights=[],[],[]
    for root in data["roots"].values():
        labels=[(leaf,observed(leaf,range(3))) for group in root["leaves"].values()
                for leaf in group.values() if leaf["outcome"]==-2]
        settled=[(leaf,outcome) for leaf,outcome in labels if outcome is not None]
        if settled:
            x.extend(l["x"] for l,_ in settled)
            y.extend(o@PAYOFF for _,o in settled)
            weights.extend([1/len(settled)]*len(settled))
    require(x,"empty residual training evidence")
    matrix=np.array(x)
    features=residual_features(matrix)
    weights=np.array(weights)/sum(weights)
    mean=weights@features
    scale=np.maximum(np.sqrt(weights@((features-mean)**2)),.05)
    norm=np.clip((features-mean)/scale,-6,6)
    design=np.column_stack((np.ones(len(x)),norm))
    target=np.array(y)-probability(matrix,base)@PAYOFF
    ridge=np.diag([.5,1.,1.,1.,1.,1.,1.,1.,1.])
    coeff=np.linalg.solve(design.T@(weights[:,None]*design)+ridge,design.T@(weights*target))
    return dict(mean=mean.tolist(),scale=scale.tolist(),weights=coeff.tolist(),alpha=alpha,max_shift=.12)


def residual_probability(x,base,model):
    norm=np.clip((residual_features(x)-model["mean"])/model["scale"],-6,6)
    delta=model["alpha"]*(np.column_stack((np.ones(len(x)),norm))@model["weights"])
    limit=model["max_shift"]
    step=np.clip(delta,-np.minimum(limit,base[:,2]),np.minimum(limit,base[:,0]))
    pred=base.copy()
    pred[:,2]+=step
    pred[:,0]-=step
    return pred


def summarize(data,predictions):
    names=tuple(next(iter(predictions.values()))["choice"])
    calibration={}
    for policy in (0,1,2,3):
        error={m:[] for m in names if m!="legacy"}
        for key,r in data["roots"].items():
            for a,leaves in r["leaves"].items():
                for i,l in enumerate(leaves.values()):
                    y=observed(l,[policy])
                    if y is None or l["outcome"]!=-2:
                        continue
                    for m in error:
                        p=np.array(predictions[key]["scores"][m][a]["probabilities"][i])
                        error[m].append((float(np.sum(p*p)-2*p@y+1),float(-y@np.log(np.maximum(p,1e-12)))))
        calibration[policy]={m:dict(brier=float(np.mean(np.array(e)[:,0])),
                                      logloss=float(np.mean(np.array(e)[:,1])),leaves=len(e)) for m,e in error.items()}
    effects={}
    comparisons=[("legacy","calibrated"),("legacy","state"),("calibrated","state")]
    if "residual" in names:
        comparisons.extend((("legacy","residual"),("calibrated","residual")))
    for old,new in comparisons:
        switched=0
        pairs={t:[] for t in range(4)}
        for key,r in data["roots"].items():
            a,b=(predictions[key]["choice"][m] for m in (old,new))
            switched+=a!=b
            for t in pairs:
                ds=[(0.,0.) if a==b else score_difference(r["review"][a][f"{t}/{s}"],r["review"][b][f"{t}/{s}"])
                    for s in range(8)]
                pairs[t].append(tuple(sum(x)/8 for x in zip(*ds,strict=True)))
        effects[f"{new}_vs_{old}"]=dict(switched=switched,tables={t:contrast(ds) for t,ds in pairs.items()})
    reference={t:[] for t in range(4)}
    for key,r in data["roots"].items():
        if not data["spec"][key]["reference"]:
            continue
        estimates={}
        for a,rows in r["review"].items():
            base=[rows[f"{t}/{i}"] for t,i in product(range(3),range(4))]
            estimates[a]=dict(score=sum(0 if s<0 else s/2 for s in base)/len(base))
        a=predictions[key]["choice"]["legacy"]
        b=a if predictions[key]["forced"] else choose(list(estimates),estimates)
        for t in reference:
            ds=[(0.,0.) if a==b else score_difference(r["review"][a][f"{t}/{s}"],r["review"][b][f"{t}/{s}"])
                for s in range(4,8)]
            reference[t].append(tuple(sum(v)/4 for v in zip(*ds,strict=True)))
    return dict(roots=len(data["roots"]),calibration=calibration,effects=effects,
                offline_reference={t:contrast(r) for t,r in reference.items() if r},
                root_review_count=sum(len(s) for r in data["roots"].values() for s in r["review"].values()),
                feature_micros=sum(l["feature_micros"] for r in data["roots"].values()
                                  for a in r["leaves"].values() for l in a.values()))


def main(audit):
    folder=ROOT/"build/leaf-teacher"/os.environ["JUNQI_SERIAL_KEY"]
    base,_=frozen_wire()
    wire=base+" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    data=collect(folder,wire,audit)
    artifact_path=folder/"value.json"
    if not audit:
        value=train(data)
        value.update(source_sha=digest(wire),training_ids=list(data["spec"]),
                     training_families=[r["family"] for r in data["spec"].values() if "family" in r])
        write_receipt(artifact_path,value)
    value=read(artifact_path)
    require(value["source_sha"]==digest(wire),"replay policy changed")
    if audit:
        require(set(value["training_ids"]).isdisjoint(data["spec"]) and set(value["training_families"]).isdisjoint(
            r["family"] for r in data["spec"].values() if "family" in r),"independent leaf audience leaked")
    trace,seconds=value_rows(data,value)
    stats=summarize(data,trace)
    stats.update(value_inference_seconds=seconds,model_hash=digest(artifact_path.read_text(encoding="utf8")))
    name="confirmation" if audit else "development"
    write_receipt(folder/(name+".json"),dict(experience=data,predictions=trace,summary=stats))
    print(json.dumps(stats))
    if audit:
        intervals=[r["interval"] for r in stats["effects"]["state_vs_legacy"]["tables"].values()]
        print("JUNQI_RESEARCH "+("supported" if all(lo>0 for lo,_ in intervals) else
                                "refuted" if all(hi<0 for _,hi in intervals) else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
