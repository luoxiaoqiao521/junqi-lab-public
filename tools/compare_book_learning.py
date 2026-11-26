"""Compare matched action-value models using public inputs and fixed paired outcomes."""
from collections import defaultdict
from hashlib import sha256
import json
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np

from measure_deadline_opening import ROOT, require

DATA=ROOT/"build/deadline-opening"
SOURCES={
    "ordinary":DATA/"export-deadline-paired-learning-db497cf14e101d997710"/"paired-public.jsonl",
    "pressure":DATA/"expand-book-hard-outcomes-27d694d58dced9de82e9"/"paired-public.jsonl",
}
ENCODER="junqi-paired-public-action-summary-1"
OUTPUTS=("loss","draw","win","win_100","win_140","loss_140","log_remaining")
WIDTH=24
SEEDS=(17,31,97)
UPDATES=220


def hq(node):
    return node<120 and node%30 in (26,28)


def zone(node,actor):
    return (node//30-actor)%4 if node<120 else 4


def can_move(node,mask):
    return not hq(node) and bool(mask&~3)


def features(obs,move):
    actor,board=obs["actor"],obs["board"]
    origin,dest=divmod(move,len(board))
    _,rank,_,_=board[origin]
    owner,_,mask,moved=board[dest]
    x=[obs["ply"]/140,obs["static_plies"]/70,obs["left_100"]/100,obs["left_140"]/140,
       len(obs["legal"])/200, sum(m//len(board)==origin for m in obs["legal"])/40]
    for relative in range(4):
        pieces=[(n,ranks) for n,(s,_,ranks,_) in enumerate(board) if s==(actor+relative)%4]
        movers=[r for n,r in pieces if can_move(n,r)]
        x.extend((len(pieces)/25,len(movers)/21,sum(r&(1<<3)!=0 for r in movers)/21,
                  sum(r&(1<<2)!=0 for r in movers)/21,float(len(movers)==1)))
    for node in (origin,dest):
        x.extend([float(zone(node,actor)==z) for z in range(5)])
        x.extend([node%30//5/5 if node<120 else 0,abs(node%5-2)/2 if node<120 else 0,float(hq(node))])
    x.extend([float(rank==r) for r in range(2,12)])
    x.extend([float(owner>=0),float(moved),*(float(bool(mask&(1<<r))) for r in range(12))])
    x.extend([float(owner>=0)*x[0],float(hq(dest))*x[0],
              float(rank==2 and mask&(1<<1)!=0),float(rank==3 and mask&(1<<1)!=0)])
    return x


def records():
    splits=defaultdict(list)
    hashes={}
    seen={}
    for domain,path in SOURCES.items():
        blob=path.read_bytes()
        hashes[domain]=sha256(blob).hexdigest()
        for line in blob.splitlines():
            row=json.loads(line)
            require(row["version"]=="junqi-public-paired-1","incompatible dataset")
            family=row["family"]
            partition=row["partition"]
            identity=(domain,partition)
            require(seen.setdefault(family,identity)==identity,"layout family reused across domains or splits")
            records=[(features(row["observation"],move),label) for move,label in row["labels"]]
            require(len(records)==2,"action pair missing")
            splits[partition].append((domain,family,records))
    return splits,hashes


def label_vector(label):
    final=label["final"]
    return [float(final==s) if final is not None else np.nan for s in OUTPUTS[:3]]+[
        np.nan if label[s] is None else float(label[s]) for s in OUTPUTS[3:6]]+[
        np.nan if label["censored"] else np.log1p(label["observed_remaining"])/8]


def probabilities(logits):
    result=logits.copy()
    weights=np.exp(logits[:,:3]-logits[:,:3].max(axis=1,keepdims=True))
    result[:,:3]=weights/weights.sum(axis=1,keepdims=True)
    result[:,3:6]=1/(1+np.exp(-np.clip(logits[:,3:6],-35,35)))
    return result


def fit(x,y,seed,width,initial=None,*,updates=UPDATES,rate=.20):
    rng=np.random.default_rng(seed)
    shape=[x.shape[1],width,len(OUTPUTS)] if width else [x.shape[1],len(OUTPUTS)]
    weights=[rng.normal(0,.12/(a**.5),size=(a,b)) for a,b in zip(shape,shape[1:])]
    bias=[np.zeros(b) for b in shape[1:]]
    if initial is not None:
        for destination,source in ((weights,initial["weights"]),(bias,initial["bias"])):
            require(len(destination)==len(source),"initial model depth mismatch")
            for target,value in zip(destination,source,strict=True):
                value=np.asarray(value)
                require(value.shape==target.shape and np.isfinite(value).all(),"incompatible initial weights")
                target[:]=value
    known=np.isfinite(y)
    labels=np.nan_to_num(y)
    for _ in range(updates):
        hidden=np.tanh(x@weights[0]+bias[0]) if width else x
        grad=(probabilities(hidden@weights[-1]+bias[-1])-labels)*known
        grad[:,6]*=.2
        grad/=len(x)
        if width:
            upstream=(grad@weights[-1].T)*(1-hidden**2)
            weights[0]-=rate*(x.T@upstream+.002*weights[0])
            bias[0]-=rate*upstream.sum(axis=0)
        weights[-1]-=rate*(hidden.T@grad+.002*weights[-1])
        bias[-1]-=rate*grad.sum(axis=0)
    return {"weights":[w.tolist() for w in weights],"bias":[b.tolist() for b in bias]}


def predict(artifact,model,x):
    require(artifact["encoder"]==ENCODER and artifact["outputs"]==list(OUTPUTS)
            and x.shape[1]==len(artifact["mean"]), "incompatible action-value encoding")
    x=np.clip((x-artifact["mean"])/artifact["scale"],-6,6)
    votes=[]
    for member in artifact["models"][model]:
        weights=member["weights"]
        hidden=np.tanh(x@np.asarray(weights[0])+member["bias"][0]) if len(weights)==2 else x
        votes.append(probabilities(hidden@np.asarray(weights[-1])+member["bias"][-1]))
    return np.mean(votes,axis=0)


def utility(pred):
    return .5*pred[:,1]+pred[:,2]+.05*pred[:,3]+.15*pred[:,4]-.10*pred[:,5]


def evaluate(artifact,validation):
    report={}
    for domain in SOURCES:
        rows=[items for d,_,items in validation if d==domain]
        x=np.asarray([x for pair in rows for x,_ in pair])
        labels=[l for pair in rows for _,l in pair]
        y=np.asarray([label_vector(l) for l in labels])
        def stats(chosen):
            picks=[labels[2*i+c] for i,c in enumerate(chosen)]
            scores={"win":1,"draw":.5,"loss":0}
            known=[scores[l["final"]] for l in picks if l["final"] in scores]
            return {"team":float(np.mean(known)) if known else None,"known":len(known),
                    "win100":sum(l["win_100"] is True for l in picks),
                    "win140":sum(l["win_140"] is True for l in picks),
                    "loss140":sum(l["loss_140"] is True for l in picks),"changed":int(sum(chosen))}
        report[domain]={"pairs":len(rows),"original":stats([0]*len(rows))}
        decisions={}
        for name in artifact["models"]:
            pred=predict(artifact,name,x)
            scores=utility(pred).reshape(-1,2)
            choices=(scores[:,1]>scores[:,0]).astype(int)
            decisions[name]=choices
            error=(pred-y)**2
            report[domain][name]={**stats(choices),"mse":{
                v:float(np.nanmean(error[:,i])) for i,v in enumerate(OUTPUTS)}}
        report[domain]["different_models"]=int(np.count_nonzero(decisions["linear"]!=decisions["small_nn"]))
    return report


def main(audit):
    splits,hashes=records()
    train,validation=(splits[k] for k in ("developer_train","developer_validation"))
    x=np.asarray([x for _,_,pair in train for x,_ in pair])
    y=np.asarray([label_vector(l) for _,_,pair in train for _,l in pair])
    mean=x.mean(axis=0)
    scale=np.where(x.std(axis=0)>.025,x.std(axis=0),1)
    norm=np.clip((x-mean)/scale,-6,6)
    models={name:[fit(norm,y,seed,width) for seed in SEEDS]
            for name,width in (("linear",0),("small_nn",WIDTH))}
    artifact={"encoder":ENCODER,"outputs":list(OUTPUTS),"mean":mean.tolist(),"scale":scale.tolist(),"models":models}
    wire=json.dumps(artifact,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n"
    require(artifact==json.loads(wire),"training artifact cannot round-trip")
    summary={"inputs":hashes,"train":len(train),"validation":len(validation),
             "feature_count":len(mean),"hidden_width":WIDTH,"updates_per_seed":UPDATES,
             "validation_domains":evaluate(json.loads(wire),validation)}
    path=DATA/os.environ["JUNQI_SERIAL_KEY"]
    path.mkdir(parents=True,exist_ok=True)
    for filename,content in (("models.json",wire),("summary.json",json.dumps(summary,sort_keys=True,allow_nan=False)+"\n")):
        target=path/filename
        if audit:
            require(target.read_bytes()==content.encode(),f"training or evaluation did not reproduce: {filename}")
        else:
            pending=target.with_suffix(".tmp")
            pending.write_bytes(content.encode())
            pending.replace(target)
    print(json.dumps(summary,sort_keys=True,allow_nan=False))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
