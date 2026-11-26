"""Compare breadth, depth and a leaf ablation on equally budgeted fixed lists."""
import json
import math
import os
import sys

from serial_dynamic_reply import ROOT,require
from serial_formation_matrix import contrast,score_difference
from serial_full_scoring_audit import branches,collect,run,verified_scope
from serial_teacher_score_terms import prepare
from serial_worker import read,write_receipt

PREVIOUS=ROOT/"build/full-scoring-audit/confirm-actual-scoring-splits-and-reference-search-bbde4110d57d3af71fdb"
ARMS=("wide","six","deep","no_distance")
START=2865100


def variants(scope,wire,roots):
    tape="\n".join(f"{s} {r['table']} {r['request_cut']} 4" for s,r in roots.items())
    stats={}
    for tag,seed,a,depth,used,ended,score,less in map(str.split,run(scope,wire,tape).splitlines()):
        s,a,d,n,t=map(int,(seed,a,depth,used,ended))
        require(tag=="N" and d in (3,6,12) and 1<=n<=24 and 0<=t<=24//d and (s,a,d) not in stats,
                "invalid matched cap or duplicate native trace")
        stats[s,a,d]=(n,t,float(score),float(less))
    for s,r in roots.items():
        available=sorted((a for a in r["actions"] if r["actions"][a]["order"]>=0),
                         key=lambda a:r["actions"][a]["order"])
        for a in available:
            n,_,score,_=stats[s,a,6]
            require(n==r["actions"][a]["plies"] and math.isclose(score,r["actions"][a]["six"]/4,abs_tol=1e-12),
                    "fixed-six comparison did not reproduce")
        for name,d,col in (("wide",3,2),("six",6,2),("deep",12,2),("no_distance",6,3)):
            chosen=available[0]
            for action in available:
                if stats[s,action,d][col]>stats[s,chosen,d][col]+1e-9:
                    chosen=action
            r[name]=chosen
    return stats


def evaluate(scope,wire,roots,streams):
    choices={s:list(dict.fromkeys([r[k] for k in ARMS])) for s,r in roots.items()}
    outcomes,work=branches(scope,wire,roots,choices,(2,3,4),streams)
    results={}
    for desk in (2,3,4):
        results[desk]={}
        for key in set(ARMS)-{"six"}:
            diffs=[]
            negative=positive=0
            for s,r in roots.items():
                differences=[(0.,0.) if r["six"]==r[key] else
                             score_difference(outcomes[s,r["six"],desk,i],outcomes[s,r[key],desk,i]) for i in streams]
                sums=tuple(sum(v)/len(streams) for v in zip(*differences,strict=True))
                diffs.append(sums)
                positive+=sums[0]>0
                negative+=sums[1]<0
            results[desk][key]=dict(**contrast(diffs),positive=positive,negative=negative)
    return dict(results=results,review_plies=work,branches=len(outcomes),
                outcomes=[[*k,v] for k,v in outcomes.items()])


def main(audit):
    folder=ROOT/"build/matched-search-budget"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        prepare(folder)
    scope,wire=verified_scope(folder)
    if audit:
        manifest=[dict(seed=seed,cut=(5,22,43,70)[i%4],table=2,group="held-out")
                  for i,seed in enumerate(range(START,START+16))]
        roots=collect(scope,wire,manifest)
    else:
        rows=read(PREVIOUS/"development.json")["roots"]
        roots={int(s):dict(r,actions={int(a):x for a,x in r["actions"].items()}) for s,r in rows.items()}
    scores=variants(scope,wire,roots)
    checks=evaluate(scope,wire,roots,range(19500,19506) if audit else range(19520,19526))
    usage={d:dict(plies=sum(t[0] for k,t in scores.items() if k[2]==d),
                  terminals=sum(t[1] for k,t in scores.items() if k[2]==d)) for d in (3,6,12)}
    changed={a:sum(r[a]!=r["six"] for r in roots.values()) for a in ARMS}
    summary=dict(usage=usage,changed=changed,roots=len(roots),**checks)
    write_receipt(folder/("confirmation.json" if audit else "development.json"),
                  dict(summary,trace=[[*key,*r] for key,r in scores.items()],
                       choices={s:{k:v for k,v in r.items() if k!="actions"} for s,r in roots.items()}))
    if audit:
        support=[a for a in changed if a!="six" and all(checks["results"][d][a]["interval"][0]>0 for d in (2,3,4))]
        print("JUNQI_RESEARCH "+("supported" if support else "inconclusive"))
    print(json.dumps({k:v for k,v in summary.items() if k!="outcomes"}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no args or audit")
    main(bool(sys.argv[1:]))
