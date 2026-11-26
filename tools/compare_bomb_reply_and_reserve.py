"""Compare bomb reply, another move keeping the bomb, and a public flag guard."""
from collections import Counter, defaultdict
import json
import os
from statistics import mean
import subprocess
import sys

from audit_replay_bomb_replies import CONFIG, ROOT, RULES, compile_native, labeled, prepare, require, write_result
from compare_archive_cycle_exit_values import interval


PRIOR = ROOT/"build/replay-bomb-replies/audit-replay-bomb-replies-eea2934ece14133624ff/summary.json"
ORIGINAL = ROOT/"build/replay-bomb-replies/compare-bomb-reply-and-reserve-db707bf829df8a9e4ec2/summary.json"
FIELDS = ("plies","outcome","ending","friendly_exit","enemy_exit")
ROLES = ("bomb","reserve","guard")


def parse(data, expected):
    roots, results = {},defaultdict(dict)
    for line in data.splitlines():
        tag,family,*raw = line.split()
        if tag=="D":
            continue
        ply,ally,*values = map(int,raw)
        key = (family,ply,ally)
        require(key in expected,"unregistered bomb opportunity")
        if tag=="K":
            turn, seat, *actions = values
            selected=[m for m in actions if m>=0]
            require(key not in roots and actions[0]>=0 and len(actions)==3
                    and len(selected)==len(set(selected)) and all(m<129*129 for m in selected)
                    and turn==ply+expected[key]["delay"] and 0<=seat<4,"invalid bomb alternatives")
            roots[key]=dict(turn=turn,seat=seat,actions=actions)
        else:
            table,rep,role,*outcome=values
            run=(table,rep,role)
            require(tag=="F" and run not in results[key],"invalid or duplicate continuation")
            row=dict(zip(FIELDS,outcome,strict=True))
            require(1<=row["plies"]<=960 and row["outcome"] in (-1,0,1,2),"invalid rollout result")
            results[key][run]=row
    require(roots.keys()==results.keys()==expected.keys(),"missing original legal opportunities")
    for key,root in roots.items():
        grid={(t,r,a) for t in (0,1) for r in (0,1) for a,m in enumerate(root["actions"]) if m>=0}
        require(results[key].keys()==grid,"paired continuation missing")
    return roots,results


def score(row):
    value=row["outcome"]/2
    return (0.,1.) if value<0 else (value,value)


def panel(results, alternative, table):
    blocks=defaultdict(list)
    counts=Counter()
    for (family,_,_), runs in results.items():
        if (table,0,alternative) not in runs:
            continue
        for rep in (0,1):
            a,b = (runs[table,rep,k] for k in (0,alternative))
            x,y=score(a); lo,hi=score(b)
            blocks[family].append((x-hi,y-lo))
            counts[f"{b['outcome']}→{a['outcome']}"]+=1
    if not blocks:
        return dict(families=0,pairs=0)
    lo,hi = ([mean(x[k] for x in rows) for rows in blocks.values()] for k in (0,1))
    return dict(families=len(blocks),pairs=sum(counts.values()),outcome_transitions=dict(counts),
                bounds=[mean(lo),mean(hi)],layout_interval=[interval(lo)[0],interval(hi)[1]])


def cases(results, alternative):
    groups=defaultdict(list)
    for key,runs in results.items():
        if (0,0,alternative) not in runs:
            continue
        pairs=[(runs[t,r,0]["outcome"],runs[t,r,alternative]["outcome"]) for t in (0,1) for r in (0,1)]
        if any(min(pair)<0 for pair in pairs):
            kind="unresolved"
        else:
            tables=[sum(a-b for a,b in pairs[2*t:2*t+2]) for t in (0,1)]
            if min(tables)>0:
                kind="bomb_better_both_tables"
            elif max(tables)<0:
                kind="alternative_better_both_tables"
            elif min(tables)<0<max(tables):
                kind="table_reversal"
            elif any(tables):
                kind="single_table_signal"
            else:
                kind="no_table_sum_difference"
        groups[kind].append(list(key))
    return dict(groups)


def changes(original, roots, examples):
    require(roots=={tuple(row["key"]):{k:v for k,v in row.items() if k not in ("key","rollouts")}
                    for row in original["rows"]},"resample changed root, acting seat or candidate")
    old={tuple(key):label for label,keys in original["examples"]["reserve"].items() for key in keys}
    new={tuple(key):label for label,keys in examples["reserve"].items() for key in keys}
    require(old.keys()==new.keys(),"comparison coverage changed")
    return dict(Counter(f"{old[k]}→{new[k]}" for k in old))


def main(audit, resample=False):
    directory=ROOT/"build/replay-bomb-replies"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        compile_native(directory)
    previous=json.loads(PRIOR.read_bytes())
    feed,provenance=prepare()
    feed=labeled(feed,provenance)
    require(provenance==previous["provenance"],"archive comparison changed from the baseline")
    expected={(w["family"],w["capture_ply"],w["ally"]):w for w in previous["windows"]
              if w["source"]=="archived" and w["legal_replies"]}
    mode="bomb-reserve-resample" if resample else "bomb-reserve"
    process=subprocess.run([directory/"takeover.exe",CONFIG,RULES,"960",mode],
                           input=feed,encoding="utf8",capture_output=True,timeout=170)
    require(process.returncode==0,f"reserve probe failed: {process.stderr.strip()}")
    roots,results=parse(process.stdout,expected)
    panels={ROLES[k]:{t:panel(results,k,t) for t in (0,1)} for k in (1,2)}
    examples={ROLES[k]:cases(results,k) for k in (1,2)}
    counts={role:sum(a["actions"][i]>=0 for a in roots.values()) for i,role in enumerate(ROLES)}
    rows=[dict(key=key,**root,rollouts=[[t,r,a,*out.values()] for (t,r,a),out in results[key].items()])
          for key,root in roots.items()]
    receipt=dict(provenance=provenance,panels=panels,examples=examples,counts=counts,rows=rows)
    headline=dict(counts=counts,panels=panels,rollouts=sum(map(len,results.values())),
                  cases={k:{c:len(keys) for c,keys in groups.items()} for k,groups in examples.items()})
    if resample:
        original=json.loads(ORIGINAL.read_bytes())
        require(original["provenance"]==provenance,"resample inputs changed")
        receipt["changes"]=headline["changes"]=changes(original,roots,examples)
    write_result(directory,receipt,audit)
    print(json.dumps(headline,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
