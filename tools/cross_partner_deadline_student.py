"""Measure the effect of training on two partner distributions at equal label budget."""
from collections import Counter
from hashlib import sha256
import json
import os
import sys

from export_early_deadline_teacher_pairs import data
from measure_deadline_opening import ROOT, build, paired, require
from train_early_deadline_student import SOURCE, WIDTH, evaluation, fit_student, panels


SEED, COUNT = 1093600, 64


def keyed(lines):
    rows = [json.loads(line) for line in lines.splitlines()]
    train = [r for r in rows if r["partition"]=="development_train"]
    groups = {(r["family"],r["actor_seat"],r["table"],r["phase"]):r for r in train}
    require(len(groups)==len(train) and all(r["version"]=="junqi-early-deadline-outcomes-1" for r in rows),
            "invalid or duplicate partner training roots")
    return groups


def subset(row, size):
    original = str(row["choices"]["pressure"])
    order = [original,*(m for m in sorted(row["branches"],key=int) if m!=original)]
    return row | {"branches":{m:row["branches"][m] for m in order[:size]}}


def lessons(legacy, alternative):
    old,new = keyed(legacy),keyed(alternative)
    common = sorted(old.keys() & new.keys())
    require(len(common)>=250, "too few comparable early windows")
    hist,mix = [],[]
    counts = Counter()
    strata = Counter()
    for k in common:
        budget = min(len(old[k]["branches"]),len(new[k]["branches"]))
        stratum = k[1:]
        partner = "heuristic" if strata[stratum]%2 else "historical"
        strata[stratum]+=1
        counts[partner]+=budget
        hist.append(subset(old[k],budget))
        mix.append(subset(new[k] if partner=="heuristic" else old[k],budget))
    require(sum(counts.values())==sum(len(r["branches"]) for r in hist)
            and min(counts[p] for p in ("historical","heuristic"))>500, "invalid mixed training budget")
    excluded = {"historical":len(old)-len(common),"heuristic":len(new)-len(common)}
    return hist,mix,dict(counts),excluded


def compare(arms):
    results = panels(arms)
    for partner in (1,2):
        for table in (0,1):
            cell = results[f"{partner}/{table}"]
            hist,mix = ([r for r in arms[name] if (r["partner"],r["table"])==(partner,table)]
                        for name in ("historic_teacher","mixed_teacher"))
            delta = paired(hist,mix,cell["historic_teacher"],cell["mixed_teacher"])
            delta["win_to_loss"] = sum(a["outcome"]==2 and b["outcome"]==0 for a,b in zip(hist,mix,strict=True))
            delta["loss_to_win"] = sum(a["outcome"]==0 and b["outcome"]==2 for a,b in zip(hist,mix,strict=True))
            cell["mixed_vs_historic"] = delta
    return results


def persist(path, content, audit):
    if audit:
        require(path.read_bytes()==content, f"did not reproduce {path.name}")
    else:
        pending = path.with_suffix(".tmp")
        pending.write_bytes(content)
        pending.replace(path)


def encode(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def main(audit):
    directory = ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    alt,description = data(directory,"early-teacher-heuristic")
    persist(directory/"heuristic-teacher.jsonl",alt.encode(),audit)
    hist,mix,counts,excluded = lessons(SOURCE.read_text(encoding="utf8"),alt)
    paths=[]
    for name,rows in (("historic_teacher",hist),("mixed_teacher",mix)):
        digest=sha256(encode(rows)).hexdigest()
        artifact=fit_student(directory,rows,((name,WIDTH),),digest)
        path=directory/f"{name}.json"
        persist(path,encode(artifact),audit)
        paths.append(path)
    arms = evaluation(directory,tuple(json.loads(p.read_bytes()) for p in paths),SEED,COUNT)
    result = compare(arms)
    receipt = dict(counts=counts,roots=len(hist),excluded_roots=excluded,alternative=description,result=result,games=arms)
    persist(directory/"summary.json",encode(receipt),audit)
    print(json.dumps({"roots":len(hist),"mixed_labels":counts,"result":result},sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
