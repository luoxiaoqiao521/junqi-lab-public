"""Confirm the frozen third-generation student on new layouts without retraining."""
import json
import os
import shutil
import sys
from statistics import mean

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dynamic_reply import ROOT, require
from serial_formation_matrix import contrast, score_difference
from serial_full_mode_inheritance import SOURCE, paired_scores
from serial_full_student_evolution import games
from serial_worker import read, write_receipt

INHERITANCE=ROOT/"build/full-mode-inheritance/verify-inheritance-with-matched-frozen-controls-df26d90117073d1050bc"
STARTS=(2861100,2861200,2861300,2861400)


def frozen_inputs(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    source=SOURCE/"cycle-3"
    generation=INHERITANCE/"generation-3"
    for filename in ("rules.json","deadline_opening.exe"):
        shutil.copyfile(source/"engine"/filename,scope.engine/filename)
    for name,location in (("bundle.json",source/"inputs.json"),
                          ("reset.json",generation/"reset-student.json"),
                          ("inherited.json",generation/"inherited-student.json")):
        shutil.copyfile(location,scope.folder/name)
    bundle=read(scope.folder/"bundle.json")
    require(sha(scope.binary)==bundle["engine_sha"] and sha(scope.rules)==bundle["inputs"]["rules"],
            "confirmation engine or rule set changed")
    evidence={p.name:sha(p) for p in (scope.binary,scope.rules,scope.folder/"bundle.json",
                                     scope.folder/"reset.json",scope.folder/"inherited.json")}
    write_receipt(scope.folder/"frozen.json",evidence)


def panels(scope,starts):
    root=scope.folder
    evidence=read(root/"frozen.json")
    for name,expected in evidence.items():
        path=scope.engine/name if name in ("rules.json","deadline_opening.exe") else root/name
        require(sha(path)==expected,"locked confirmation artifact changed")
    parent=read(root/"bundle.json")["parent"]["models"]["window"]
    reset=read(root/"reset.json")
    inherited=read(root/"inherited.json")
    for start in starts:
        with isolated(scope,ROOT/"build"):
            control=games(scope.engine,parent,reset,start)
            trial=games(scope.engine,parent,inherited,start)
        paired=paired_scores(control,trial,start)
        write_receipt(root/f"panel-{start}.json",dict(reset=control,inherited=trial,paired=paired))
        print(f"Independent layout block {start}: {paired['clustered']}",flush=True)


def summarize(scope):
    pairs=dict(parent={},reset={})
    for start in STARTS:
        grid=read(scope.folder/f"panel-{start}.json")
        for name,arm in (("parent",3),("reset",4)):
            before={(r["seed"],r["table"]):r for r in grid["reset"]["rows"] if r["arm"]==arm}
            after={(r["seed"],r["table"]):r for r in grid["inherited"]["rows"] if r["arm"]==4}
            require(before.keys()==after.keys() and not pairs[name].keys()&before.keys(),"duplicate or missing layouts")
            pairs[name].update({k:(0.,0.) if r["final_hash"]==after[k]["final_hash"] else
                                score_difference(r["result"],after[k]["result"]) for k,r in before.items()})
    result={}
    for name,rows in pairs.items():
        seeds=sorted({s for s,_ in rows})
        result[name]=dict(paired_layouts=len(seeds),tables={t:contrast([rows[s,t] for s in seeds]) for t in range(3)},
                          clustered=contrast([tuple(map(mean,zip(*(rows[s,t] for t in range(3)),strict=True)))
                                               for s in seeds]))
    result["natural_finishes"]=sum(r["result"]>=0 for start in STARTS
                                   for panel in read(scope.folder/f"panel-{start}.json").values()
                                   if "rows" in panel for r in panel["rows"])
    return result


def main(audit):
    root=ROOT/"build/full-mode-inheritance-confirmation"/os.environ["JUNQI_SERIAL_KEY"]
    scope=FullModeRun(root)
    if not audit:
        frozen_inputs(scope)
        panels(scope,STARTS[:2])
    else:
        panels(scope,STARTS[2:])
        result=summarize(scope)
        write_receipt(root/"summary.json",result)
        print(json.dumps(result))
        lower=[result[name]["clustered"]["interval"][0] for name in ("parent","reset")]
        upper=[result[name]["clustered"]["interval"][1] for name in ("parent","reset")]
        verdict="supported" if min(lower)>0 else "refuted" if min(upper)<0 else "inconclusive"
        print("JUNQI_RESEARCH "+verdict)
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
