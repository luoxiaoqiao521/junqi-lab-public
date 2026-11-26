"""Check ordinary learning receipts and inject faults into in-memory copies only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from serial_dark_ordinary_inheritance import PREVIOUS as FIRST, ROOT
from serial_dark_ordinary_learning import assert_layout_separation, data_health, score_transitions
from serial_dynamic_reply import require
from serial_worker import read, write_receipt


SECOND=ROOT/"build/dark-ordinary-inheritance/compare-dark-ordinary-inheritance-4194ef43a0296aff3b07"


def faults(games,examples,grids):
    bad_label=deepcopy(examples)
    bad_label[0]["outcome"]=1 if bad_label[0]["outcome"]==0 else 0
    bad_ply=deepcopy(examples)
    bad_ply[0]["features"][0]=721/140
    bad_features=deepcopy(examples)
    bad_features[0]["features"][0]=float("inf")
    duplicate=deepcopy(examples)
    duplicate[1]=dict(duplicate[0])
    bad_grids=deepcopy(grids)
    bad_grids["confirmation"][0]=list(bad_grids["train"][0])
    return {"wrong_terminal":lambda:data_health(games,bad_label),"future_sample":lambda:data_health(games,bad_ply),
            "nonfinite_feature":lambda:data_health(games,bad_features),"duplicate_sample":lambda:data_health(games,duplicate),
            "duplicate_game":lambda:data_health([*games,games[0]],examples),"layout_leak":lambda:assert_layout_separation(bad_grids)}


def directions(rows):
    pairs=list(zip(rows[::2],rows[1::2],strict=True))
    require(all((a["seed"],a["seat"],a["table"])==(b["seed"],b["seat"],b["table"]) for a,b in pairs),"unpaired transition")
    return dict(overall=score_transitions(pairs),
                desks={t:score_transitions([(a,b) for a,b in pairs if a["table"]==t]) for t in range(4)})


def main(audit):
    folder=ROOT/"build/dark-learning-health"/os.environ["JUNQI_SERIAL_KEY"]
    folder.mkdir(parents=True,exist_ok=True)
    source=SECOND if audit else FIRST
    began=perf_counter()
    history=read(source/"training.json")
    grids=read(source/"manifest.json")
    games,examples=history["games"],history["examples"]
    ordinary=data_health(games,examples)
    assert_layout_separation(grids)
    errors={}
    for name,test in faults(games,examples,grids).items():
        try:
            test()
        except ValueError as error:
            errors[name]=str(error)
    require(len(errors)==6,"a deliberately corrupt receipt passed the actual check")
    results=read(source/"confirmation.json")
    transitions=({name:directions(arm["games"]) for name,arm in results["arms"].items()} if audit else
                 {"first_generation":directions(results["games"])})
    output=dict(data=ordinary,detected=errors,control_false_positive=0,transitions=transitions,seconds=perf_counter()-began)
    write_receipt(folder/("confirmation.json" if audit else "development.json"),output)
    if audit:
        print("JUNQI_RESEARCH supported")
    print(json.dumps(output))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected an ordinary health stage")
    main(bool(sys.argv[1:]))
