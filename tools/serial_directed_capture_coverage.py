"""Replay historical identity branches within a shared experimental work envelope."""
from hashlib import sha256
import json
import os
from statistics import median
import sys

from measure_deadline_opening import ROOT, build, require
from serial_capture_history import SOURCE, select
from serial_capture_replies import comparisons, sources
from serial_capture_world_coverage import inputs as prior_inputs, MODEL, EXCHANGE
from serial_visible_student_compare import collect as reply_panel, independent_sources
from serial_worker import write_receipt
from train_early_deadline_student import model_arguments, native

PANEL = ("attempts","nodes","replays","replay_plies","worlds","replay_failures","contradicted","unresolved",
         "unavailable","sides","outcomes","targeted","excluded","microseconds")
ARMS = ("old","direct","targeted")
KINDS = ("bomb","engineer","high_rank")
TRIAD = (4260,4337,4332)


def inputs(audit):
    if not audit:
        return {key:" 0 ".join(wire.rsplit(" ",1))
                for key,wire in prior_inputs(False).items()}
    census = json.loads((SOURCE/"probe.json").read_bytes())["rows"]
    old = sources()
    excluded_keys = set(old)|set(independent_sources(old))|set(prior_inputs(True))
    excluded = {row["family"] for row in census if row["key"] in excluded_keys}
    excluded.update(family for heldout in (False,True) for _,family in select(heldout)[0].values())
    pool = (r for r in census if r["format"]==".jgs" and r["mid"] and r["result"] in (1,2,3) and
            not any(r[k] for k in ("prior_layout","prior_trajectory","stopped","notice")))
    keys = {}
    for row in sorted(pool,key=lambda r:sha256(("decision-branch-v1:"+r["family"]).encode()).digest()):
        if row["family"] not in excluded:
            keys[row["key"]] = (row,len(keys)%3)
            excluded.add(row["family"])
        if len(keys)==12:
            break
    wires = {}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape = line.split(maxsplit=2)
        if key in keys:
            require(tag=="G","expected an archived JGS prefix")
            row,preferred = keys[key]
            wires[key] = f'{key} {row["learner"]} {row["mid"]-1} {row["direction"]} 2026112331 {preferred} {tape}'
    require(len(wires)==12 and wires.keys()==keys.keys(),"missing registered independent layouts")
    return wires


def read_rows(output, wires):
    rows = {}
    for line in output.splitlines():
        tag,key,*values = line.split()
        require(key in wires,"unexpected branch root")
        values = list(map(int,values))
        if tag=="T":
            cut,kind,action = values
            require(key not in rows and -1<=kind<3 and (kind==-1)==(action==-1),"invalid branch selector")
            rows[key] = dict(cut=cut,kind=kind,action=action,arms={})
        elif tag=="C":
            moves,witnesses,invariant,negative = values
            require(moves==rows[key]["cut"] and witnesses==invariant,"visibility control changed with hidden state")
            rows[key]["controls"] = dict(witnesses=witnesses,negative_ply=negative)
        elif tag=="P":
            arm,*values = values
            name = ARMS[arm]
            panel = dict(zip(PANEL,values,strict=True))
            require(name not in rows[key]["arms"] and 0<=panel["worlds"]<=panel["replays"]-panel["replay_failures"] and
                    panel["replays"]+panel["contradicted"]+panel["unresolved"]+panel["unavailable"]==panel["attempts"] and
                    panel["nodes"]<=4096 and panel["attempts"]<=48 and panel["replays"]<=4 and
                    not (panel["sides"]&panel["excluded"]),"shared budget or branch status is inconsistent")
            rows[key]["arms"][name] = panel
        else:
            require(tag=="W","unknown branch evidence record")
    require(rows.keys()==wires.keys() and all(set(row["arms"])==set(ARMS) and "controls" in row for row in rows.values()),
            "missing archived branch coverage panel")
    return rows


def collect(directory, wires):
    model = json.loads(MODEL.read_bytes())
    wire = "\n".join([model_arguments(model,"collision",EXCHANGE),*wires.values()])+"\n"
    return read_rows(native(directory,"directed-capture-coverage",wire),wires)


def tally(rows):
    result = {}
    for arm in ARMS:
        panels = [row["arms"][arm] for row in rows.values()]
        result[arm] = {k:sum(p[k] for p in panels) for k in ("worlds","nodes","replay_plies","microseconds","unresolved","targeted")}
        result[arm].update(covered=sum(p["sides"]==3 for p in panels),
                           outcome_contrasts=sum(p["sides"]==3 and p["outcomes"].bit_count()>1 for p in panels),
                           zero_roots=sum(p["worlds"]==0 for p in panels),
                           median_microseconds=median(p["microseconds"] for p in panels))
    return {"sources":len(rows),"eligible":{kind:sum(row["kind"]==i for row in rows.values()) for i,kind in enumerate(KINDS)},
            "controls":sum(row["controls"]["witnesses"] for row in rows.values()),"arms":result}


def stable(rows):
    return {key:{**row,"arms":{name:{k:v for k,v in panel.items() if k!="microseconds"}
                              for name,panel in row["arms"].items()}} for key,row in rows.items()}


def verify_normal(rows):
    # A public no-trigger must leave the old proposal path and seeds intact.
    return all(r["arms"]["old"]==r["arms"]["targeted"] for r in stable(rows).values() if r["kind"]<0)


def triad(directory):
    archive = independent_sources(sources())
    key, = [key for key in archive if key.startswith("2bb91adf1b33")]
    results = []
    for seeds in ((2026112347,2026112371),(2026112348,2026112372)):
        a,b = (reply_panel(directory,archive,seeds,{key:move})[key] for move in TRIAD[:2])
        require(a["trace"]==b["trace"] and a["original"]==b["original"]==TRIAD[2] and
                a["worlds"]==b["worlds"] and
                a["actions"].get(TRIAD[2])==b["actions"].get(TRIAD[2]),"triad did not share the same frozen baseline")
        grid = {}
        if a["worlds"]>0:
            all_actions = a["actions"]|b["actions"]
            for left,right in ((0,1),(0,2),(1,2)):
                pair = a|{"original":TRIAD[right],"actions":all_actions}
                grid[f"{TRIAD[left]}-{TRIAD[right]}"] = comparisons(pair,TRIAD[left])
        results.append({"worlds":a["worlds"],"seeds":seeds,"difference_x2_by_desk":grid})
    return results


def main(audit):
    directory = ROOT/"build/directed-capture-coverage"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
        rows = collect(directory,inputs(False))
        require(all(r["controls"]["negative_ply"]>0 for r in rows.values()),"negative collision control missing")
        write_receipt(directory/"registered.json",rows)
    else:
        registered = json.loads((directory/"registered.json").read_bytes())
        require(stable(registered)==stable(collect(directory,inputs(False))),"registered behavior changed on replay")
        rows = collect(directory,inputs(True))
        require(verify_normal(rows),"hybrid changed a no-trigger decision's ordinary proposals")
        totals = tally(rows)["arms"]
        report = dict(rows=rows,tally=tally(rows),triad=triad(directory),
                      engine_sha256=sha256((directory/"deadline_opening.exe").read_bytes()).hexdigest(),
                      model_sha256=sha256(MODEL.read_bytes()).hexdigest())
        write_receipt(directory/"holdout.json",report)
        print(json.dumps({"triad":report["triad"]}))
        better = totals["targeted"]["outcome_contrasts"]>max(totals[k]["outcome_contrasts"] for k in ARMS[:2])
        print("JUNQI_RESEARCH "+("supported" if better else "inconclusive"))
    print(json.dumps(tally(rows)))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
