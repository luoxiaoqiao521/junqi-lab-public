"""Compare bounded historical capture witnesses and verify visible-feature isolation."""
from hashlib import sha256
import json
import os
import sys

from measure_deadline_opening import ROOT, build, require
from serial_capture_history import SOURCE, select
from serial_capture_replies import sources
from serial_visible_student_compare import MODEL, EXCHANGE, FRESH_SEED, independent_sources
from serial_worker import write_receipt
from train_early_deadline_student import model_arguments, native

KNOWN = ("78791f48bdcc", "2bb91adf1b33")
PANEL = ("attempts", "nodes", "unavailable", "contradicted", "unresolved", "joint_witnesses", "replay_failures",
         "replay_plies", "unique_worlds", "target_ranks", "collision_conflicts", "inventory_conflicts",
         "first_collision_conflict", "first_replay_failure", "microseconds")


def inputs(audit):
    census = json.loads((SOURCE/"probe.json").read_bytes())["rows"]
    previous = sources()
    fresh = independent_sources(previous)
    excluded = {r["family"] for r in census if r["key"] in previous or r["key"] in fresh}
    excluded.update(family for heldout in (False,True) for _,family in select(heldout)[0].values())
    if audit:
        eligible = (r for r in census if r["format"]==".jgs" and r["mid"] and r["result"] in (1,2,3)
                    and not any(r[field] for field in ("prior_layout","prior_trajectory","stopped","notice")))
        keys = {}
        for row in sorted(eligible,key=lambda r:sha256(("joint-coverage:audit:"+r["family"]).encode()).digest()):
            if row["family"] in excluded:
                continue
            keys[row["key"]] = row
            excluded.add(row["family"])
            if len(keys)==8:
                break
    else:
        keys = {r["key"]:r for r in census if r["key"] in fresh and r["key"].startswith(KNOWN)}
    wires = {}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape = line.split(maxsplit=2)
        if key not in keys:
            continue
        require(tag=="G","capture coverage expects a JGS prefix")
        row = keys[key]
        seed = FRESH_SEED[0] if not audit else 2026112321
        wires[key] = f'{key} {row["learner"]} {row["mid"]-1} {row["direction"]} {seed} {tape}'
    require(wires.keys()==keys.keys() and len(keys)==(8 if audit else 2),"missing selected coverage sources")
    return wires


def collect(directory, wires):
    model = json.loads(MODEL.read_bytes())
    wire = "\n".join([model_arguments(model,"collision",EXCHANGE),*wires.values()])+"\n"
    output = native(directory,"capture-world-coverage",wire)
    rows = {}
    for line in output.splitlines():
        tag,key,*columns = line.split()
        require(key in wires,"unknown capture coverage case")
        values = list(map(int,columns))
        if tag=="T":
            cut,target,legal = values
            rows[key] = dict(cut=cut,target=target,legal=legal,panels={})
        elif tag=="C":
            moves,worlds,invariant,negative_ply = values
            require(moves==rows[key]["cut"] and worlds==invariant,"capture control or visibility failed")
            rows[key]["controls"] = dict(moves=moves,nearby_worlds=worlds,invariant=invariant,negative_ply=negative_ply)
        elif tag=="P":
            mode,*metrics = values
            row = rows[key]
            p = dict(zip(PANEL,metrics,strict=True))
            require(0<=p["unique_worlds"]<=p["joint_witnesses"]-p["replay_failures"] and
                    p["joint_witnesses"]+p["contradicted"]+p["unresolved"]+p["unavailable"]==p["attempts"] and
                    0<=p["nodes"]<=(4096 if mode<2 else 1536),"invalid coverage accounting")
            if mode>=2:
                p["branch_status"] = ("witnessed" if p["unique_worlds"] else "contradicted" if p["contradicted"] else "unknown")
                require(not p["unique_worlds"] or bool(p["target_ranks"] & (1<<2))==(mode==3),"wrong identity branch")
            row["panels"][str(mode)] = p
        else:
            require(tag=="W","unexpected capture coverage output")
    require(rows.keys()==wires.keys() and all(set(r["panels"])==({"0","1","2","3"} if r["target"]>=0 else {"0","1"})
            for r in rows.values()),"missing coverage panels")
    return rows


def aggregate(rows):
    normal = [r for r in rows.values() if r["panels"]["0"]["unique_worlds"]>0]
    arms = {side:{field:sum(r["panels"][side][field] for r in rows.values()) for field in PANEL
                  if field not in ("target_ranks","first_collision_conflict","first_replay_failure")} for side in ("0","1")}
    return dict(sources=len(rows),controls=sum(r["controls"]["invariant"] for r in rows.values()),arms=arms,
                rescue=sum(not r["panels"]["0"]["unique_worlds"] and r["panels"]["1"]["unique_worlds"]>0 for r in rows.values()),
                negative_controls=sum(r["controls"]["negative_ply"]>0 for r in rows.values()),
                normal_sources=len(normal),normal_direct_us=sum(r["panels"]["1"]["microseconds"] for r in normal),
                normal_old_us=sum(r["panels"]["0"]["microseconds"] for r in normal))


def without_timing(rows):
    return {key:{**r,"panels":{mode:{k:v for k,v in p.items() if k!="microseconds"}
                               for mode,p in r["panels"].items()}} for key,r in rows.items()}


def main(audit):
    directory = ROOT/"build/capture-world-coverage"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
        rows = collect(directory,inputs(False))
        require(aggregate(rows)["negative_controls"] and all(r["controls"]["invariant"] for r in rows.values()),
                "missing illegal collision or alternate hidden-history control")
        write_receipt(directory/"controls.json",rows)
    else:
        controls = json.loads((directory/"controls.json").read_bytes())
        require(without_timing(controls)==without_timing(collect(directory,inputs(False))),"locked coverage controls changed")
        rows = collect(directory,inputs(True))
        write_receipt(directory/"holdout.json",rows)
        rescue = aggregate(controls)["rescue"]
        improved = aggregate(rows)["rescue"] or any(r["panels"]["1"]["unique_worlds"]>r["panels"]["0"]["unique_worlds"]
                                                    for r in rows.values())
        print("JUNQI_RESEARCH "+("supported" if rescue and improved else "inconclusive"))
    print(json.dumps(aggregate(rows)))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
