"""Measure what changes with the selfplay retirement policy on existing paired blocks."""
from hashlib import sha256
import json
from statistics import mean
import sys

from audit_public_attack_windows import write_json
from calibrate_layout_evaluation_pressure import FULL, OUT, RESULT as BASELINE, simulate_rows, value
from partner_crossplay import model_roster
from validate_layout_source_transfer import (
    ARMS, compile_probe, freeze_inputs, interval, combine_orientations, stat_bounds, summarize,
)

RESULT = OUT / "retirement.json"
SPLIT = ("split_ply", "archive_mask", "training_mask")
MATCH = ("block","source","orientation","table","seat","first","seed")


def baseline_input():
    raw = BASELINE.read_bytes()
    rows = [r for r in json.loads(raw)["rows"] if r["budget"]==FULL]
    return rows, sha256(raw).hexdigest()


def compare(old, new):
    result = {}
    for table in (0,1):
        comparisons = []
        for source, name in enumerate(ARMS):
            pairs = [(a,b) for a,b in zip(old,new,strict=True) if a["table"]==table and a["source"]==source]
            changed = [b for _,b in pairs if b["split_ply"]]
            deltas = [stat_bounds(b["score"],a["score"]) for a,b in pairs]
            comparisons.append((name,dict(old=value([a for a,_ in pairs]),new=value([b for _,b in pairs]),
                   split=len(changed),
                   archive_only_at_first_split=sum(bool(r["archive_mask"]&~r["training_mask"]) for r in changed),
                   training_only_at_first_split=sum(bool(r["training_mask"]&~r["archive_mask"]) for r in changed),
                   no_mask_difference_at_first_split=sum(r["archive_mask"]==r["training_mask"] for r in changed),
                   raised=sum(min(a["score"],b["score"])>=0 and b["score"]>a["score"] for a,b in pairs),
                   lowered=sum(min(a["score"],b["score"])>=0 and b["score"]<a["score"] for a,b in pairs),
                   bounds=list(map(mean,zip(*deltas))), interval=interval(combine_orientations(deltas)))))
        result[str(table)] = dict(comparisons)
    return result


def run():
    old, base = baseline_input()
    batches, info = freeze_inputs()
    roster = model_roster()["candidate"]
    rows, elapsed = simulate_rows(batches,roster,mode="retirement",extra=SPLIT)
    final = [r for r in rows if r["budget"]==FULL]
    layout = summarize(final)
    write_json(RESULT,dict(baseline=base, sources=info, model=roster["artifact_sha256"], rows=final,
                           elapsed_seconds=elapsed, rule_deltas=compare(old,final), layout=layout))
    audit()


def audit():
    old, base = baseline_input()
    data = json.loads(RESULT.read_text(encoding="utf8"))
    assert data["baseline"]==base and data["model"]==model_roster()["candidate"]["artifact_sha256"]
    for left,right in zip(old,data["rows"],strict=True):
        assert all(left[key]==right[key] for key in MATCH)
        assert right["score"] in (-1,0,1,2) and 0<right["plies"]<=FULL
        if right["split_ply"]==0:
            assert not right["archive_mask"] and not right["training_mask"]
            assert left=={k:v for k,v in right.items() if k not in SPLIT}
        else:
            assert 0<right["split_ply"]<=min(left["plies"],right["plies"])
            assert all(0<=right[k]<16 for k in SPLIT[1:])
    assert data["rule_deltas"]==compare(old,data["rows"])
    assert data["layout"]==summarize(data["rows"])
    print("PASS paired rules",len(old),"first differences",sum(row["split_ply"]>0 for row in data["rows"]))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
