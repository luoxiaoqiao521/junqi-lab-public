"""Calibrate evaluation conditions before ranking the value of opening data."""
from hashlib import sha256
import json
import subprocess
import sys
import tempfile
import time

from audit_public_attack_windows import write_json
from partner_crossplay import model_roster
from replay_deadline_transfer import encode_layout
from validate_layout_source_transfer import (
    EXE, METRICS, OUT, RESULT as PREVIOUS, compile_probe, freeze_inputs, summarize,
)

RESULT = OUT / "pressure.json"
SMALL, FULL = 800, 1600


def load_prior():
    raw = PREVIOUS.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def with_budget(rows, limit):
    return [{k: v for k, v in row.items() if k != "budget"} for row in rows if row["budget"] == limit]


def value(rows):
    plies = sum(row["plies"] for row in rows)
    cutoff = sum(row["score"] < 0 for row in rows)
    decisive = sum(row["score"] in (0, 2) for row in rows)
    return dict(games=len(rows), plies=plies, cutoff=cutoff, decisive=decisive,
                natural=len(rows)-cutoff, decisive_per_10000=10000*decisive/plies)


def summarize_pressure(rows):
    results = {}
    for table in (0,1):
        low = [r for r in rows if r["table"] == table and r["budget"] == SMALL]
        high = [r for r in rows if r["table"] == table and r["budget"] == FULL]
        first, last = value(low), value(high)
        extra = last["plies"]-first["plies"]
        added = last["natural"]-first["natural"]
        decisive = last["decisive"]-first["decisive"]
        results[str(table)] = dict(at_800=first, at_1600=last, extra_plies=extra,
                                   recovered=added, recovered_decisive=decisive,
                                   outcomes_per_added_10000=10000*added/extra if extra else None,
                                   decisive_per_added_10000=10000*decisive/extra if extra else None)
    return dict(profiles=results, layout_deltas={str(limit):summarize(with_budget(rows,limit))
                                               for limit in (SMALL,FULL)})


def simulate_rows(batches, roster, mode="pressure", extra=()):
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as feed:
        print(len(batches),file=feed)
        print(*roster["weights"],file=feed)
        for idx, pair in enumerate(batches):
            print(635800+idx,file=feed)
            for layout in pair:
                print(*encode_layout(layout),file=feed)
        feed.seek(0)
        begin=time.perf_counter()
        process=subprocess.run([EXE,mode],stdin=feed,check=True,capture_output=True,text=True,timeout=145)
        elapsed=time.perf_counter()-begin
    rows = [dict(zip((*METRICS,"budget",*extra),map(int,line.split()),strict=True))
            for line in process.stdout.splitlines()]
    return rows, elapsed


def run():
    previous, baseline = load_prior()
    batches, info = freeze_inputs()
    roster = model_roster()["candidate"]
    if previous["source"] != info or previous["model"] != roster["artifact_sha256"]:
        raise ValueError("baseline inputs or model changed")
    rows, elapsed = simulate_rows(batches,roster)
    write_json(RESULT,dict(baseline=baseline,model=roster["artifact_sha256"], rows=rows,
                           elapsed_seconds=elapsed, summary=summarize_pressure(rows)))
    audit()


def audit():
    previous, baseline=load_prior()
    r=json.loads(RESULT.read_text(encoding="utf8"))
    rows=r["rows"]
    assert r["baseline"]==baseline and r["model"]==previous["model"]
    low, high = (with_budget(rows,limit) for limit in (SMALL,FULL))
    order = ("block","source","orientation","table","seat","first","seed")
    expected = [tuple(row[k] for k in order) for row in previous["rows"]]
    assert len(rows)==2*len(expected)
    assert all([tuple(row[k] for k in order) for row in points]==expected for points in (low,high))
    assert [row for row in low if row["table"]==0]==[row for row in previous["rows"] if row["table"]==0]
    for early, late in zip(low,high,strict=True):
        assert early["score"] in (-1,0,1,2) and late["score"] in (-1,0,1,2)
        assert 0<early["plies"]<=SMALL and early["plies"]<=late["plies"]<=FULL
        assert (late["score"]!=-1 or late["plies"]==FULL)
        if early["score"]>=0:
            assert late==early
        else:
            assert early["plies"]==SMALL and late["plies"]>SMALL
            assert all(early[k]<=late[k] for k in ("contacts","no_moves","own_no_move","own_flag_taken"))
            assert all(early[k]==late[k] for k in ("first_contact","first_flag","flag_front") if early[k])
    assert r["summary"]==summarize_pressure(rows)
    print("PASS 768 random checkpoints equal previous run; natural results preserved")
    print(json.dumps(r["summary"]["profiles"],sort_keys=True))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
