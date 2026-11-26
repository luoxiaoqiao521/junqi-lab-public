"""Audit an observed action reversal using only archive-reachable worlds."""
from collections import Counter
from hashlib import sha256
import json
import os
import sys

from serial_replay_phase import INTAKE, ROOT, TAGS, build, native, payload, require, select, unpack, write_receipt

PRIOR = ROOT/"build/replay-phase/compare-student-error-coverage-by-phase-acd3593acb18f4a53f1a/summary.json"


def witnesses():
    roots = json.loads(PRIOR.read_bytes())["rows"]
    cut = {(r["key"],r["phase"]):r for r in roots if r["format"]==".jgs" and r["status"]=="direction_flip"}
    require(len(cut)==5,"expected five registered diagnostic roots")
    probe = json.loads((INTAKE/"probe.json").read_bytes())["rows"]
    sources = {r["key"]:r|{p:r[p] if (r["key"],p) in cut else 0 for p in TAGS}
               for r in probe if any((r["key"],p) in cut for p in TAGS)}
    return cut,sources


def inspect(output,cuts,sources):
    log = {}
    moves = []
    for line in output.splitlines():
        if line.startswith("W "):
            _,key,kind,*numbers = line.split()
            phase = TAGS[int(kind)]
            attempts,valid_layouts,valid_prefixes,worlds = map(int,numbers)
            require((key,phase) in cuts and (key,phase) not in log and
                    0<=worlds<=min(6,valid_prefixes)<=valid_layouts<=attempts<=96,
                    "invalid prefix witness receipt")
            log[key,phase] = dict(attempts=attempts,legal_deployments=valid_layouts,
                                  exact_prefixes=valid_prefixes,distinct_worlds=worlds)
        else:
            moves.append(line)
    rows = unpack("\n".join(moves),sources)
    require(set(log)==set(cuts),"missing prefix witness receipt")
    for row in rows:
        key = row["key"],row["phase"]
        old = cuts[key]
        entry = log[key]
        require(row["worlds"]==entry["distinct_worlds"] and
                all(old[f]==row[f] for f in ("ply","student","human","offered","legal")),
                "changing hypothetical worlds changed the fixed action comparison")
        row.update(entry,projection_bounds=old["panel_score_bounds_x2"])
    return rows


def main(audit):
    out = ROOT/"build/replay-phase"/os.environ["JUNQI_SERIAL_KEY"]
    out.mkdir(parents=True,exist_ok=True)
    cuts,sources = witnesses()
    wire = payload(sources)
    if not audit:
        build(out)
    raw = native(out,"replay-phase-witness",wire)
    rows = inspect(raw,cuts,sources)
    status = Counter(row["status"] for row in rows)
    report = dict(input_sha256=sha256(wire.encode()).hexdigest(),status=dict(status),rows=rows)
    if audit:
        require((out/"branches.txt").read_text(encoding="utf8")==raw and
                json.loads((out/"summary.json").read_bytes())==json.loads(json.dumps(report)),
                "exact replay audit did not reproduce")
        prior_cases = select(json.loads((INTAKE/"probe.json").read_bytes()))
        unchanged = native(out,"replay-phase-split",payload(prior_cases))
        require((PRIOR.parent/"branches.txt").read_text(encoding="utf8")==unchanged,
                "witness mode changed the original phase projection")
        print("JUNQI_RESEARCH "+("supported" if status["direction_flip"] else "inconclusive"))
    else:
        (out/"branches.txt").write_text(raw,encoding="utf8")
        write_receipt(out/"summary.json",report)
    print(json.dumps({"status":status,"exact_prefixes":sum(r["exact_prefixes"] for r in rows),
                      "games":sum(r["games"] for r in rows),"plies":sum(r["plies"] for r in rows)}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
