"""Extend the same verified takeover trials and audit the old 240-ply prefix."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from probe_replay_midgame_takeover import (
    ENDS, MODES, ROOT, compile_native, parse, read, sample,
)

PREVIOUS = ROOT / "build/replay-midgame-takeover/probe-replay-midgame-takeover-b6f4d2def4257a4e6af1"
OLD_CAP = 240
CAP = 960
MILESTONES = ("contact","enemy_loss","enemy_exit","own_exit")


def verify_extension(old_text,new_text,old_evidence,new_evidence):
    if new_evidence["manifest"] != old_evidence["manifest"]:
        raise ValueError("a policy, input or rules file changed")
    sources = old_evidence["manifest"]["sources"]
    before = parse(old_text,sources,OLD_CAP)
    after = parse(new_text,sources,CAP)
    if before[:2] != after[:2] or before[3] != after[3] or before[2].keys() != after[2].keys():
        raise ValueError("recorded roots, human outcomes, or trial grid changed")
    changes = {}
    for table in range(2):
        for mode, label in MODES.items():
            counts = Counter()
            for key, old in before[2].items():
                if (key[2],key[4]) != (table,mode):
                    continue
                new = after[2][key]
                if old["ending"]!=5:
                    if old!=new:
                        raise ValueError("already settled trajectory changed")
                    counts["settled_before"]+=1
                    continue
                if new["plies"]<=OLD_CAP or new["ending"]==0:
                    raise ValueError("truncated old trajectory did not continue")
                for field in ("first_from","first_to","first_contact"):
                    if old[field]!=new[field]:
                        raise ValueError("opening action changed with the cap")
                for name in MILESTONES:
                    old_step = old[name]
                    new_step = new[name] if new[name] <= OLD_CAP else -1
                    if old_step != new_step:
                        raise ValueError("a prior milestone changed with the cap")
                counts["old_cutoffs"]+=1
                counts[ENDS[new["result"]]]+=1
            changes[f"{table}/{label}"]=dict(counts)
    return changes


def main(audit):
    key = os.environ.get("JUNQI_SERIAL_KEY","manual")
    if not key or not all(c.isalnum() or c=="-" for c in key):
        raise ValueError("invalid local serial key")
    out = ROOT / "build/replay-midgame-horizon" / key
    started = time.perf_counter()
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        compile_native(out)
    data,text = sample(out,CAP)
    prior = read(PREVIOUS / "summary.json")["evidence"]
    old_rows = (PREVIOUS / "rows.txt").read_text(encoding="utf-8")
    data["extension"] = verify_extension(old_rows,text,prior,data)
    end = out / "summary.json"
    if audit:
        if read(end)["evidence"]!=data or (out / "rows.txt").read_text(encoding="utf-8")!=text:
            raise ValueError("longer takeover trials are not reproducible")
    else:
        (out / "rows.txt").write_text(text,encoding="utf-8")
        temporary=end.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(evidence=data,wall_seconds=round(time.perf_counter()-started,3)),
                                        indent=2,sort_keys=True)+"\n",encoding="utf-8")
        temporary.replace(end)
    conclusion = {
        table: {name: dict(outcomes=row["outcomes"],win=row["win"])
                for name,row in results.items() if name in MODES.values()}
        for table,results in data["summary"]["tables"].items()
    }
    print(json.dumps(dict(extension=data["extension"],cap=CAP,results=conclusion),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
