"""Try candidate coverage at a fixed takeover schedule and frozen weights."""
import hashlib
import json
import os
import sys
from time import perf_counter

from serial_dynamic_reply import PRIOR, ROOT, build, frozen_wire, scheduled_games
from serial_evaluator_panel import require
from serial_worker import read, write_receipt


def main(audit):
    began=perf_counter()
    out=ROOT/"build/student-candidate-coverage"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
    wire,peer=frozen_wire()
    weights=hashlib.sha256((PRIOR/"students.json").read_bytes()).hexdigest()
    results=scheduled_games(out,wire,audit,coverage=True)
    results.update(seconds=round(perf_counter()-began,3),weights=weights,peer=peer)
    if audit:
        previous=read(out/"development.json")
        require(previous["weights"]==weights and previous["peer"]==peer and
                {r["seed"] for r in previous["rows"]}.isdisjoint(r["seed"] for r in results["rows"]),
                "new layouts or frozen student artifact drifted")
        print("JUNQI_RESEARCH inconclusive")
    write_receipt(out/("confirmation.json" if audit else "development.json"),results)
    print(json.dumps(results["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
