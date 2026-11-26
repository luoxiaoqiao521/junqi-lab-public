"""Diagnose what the frozen short scorer and student can learn from full visibility."""
import json
import os
import sys

from measure_human_winner_takeover import MODEL, ROOT, build, model_arguments, native, require
from serial_worker import write_receipt


def main(audit):
    directory = ROOT/"build/visible-learning"/os.environ["JUNQI_SERIAL_KEY"]
    directory.mkdir(parents=True,exist_ok=True)
    artifact = json.loads(MODEL.read_bytes())
    arguments = model_arguments(artifact,"replay_control")
    if not audit:
        build(directory)
    observed = [list(map(float,row.split())) for row in native(directory,"audit-visible-learning",arguments).splitlines()]
    require(len(observed)==4 and all(len(row)==6 and row[0]==seat and row[1]>=2 and row[2:]==[1,0,.5,.5]
                                    for seat,row in enumerate(observed)), "missing rotation contrast")
    bounded = native(directory,"prove-evaluator-controls","")
    controls = [row.split()[0] for row in bounded.splitlines()]
    require(len(controls)==12 and len(set(controls))==12,"missing bounded planning controls")
    report = dict(rotations=observed, bounded_controls=bounded.splitlines(),
                  score_paths=["short alias","frozen 41 alias","44 visible contrast","44 dark invariance"],
                  full_rejected_by_dark_history=True)
    if audit:
        require(json.loads((directory/"summary.json").read_bytes())==report,"visible learning evidence did not replay")
        print("JUNQI_RESEARCH supported")
    else:
        write_receipt(directory/"summary.json",report)
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
