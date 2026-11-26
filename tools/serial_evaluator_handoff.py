"""Verify the hidden-information boundary before releasing the natural-opening ticket."""
import json

from serial_evaluator_panel import directory, native, require
from serial_worker import write_receipt


def main():
    path=directory()
    result=json.loads((path/"panel.json").read_text(encoding="utf8"))
    require(result["decision"]=="diagnostic_only","panel requires an explicit review decision")
    bounds=native(path,"prove-evaluator-controls")
    hidden=native(path,"conditional-reply-proof")
    require(all(label in bounds for label in ("E2_weak_take","E2_strong_take","E2_weak_quiet","E2_strong_quiet")) and
            len(hidden.splitlines())>0,"information boundary controls did not execute")
    write_receipt(path/"handoff.json",dict(e1=result["decision"],e2="boundary_regressions_passed",
        hidden_controls=hidden.splitlines(), world_controls=[line for line in bounds.splitlines() if line.startswith("E2_")],
        next_ticket="audit-first-enemy-retirement-progress",
        parked=["Blind E1 confirmation needs independently designed board structures.",
                "E2 verification needs reachable equivalent histories through the actual decision entry.",
                "The single-reply case 1202053 remains a diagnostic; causality at ply 53 is unproven."],
        adoption=False))
    print(hidden,end="")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    main()
