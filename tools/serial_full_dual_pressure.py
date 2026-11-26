"""Verify seat-order pressure and bounded proof cutoffs through the serial entry."""
import json
import os
import subprocess
import sys

from measure_deadline_opening import ROOT, RULES, build, require
from serial_worker import write_receipt


def run_native(directory, mode):
    result = subprocess.run([directory/"deadline_opening.exe", RULES, mode], input="", encoding="utf8",
                            capture_output=True, timeout=35)
    require(result.returncode == 0, f"{mode} failed: {result.stderr.strip()}")
    return result.stdout


def pressure_controls(directory):
    names = ("rotation", "condition", "immediate_enemy_seats", "actions", "safe", "lost", "unknown",
             "chosen_action", "chosen_lower", "chosen_upper", "guard_lower", "guard_upper",
             "team_guard_lower", "team_guard_upper", "flag_nodes", "team_nodes")
    rows = []
    for line in run_native(directory, "dual-pressure").splitlines():
        tag, *fields = line.split()
        require(tag == "D", "unexpected dual-pressure output")
        rows.append(dict(zip(names, map(int, fields), strict=True)))
    require([(r["rotation"], r["condition"]) for r in rows] == [(s,c) for s in range(4) for c in range(4)],
            "incomplete pressure controls")
    for row in rows:
        case = row["condition"]
        safe = int(case in (0,3))
        require(row["immediate_enemy_seats"] == (1 if case == 3 else 2) and row["actions"] == 7 and
                (row["safe"], row["lost"], row["unknown"]) == (safe, 7-safe, 0),
                "turn order or flag survival contradicted the control")
        require(row["guard_lower"] == row["guard_upper"] == safe*2 and
                (row["team_guard_lower"], row["team_guard_upper"]) == (0,2),
                "expected flag witness or unresolved team horizon changed")
        require(0 <= row["chosen_lower"] <= row["chosen_upper"] <= 2 and
                0 <= row["team_nodes"] <= 12000 and 0 <= row["flag_nodes"] <= 12000*row["actions"],
                "invalid proof or node budget")
    return rows


def main(confirm):
    out = ROOT/"build/full-dual-pressure"/os.environ["JUNQI_SERIAL_KEY"]
    if not confirm:
        out.mkdir(parents=True, exist_ok=True)
        build(out)
        rows = pressure_controls(out)
        write_receipt(out/"pressure.json", rows)
    else:
        rows = pressure_controls(out)
        require(rows == json.loads((out/"pressure.json").read_bytes()), "pressure evidence did not replay exactly")
        cutoffs = run_native(out, "prove-evaluator-controls").splitlines()
        require(cutoffs[:2] == ["E1_max_first_child 1 2 2 1", "E1_min_first_child 1 0 0 1"],
                "bounded solver did not evaluate beyond the adverse first child")
        earlier = run_native(out, "full-mode-baseline").splitlines()
        require(len(earlier) == 4 and earlier[-1] == "X route_controls 2", "previous full-mode proof changed")
        improved_guard = sum(r["guard_lower"] > r["chosen_upper"] for r in rows if r["condition"] == 3)
        write_receipt(out/"confirmation.json", dict(pressure=rows, solver_controls=cutoffs, full_routes=earlier,
                                                   improved_guard_rotations=improved_guard))
        print("JUNQI_RESEARCH "+("supported" if improved_guard else "inconclusive"))
    print(json.dumps([r for r in rows if r["rotation"] == 0]))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    require(sys.argv[1:] in ([], ["audit"]), "expected audit or no argument")
    main(bool(sys.argv[1:]))
