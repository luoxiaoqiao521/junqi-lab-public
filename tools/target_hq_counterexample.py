"""Test an alternative flag placement without changing the player's history."""
import json
import shutil
import subprocess
import sys
import time

from confirm_parent_boundaries import (
    ARCHIVE, EXE, PANELS, REPORT as CONFIRM, ROOT, cases, cell_summary, inputs, read, sha,
)
from parent_update_audit import validate_trace

TICKET = "test-targeted-hq-counterexample"
RUN = ROOT / "build" / (TICKET + "-v1")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")


def collect():
    source, old, _ = next(cases())
    if (source["seed"], source["seat"], old["ply"]) != (299025, 2, 447):
        raise ValueError("unexpected frozen test source")
    _, roster, weights = inputs()
    partner = roster["random"]
    args = [RUN / "junqi.exe", "parent-candidate-hq-swap", RUN / "ruleset.json",
            source["seed"], source["seat"], source["anchor"], partner["policy"],
            *roster["candidate"]["weights"], *weights[source["child"]], *partner["weights"],
            old["B"]["picked"], 118]
    result = subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True,
                            timeout=25, creationflags=0x40 | 0x08000000)
    row = json.loads(result.stdout)
    validate_trace(source, row)
    return row


def summarize(row):
    prior = read(CONFIRM)["rows"][0]["point"]
    point = row["point"]
    for key in ("ply", "legal", "choices", "picked", "same_sample"):
        if point[key] != prior[key]:
            raise ValueError("the frozen point changed")
    swap = point["swap"]
    if point["targets"] != [116, 118] or swap not in ("compatible", "illegal_layout", "changes_public_prefix"):
        raise ValueError("unexpected HQ swap status")
    count = 8 if swap == "compatible" else 4
    if (point["confirm"][:4] != prior["confirm"][:4] or
            point["reply_edits"][:4] != prior["reply_edits"][:4]):
        raise ValueError("original world did not repeat the prior results")
    cells = [cell_summary(scores, edits, panel) for scores, edits, panel in
             zip(point["confirm"], point["reply_edits"], PANELS[:count], strict=True)]
    changes = point["first_events"]
    moves = [point["choices"][i][:2] for i in (0, point["picked"])]
    if len(changes) != count // 4 or any([event[1:3] for event in events] != moves for events in changes):
        raise ValueError("unexpected first-move audit")
    return dict(swap=swap, cells=cells, west_exits_after_candidate=[bool(e[1][4] & 8) for e in changes])


def frozen():
    return dict(source=sha(CONFIRM), engine=sha(RUN / "junqi.exe"), rules=sha(RUN / "ruleset.json"))


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("HQ counterexample already exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE, RUN / "junqi.exe")
    shutil.copyfile(ARCHIVE / "ruleset.json", RUN / "ruleset.json")
    started = time.monotonic()
    row = collect()
    data = dict(frozen=frozen(), row=row, summary=summarize(row), seconds=time.monotonic() - started)
    payload = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(payload) > 64 * 1024:
        raise ValueError("HQ test exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(payload)
    print("Recorded original and targeted HQ placement results", data["summary"]["swap"])


def audit():
    data = read(REPORT)
    row = collect()
    if data["frozen"] != frozen() or data["row"] != row or data["summary"] != summarize(row):
        raise ValueError("counterexample replay did not match")
    print("PASS frozen original score, full visible prefix and targeted swap replay")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: target_hq_counterexample.py [audit]")
