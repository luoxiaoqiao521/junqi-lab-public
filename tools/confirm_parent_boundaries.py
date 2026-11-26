"""Repeat a fixed shortlist with fresh continuation seeds and a tested reply."""
from itertools import product
import json
import shutil
import subprocess
import sys
import time

from parent_update_audit import (
    ARCHIVE, EXE, REPORT as SOURCE, ROOT, SCORES, inputs, method_evidence, read, sha,
    validate_trace,
)

TICKET = "confirm-frozen-candidate-boundaries"
RUN = ROOT / "build" / (TICKET + "-v1")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")
PAIRS = 24
PANELS = list(product(("original", "compatible"), ("parent", "child"), ("normal", "pressure")))


def cases():
    data = read(SOURCE)
    nominated = data["summary"]["local_candidates"]
    selected = [(r["seed"], r["seat"], "nominee") for r in nominated if r["method"] == "B"]
    if len(selected) != 2:
        raise ValueError("expected two frozen B nominees")
    selected.append((299005, 0, "reversal_control"))
    archived = {(s["seed"], s["seat"]): (s, row["point"])
                for s, row in zip(data["declared"], data["rows"], strict=True)}
    for seed, seat, role in selected:
        source, old = archived[seed, seat]
        method = old["B"]
        choice = method["picked"]
        pair = [old["choices"][i] for i in (0, choice)]
        if role == "nominee":
            if (method_evidence(method, len(old["choices"]) - 1, old["worlds"]) != "positive_all_cells"
                    or not any(r["method"] == "B" and r["seed"] == seed and r["seat"] == seat
                               and r["moves"] == pair for r in nominated)):
                raise ValueError("archived nomination does not match the fixed pair")
        elif ([r[:2] for r in pair] != [[2, 127], [17, 12]] or
              method_evidence(method, len(old["choices"]) - 1, old["worlds"]) != "reversed"):
            raise ValueError("archived control does not match the fixed reversal")
        yield source, old, role


def replay(source, old, roster, weights):
    partner = roster["random" if source["context"] == "normal" else "historical"]
    args = [RUN / "junqi.exe", "parent-candidate-confirm", RUN / "ruleset.json",
            source["seed"], source["seat"], source["anchor"], partner["policy"],
            *roster["candidate"]["weights"], *weights[source["child"]], *partner["weights"],
            old["B"]["picked"]]
    result = subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True,
                            timeout=25, creationflags=0x40 | 0x08000000)
    return json.loads(result.stdout)


def cell_summary(scores, edits, panel):
    world, follow, reply = panel
    if (len(scores) != PAIRS or any(a not in SCORES or b not in SCORES for a, b in scores)
            or len(edits) != 2 or any(type(n) is not int or not 0 <= n <= PAIRS for n in edits)
            or (reply == "normal" and any(edits))):
        raise ValueError("incomplete score or reply coverage")
    low = sum(max(b, 0) - (2 if a == -1 else a) for a, b in scores)
    high = sum((2 if b == -1 else b) - max(a, 0) for a, b in scores)
    return dict(world=world, follow=follow, reply=reply, low=low / (2 * PAIRS),
                high=high / (2 * PAIRS), truncated_pairs=sum(-1 in pair for pair in scores),
                reply_edits=edits)


def summarize(source, old, role, row):
    validate_trace(source, row)
    point = row["point"]
    for key in ("ply", "legal", "choices", "same_sample", "worlds"):
        if point[key] != old[key]:
            raise ValueError("source decision changed from the preregistered shortlist")
    if point["worlds"] != 2 or point["picked"] != old["B"]["picked"]:
        raise ValueError("different first move or compatible-world coverage")
    cells = [cell_summary(scores, edits, panel) for scores, edits, panel in
             zip(point["confirm"], point["reply_edits"], PANELS, strict=True)]
    low = min(cell["low"] for cell in cells)
    high = max(cell["high"] for cell in cells)
    has_gain = any(cell["low"] > 0 for cell in cells)
    has_loss = any(cell["high"] < 0 for cell in cells)
    if low > 0 and any(any(cell["reply_edits"]) for cell in cells):
        outcome = "positive_in_tested_panels"
    elif has_gain and has_loss:
        outcome = "direction_changes"
    else:
        outcome = "positive_not_reproduced"
    return dict(seed=source["seed"], seat=source["seat"], child=source["child"],
                context=source["context"], stratum=source["kind"], role=role, ply=point["ply"],
                moves=[point["choices"][i] for i in (0, point["picked"])], cells=cells,
                span=[low, high], outcome=outcome)


def frozen():
    return dict(source=sha(SOURCE), engine=sha(RUN / "junqi.exe"), rules=sha(RUN / "ruleset.json"))


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("confirmation already exists; use audit")
    frozen_cases = list(cases())
    _, roster, weights = inputs()
    RUN.mkdir()
    shutil.copyfile(EXE, RUN / "junqi.exe")
    shutil.copyfile(ARCHIVE / "ruleset.json", RUN / "ruleset.json")
    started = time.monotonic()
    rows = [replay(source, old, roster, weights) for source, old, _ in frozen_cases]
    summary = [summarize(*case, row) for case, row in zip(frozen_cases, rows, strict=True)]
    data = dict(frozen=frozen(), pairs_per_panel=PAIRS, rows=rows, cases=summary,
                seconds=time.monotonic() - started)
    payload = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(payload) > 64 * 1024:
        raise ValueError("confirmation artifact exceeded byte cap")
    with REPORT.open("xb") as target:
        target.write(payload)
    print(f"Recorded {len(rows) * len(PANELS) * PAIRS} fresh paired continuations")


def audit():
    data = read(REPORT)
    frozen_cases = list(cases())
    _, roster, weights = inputs()
    if (data["frozen"] != frozen() or data["pairs_per_panel"] != PAIRS or
            data["cases"] != [summarize(*case, row) for case, row in
                              zip(frozen_cases, data["rows"], strict=True)]):
        raise ValueError("confirmation does not match its frozen inputs")
    for (source, old, _), row in zip(frozen_cases, data["rows"], strict=True):
        if replay(source, old, roster, weights) != row:
            raise ValueError("fixed confirmation did not replay")
    print("PASS frozen candidates, fresh seeds, response coverage and deterministic replay")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: confirm_parent_boundaries.py [audit]")
