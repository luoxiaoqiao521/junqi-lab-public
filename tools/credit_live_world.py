"""Measure whether proposing live hidden swaps recovers missing worlds."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

from credit_window import COHORTS, cohort_summary, collect, classify
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/credit-live-world-coverage-v1"
PREVIOUS = ROOT / "docs/REPORTS/feedback-credit-window.json"
REPORT = ROOT / "docs/REPORTS/credit-live-world-coverage.json"


def source_rows():
    return json.loads(PREVIOUS.read_text(encoding="utf8"))["cohorts"]


def compare_window(old, new, counts, recovered):
    if old is None or new is None:
        if old != new:
            raise ValueError("world search changed window eligibility")
        return
    for key in ("ply", "remaining", "played"):
        if old[key] != new[key]:
            raise ValueError("world search changed the fixed public decision")
    if len(old["alternatives"]) != 2 or len(new["alternatives"]) != 2:
        raise ValueError("incomplete fixed move contrast")
    counts["original_world_coverage"] += old["worlds"] == 2
    counts["new_world_coverage"] += new["worlds"] == 2
    counts["eligible_windows"] += 1
    if old["worlds"] == 2:
        if old != new:
            raise ValueError("world search changed a previously valid world")
        return
    if new["worlds"] == 1:
        if old != new:
            raise ValueError("world search changed an uncovered original")
        return
    counts["recovered_windows"] += 1
    for kind, a, b in zip(("same-five", "different-five"),
                          old["alternatives"], new["alternatives"], strict=True):
        if a is None and b is None:
            continue
        if a is None or b is None or a["action"] != b["action"] or a["cells"] != b["cells"][:2]:
            raise ValueError("world search changed a chosen action or original continuation")
        status, direction = classify(b["cells"], new["worlds"])
        recovered[kind][status] += 1
        recovered[kind]["alternative-better"] += direction > 0
        recovered[kind]["played-better"] += direction < 0


def summarize(original, revised):
    if set(original) != set(COHORTS) or set(revised) != set(COHORTS):
        raise ValueError("unexpected source layout groups")
    groups = {}
    for key, start in COHORTS.items():
        first = cohort_summary(original[key], start)
        second = cohort_summary(revised[key], start)
        counts = Counter()
        recovered = {kind: Counter() for kind in ("same-five", "different-five")}
        for a, b in zip(original[key], revised[key], strict=True):
            if any(a[field] != b[field] for field in ("seed", "seat", "variant", "score", "plies")):
                raise ValueError("source game drift during world coverage comparison")
            for old, new in zip(a["windows"], b["windows"], strict=True):
                compare_window(old, new, counts, recovered)
        groups[key] = dict(coverage=dict(counts),
                            new_pairs={kind: dict(outcomes) for kind, outcomes in recovered.items()},
                            prior_examples=first["examples"], new_examples=second["examples"])
    return groups


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), previous=sha(PREVIOUS), parent=parent):
        raise ValueError("frozen world coverage artifacts changed")
    if data["summary"] != summarize(source_rows(), data["rows"]):
        raise ValueError("coverage differs from archived branches")
    for rows in data["rows"].values():
        for row in (rows[0], rows[-1]):
            if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"], prefer_live=True):
                raise ValueError("live-world fallback not reproducible")
    print("PASS exact original branches and stable supplemental live-world replays")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("world coverage archive exists; use audit")
    original = source_rows()
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    revised = {}
    for key, rows in original.items():
        revised[key] = [collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                                prefer_live=True) for row in rows]
        print(f"Checked {key} with identical choices and live-world fallback", flush=True)
    data = dict(rows=revised, summary=summarize(original, revised), seconds=time.monotonic()-started,
                frozen=dict(engine=sha(engine), rules=sha(rules), previous=sha(PREVIOUS), parent=parent))
    contents = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(contents) > 2*1024**2:
        raise ValueError("live-world output byte budget exceeded")
    with REPORT.open("xb") as output:
        output.write(contents)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
