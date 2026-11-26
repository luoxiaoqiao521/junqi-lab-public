"""Re-check positional aliases with fresh rollouts and new source layouts."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

from credit_window import classify, collect, expected_rows
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/credit-direction-confirm-v1"
PREVIOUS = ROOT / "docs/REPORTS/feedback-credit-window.json"
REPORT = ROOT / "docs/REPORTS/credit-direction-confirm.json"
STARTS = (286400, 286432)
CASES = ((286213, 1, 0, 739), (286223, 3, 0, 612))


def original_cases():
    raw = json.loads(PREVIOUS.read_text(encoding="utf8"))["cohorts"]["development"]
    rows = {(r["seed"], r["seat"], r["variant"]): r for r in raw}
    return [rows[(seed, seat, variant)] for seed, seat, variant, _ in CASES]


def window_at(row, ply):
    return next(window for window in row["windows"] if window is not None and window["ply"] == ply)


def alias_status(window):
    alt = window["alternatives"][0]
    if not alt:
        return "missing-alias", None
    if window["played"][2:7] != alt["action"][2:7] or window["played"][:2] == alt["action"][:2]:
        raise ValueError("invalid same-five action pair")
    delta = alt["action"][7] - window["played"][7]
    if not delta:
        return "same-position", None
    key, sign = classify(alt["cells"], window["worlds"])
    return ("more-progress" if sign*delta > 0 else "less-progress") if sign else key, alt["cells"]


def without_samples(row):
    return {**row, "windows": [None if not window else {
        **window, "alternatives": [None if not alt else alt["action"] for alt in window["alternatives"]]
    } for window in row["windows"]]}


def paired_half_signs(cell):
    half = len(cell)//2
    signs = []
    for records in (cell[:half], cell[half:]):
        if any(-1 in pair[:2] or 1 in pair[2:] for pair in records):
            return 0
        difference = sum(record[1]-record[0] for record in records)
        signs.append((difference > 0)-(difference < 0))
    return signs[0] if signs[0] == signs[1] else 0


def sensitivity(cells):
    if not cells or len(cells) != 4:
        return False, False
    a, b, c, d = map(paired_half_signs, cells)
    return a*b < 0 or c*d < 0, a*c < 0 or b*d < 0


def fresh_summary(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != [
            key for start in STARTS for key in expected_rows(start)]:
        raise ValueError("new source grid differs from frozen layout ranges")
    result = {}
    for variant in (0, 2):
        counts = Counter()
        hits = {"more-progress": set(), "less-progress": set()}
        for row in rows:
            if row["variant"] != variant:
                continue
            for window in row["windows"]:
                if not window:
                    counts["missing-window"] += 1
                    continue
                label, cells = alias_status(window)
                counts[label] += 1
                if label in hits:
                    hits[label].add(row["seed"])
                response, world = sensitivity(cells)
                counts["response-reversal-in-both-halves"] += response
                counts["world-reversal-in-both-halves"] += world
        result[str(variant)] = dict(counts=dict(counts), layouts={k: sorted(v) for k, v in hits.items()})
    return result


def summarize(rechecks, fresh):
    if len(rechecks) != len(CASES):
        raise ValueError("missing known cases")
    seen = []
    for original, recheck, (seed, seat, variant, ply) in zip(original_cases(), rechecks, CASES, strict=True):
        if without_samples(recheck) != without_samples(original):
            raise ValueError("rechecking changed reachable windows or public action pairs")
        a, _ = alias_status(window_at(original, ply))
        b, cells = alias_status(window_at(recheck, ply))
        if cells == window_at(original, ply)["alternatives"][0]["cells"]:
            raise ValueError("new continuation streams did not change recorded trials")
        seen.append(dict(seed=seed, seat=seat, variant=variant, ply=ply, prior=a, new=b,
                         repeated_response_reversal=sensitivity(cells)[0],
                         repeated_world_reversal=sensitivity(cells)[1]))
    new = fresh_summary(fresh)
    counts = {direction: len({seed for variant in new.values() for seed in variant["layouts"][direction]})
              for direction in ("more-progress", "less-progress")}
    eligible = (all(item["new"] == "less-progress" for item in seen) and
                counts["less-progress"] >= 4 and counts["more-progress"] == 0)
    return dict(known=seen, fresh=new, fresh_direction_layouts=counts, limited_probe_eligible=eligible)


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), previous=sha(PREVIOUS), parent=parent):
        raise ValueError("frozen continuation inputs changed")
    if data["summary"] != summarize(data["rechecks"], data["fresh"]):
        raise ValueError("direction counts differ from paired continuations")
    for old, (seed, seat, variant, _) in zip(original_cases(), CASES, strict=True):
        if collect(engine, rules, parent, seed, seat, variant) != old:
            raise ValueError("default rollout sequence no longer matches the previous implementation")
    print("PASS original RNG compatibility, preserved actions, different rollout streams and matched summaries")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("direction confirmation archive exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rechecks = [collect(engine, rules, parent, seed, seat, variant, reseed=True)
                for seed, seat, variant, _ in CASES]
    print("Rechecked two frozen action pairs", flush=True)
    fresh = [collect(engine, rules, parent, *key, reseed=True)
             for start in STARTS for key in expected_rows(start)]
    data = dict(rechecks=rechecks, fresh=fresh, summary=summarize(rechecks, fresh),
                seconds=time.monotonic()-started,
                frozen=dict(engine=sha(engine), rules=sha(rules), previous=sha(PREVIOUS), parent=parent))
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(content) > 2*1024**2:
        raise ValueError("direction confirmation byte budget exceeded")
    with REPORT.open("xb") as output:
        output.write(content)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
