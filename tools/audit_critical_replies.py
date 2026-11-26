"""Re-evaluate locked discovery candidates with a different continuation policy."""
import argparse
from collections import Counter
from itertools import groupby
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median
import subprocess
import time

from discover_public_actions import ARMS, COUNT, RUN, START, parse_root
from measure_deadline_opening import LABEL, ROOT, RULES, SCORE, build, file_sha, model_roster, require

PROFILES = ("frozen", "active")
FIELDS = ("outcome", "plies", "overrides", "first_override", "wins_taken", "guards_taken",
          "flag_offered", "flag_played", "contact_offered", "contact_played", "leaf_value")


def decode(fields):
    record = dict(zip(FIELDS, [*map(int, fields[:-1]), float(fields[-1])], strict=True))
    require(record["outcome"] in SCORE and isfinite(record["leaf_value"]) and
            0 <= record["overrides"] < record["plies"], "invalid continuation result")
    first = record["first_override"]
    require((2 <= first <= record["plies"]) if record["overrides"] else first == 0, "invalid first override")
    require(record["wins_taken"] + record["guards_taken"] <= record["overrides"], "invalid override sum")
    for kind in ("flag", "contact"):
        offer, played = record[f"{kind}_offered"], record[f"{kind}_played"]
        require(offer in (-1, 0, 1) and played in (0, offer) and (record["plies"] > 9 or not offer),
                "invalid cutoff opportunity")
    return record


def read_root(lines):
    root = parse_root([line for line in lines if not line.startswith("C ")])
    rows = {}
    frozen = {(p["world"], p["table"]): p["results"] for p in root.get("pairs", [])}
    for line in lines:
        if not line.startswith("C "):
            continue
        _, world, table, arm, *fields = line.split()
        key = tuple(map(int, (world, table, arm)))
        world, table, arm = key
        before, after = decode(fields[:len(FIELDS)]), decode(fields[len(FIELDS):])
        require(key not in rows and not before["overrides"] and
                (before["outcome"], before["plies"]) == frozen[world, table][arm], "frozen pair mismatch")
        require(after["overrides"] or before == after, "continuations differ without an override")
        rows[key] = dict(zip(PROFILES, (before, after), strict=True))
    require(set(rows) == {(w, t, i) for (w, t), results in frozen.items() for i in range(len(results))},
            "missing paired challenge")
    return {"seed": root["seed"], "ply": root["ply"], "arms": root.get("arms", {}),
            "excluded": root.get("excluded"), "replies": [{"world": w, "table": t, "arm": ARMS[i], **reply}
                                                         for (w, t, i), reply in rows.items()]}


def delta(left, right):
    scores = [(SCORE[a], SCORE[b]) for a, b in zip(left, right, strict=True)]
    return [mean((b if b is not None else fill) - (a if a is not None else 1-fill)
                 for a, b in scores) for fill in (0, 1)]


def compare_arms(roots, profile, reference):
    comparisons = {}
    for table in (1, 2):
        pairs = []
        for root in roots:
            choices = {(r["world"], r["arm"]): r[profile]["outcome"] for r in root["replies"] if r["table"] == table}
            pairs.extend((choices[world, reference], value) for (world, arm), value in choices.items() if arm == "outside")
        comparisons[table] = delta(*zip(*pairs, strict=True))
    return comparisons


def parse_wire(lines, reader):
    groups = groupby(lines, key=lambda line: line.startswith("R "))
    return [reader([*header, *next(groups)[1]]) for is_header, header in groups if is_header]


def require_disjoint(before, after):
    for old, new in zip(parse_wire(before, parse_root), parse_wire(after, parse_root), strict=True):
        for field in ("world", "stream"):
            require(not ({p[field] for p in old.get("pairs", [])} & {p[field] for p in new.get("pairs", [])}),
                    f"independent confirmation shares a {field} with development")


def summarize(roots):
    tested = [root for root in roots if root["replies"]]
    rows = [{**r, "root_ply": root["ply"]} for root in tested for r in root["replies"]]
    arm_scores = {profile: {arm: dict(Counter(LABEL[r[profile]["outcome"]] for r in rows if r["arm"] == arm))
                            for arm in ARMS} for profile in PROFILES}
    cutoffs = {}
    for profile in PROFILES:
        records = [r[profile] for r in rows if r[profile]["plies"] > 9]
        favorable = [r for r in records if r["leaf_value"] > 0]
        cutoffs[profile] = {"reached": len(records), "positive_leaf_results": dict(Counter(LABEL[r["outcome"]] for r in favorable))}
        for kind in ("flag", "contact"):
            cases = [r for r in records if r[f"{kind}_offered"] == -1]
            cutoffs[profile][f"enemy_{kind}"] = dict(offered=len(cases), played=sum(r[f"{kind}_played"] == -1 for r in cases),
                                                    positive_leaf=sum(r["leaf_value"] > 0 for r in cases))
    replans = {field: sum(r["active"][field] for r in rows) for field in ("overrides", "wins_taken", "guards_taken")}
    transitions = Counter(f"{LABEL[r['frozen']['outcome']]}→{LABEL[r['active']['outcome']]}" for r in rows)
    positive, reversal = [], []
    for root in tested:
        old, new = [compare_arms([root], profile, "shortlist") for profile in PROFILES]
        if all(lo > 0 for lo, _ in new.values()):
            positive.append([root["seed"], root["ply"], root["arms"]["outside"]])
        if any((old[t][1] < 0 < new[t][0]) or (new[t][1] < 0 < old[t][0]) for t in new):
            reversal.append(dict(seed=root["seed"], ply=root["ply"], frozen=old, active=new))
    return dict(roots=len(tested), pairs=len(rows), arm_scores=arm_scores, transitions=dict(transitions),
                cutoffs=cutoffs, replans=replans, branches_changed=sum(r["active"]["overrides"] > 0 for r in rows),
                first_override_median=median(r["active"]["first_override"] for r in rows if r["active"]["overrides"]),
                wins_by140={profile: sum(r[profile]["outcome"] == 2 and r["root_ply"] + r[profile]["plies"] <= 140
                                        for r in rows) for profile in PROFILES},
                outside_minus_shortlist={p: compare_arms(tested, p, "shortlist") for p in PROFILES},
                outside_ahead_both_tables=positive, roots_with_reversal=reversal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RUN / "run-1.txt")
    parser.add_argument("--holdout", type=Path, help="development results.json, with nominations already locked")
    args = parser.parse_args()
    reference = args.reference.read_text(encoding="utf8").splitlines()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    build(output)
    roster = model_roster()
    models = [roster[name] for name in ("candidate", "historical")]
    exe = output / "deadline_opening.exe"
    payload = " ".join(map(str, (START, COUNT, *(w for model in models for w in model["weights"]))))
    elapsed, outputs = [], []
    mode = "confirm-critical-replies" if args.holdout else "audit-critical-replies"
    for attempt in (1, 2):
        now = time.monotonic()
        run = subprocess.run([exe, RULES, mode], input=payload, encoding="utf8",
                             capture_output=True, timeout=150, check=True, cwd=ROOT)
        outputs.append(run.stdout)
        (output / f"run-{attempt}.txt").write_text(run.stdout, encoding="utf8")
        elapsed.append(round(time.monotonic()-now, 2))
    require(outputs[0] == outputs[1], "challenge is not reproducible")
    lines = outputs[0].splitlines()
    audited = [line for line in lines if not line.startswith("C ")]
    if args.holdout:
        nominations = lambda batch: [line for line in batch if not line.startswith("P ")]
        require(nominations(reference) == nominations(audited), "archived nominations changed")
        require_disjoint(reference, audited)
    else:
        require(audited == reference, "archived nomination, world or frozen continuation changed")
    roots = parse_wire(lines, read_root)
    summary = summarize(roots)
    if args.holdout:
        development = json.loads(args.holdout.read_text(encoding="utf8"))
        selected = development["summary"]["outside_ahead_both_tables"]
        roots_by_key = {(r["seed"], r["ply"], r["arms"].get("outside")): r for r in roots}
        summary["locked_cases"] = [dict(seed=key[0], ply=key[1], move=key[2],
                                        **{f"outside_minus_{arm}": compare_arms([roots_by_key[tuple(key)]], "active", arm)
                                           for arm in ("frozen", "shortlist", "random_outside")}) for key in selected]
    result = dict(engine=file_sha(exe), rules=file_sha(RULES), reference=file_sha(args.reference), models=models,
                  development=file_sha(args.holdout) if args.holdout else None, deterministic=True,
                  run_seconds=elapsed, total_seconds=round(time.monotonic()-started, 2), summary=summary, roots=roots)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
