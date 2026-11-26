"""Audit old ties and compare four continuation arms on fresh generated roots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from audit_critical_replies import parse_wire
from discover_reply_counterfactuals import ARMS, KINDS, bounds, grouped_interval, read_root
from measure_deadline_opening import LABEL, ROOT, RULES, SCORE, build, file_sha, model_roster, require

START, COUNT = 1202050, 12
OLD = ROOT / "build/tactic-discovery/enemy-replies-verified/results.json"
FIELDS = ("outcome", "plies", "ending", "first_override", "baseline_move", "actual_move")
ENDINGS = {1: "team_eliminated", 2: "rule_draw", 3: "simultaneous_exhaustion", 4: "no_legal_actions", 5: "ply_cap"}


def audit_old(path):
    data = json.loads(path.read_text(encoding="utf8"))
    audit = {}
    for cohort in ("development", "confirmation"):
        layouts = Counter()
        outcomes = Counter()
        for root in data[cohort]["roots"]:
            for reply in root["receipts"]:
                for choice in reply["choices"]:
                    trials = choice["repeats"]
                    if trials:
                        layouts[root["seed"]] += 1
                    outcomes.update((LABEL[r["frozen"][0]], LABEL[r["alternate"][0]]) for r in trials)
        audit[cohort] = dict(sources=dict(layouts), cases=layouts.total(), pairs=outcomes.total(),
                             outcomes={f"{a}/{b}": count for (a,b), count in outcomes.items()})
    return audit


def read_arm(fields, at):
    end = dict(zip(FIELDS, map(int, fields), strict=True))
    outcome, ending, step = end["outcome"], end["ending"], end["first_override"]
    compatible = {2: (1,), 1: (2,), 0: (1,), -1: (3, 4, 5)}
    require(outcome in SCORE and ending in compatible[outcome] and end["plies"] > at, "missing or inconsistent end reason")
    require((not step and end["baseline_move"] == end["actual_move"] == 0) or
            (at+1 < step <= end["plies"] and end["baseline_move"] != end["actual_move"]), "invalid replan event")
    return {**end, "reason": ENDINGS[ending]}


def read_calibration(lines):
    root = read_root([line for line in lines if not line.startswith("T ")])
    originals = {(r["world"], r["table"], r["arm"]): r for r in root["receipts"]}
    rows = {}
    for line in lines:
        if line.startswith("J "):
            _, world, table, arm, *_ = line.split()
            window = (int(world), int(table), ARMS[int(arm)])
        if not line.startswith("T "):
            continue
        _, kind, source, flag, seed, *fields = line.split()
        require(kind in KINDS and kind != "A" and source in ("H", "O") and flag in ("0", "1"), "invalid nomination")
        arms = {name: read_arm(fields[6*i:6*i+6], root["ply"]+9) for i, name in enumerate("ABCD")}
        require(len(fields)==24 and arms["A"]["first_override"] == arms["B"]["first_override"] == 0, "unpaired four-arm row")
        row = rows.setdefault((*window, kind), dict(world=window[0], table=window[1], arm=window[2],
            kind=KINDS[kind], selection=source, flag=bool(int(flag)), repeats=[]))
        require(row["selection"] == source and row["flag"] == bool(int(flag)), "classification changed within the same cutoff")
        row["repeats"].append(dict(seed=int(seed), **arms))
    for (world, table, arm, kind), item in rows.items():
        origin = originals[world, table, arm]
        before = origin["choices"][0]
        after = next(choice for choice in origin["choices"] if choice["kind"] == KINDS[kind])
        item["original"], item["challenger"] = before["move"], after["move"]
        item["class"] = "known_flag" if item["flag"] else "known_contact" if origin["known_hits"] else "ordinary"
        require(len(item["repeats"]) == len({v["seed"] for v in item["repeats"]}) == 8, "incomplete independent tail set")
    return {**root, "calibration": list(rows.values())}


def direction(a, b):
    if min(a["outcome"], b["outcome"]) < 0:
        return "unresolved"
    if a["outcome"] != b["outcome"]:
        return "damage" if a["outcome"] > b["outcome"] else "gain"
    return "both_" + LABEL[a["outcome"]]


def stable_signal(directions):
    return directions.count("damage") >= 4 and "gain" not in directions and "unresolved" not in directions


def summarize_stratum(rows):
    effects = {k: [] for k in ("challenge_frozen", "challenge_replan", "replan_given_original", "replan_given_challenge", "interaction")}
    directions = {k: Counter() for k in ("AB", "CD")}
    endings = {k: Counter() for k in "ABCD"}
    winners = {k: 0 for k in directions}
    gate = Counter()
    for source, cutoff, item in rows:
        local = {k: [] for k in directions}
        for t in item["repeats"]:
            for key in directions:
                verdict = direction(t[key[0]], t[key[1]])
                local[key].append(verdict)
                directions[key][verdict] += 1
            for key in endings:
                endings[key][t[key]["reason"]] += 1
            a, b, c, d = (t[key]["outcome"] for key in "ABCD")
            ab, cd, ac, bd = bounds(b,a), bounds(d,c), bounds(c,a), bounds(d,b)
            for values, effect in zip(effects.values(), (ab, cd, ac, bd, [cd[0]-ab[1], cd[1]-ab[0]]), strict=True):
                values.append((source, effect))
        kept = any("damage" in values[:4] or "unresolved" in values[:4] for values in local.values())
        stable = {k: stable_signal(values) for k, values in local.items()}
        for k in stable:
            winners[k] += stable[k]
        gate["continue" if kept else "stop"] += 1
        gate["would_miss_stable"] += not kept and any(stable.values())
        for i, t in enumerate(item["repeats"]):
            steps = sum(t[key]["plies"]-cutoff for key in endings)
            gate["all_tail_plies"] += steps
            if not kept and i >= 4:
                gate["deferred_tail_plies"] += steps
    return dict(windows=len(rows), directions=directions, endings=endings, local_signal=winners,
                gate=dict(gate), effects={k: grouped_interval(v) for k,v in effects.items()})


def summarize(roots):
    rows = [(r["seed"], r["ply"]+9, c) for r in roots for c in r["calibration"]]
    def subset(field, label):
        return summarize_stratum([row for row in rows if row[-1][field] == label])
    examples = [dict(seed=seed, cutoff=cutoff, profile=key, **item) for seed, cutoff, item in rows for key in ("AB", "CD")
                if stable_signal([direction(t[key[0]],t[key[1]]) for t in item["repeats"]])]
    return dict(roots=len(roots), chances=sum(len(r["receipts"]) for r in roots),
                all=summarize_stratum(rows), local_examples=examples, by_selection={k: subset("selection", k) for k in ("H", "O")},
                by_tactic={k: subset("class", k) for k in ("known_flag", "known_contact", "ordinary")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=OLD)
    args = parser.parse_args()
    prior = audit_old(args.prior)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    build(output)
    exe = output / "deadline_opening.exe"
    roster = model_roster()
    models = [roster[name] for name in ("candidate", "historical")]
    stdin = " ".join(map(str, (START, COUNT, *(w for model in models for w in model["weights"]))))
    results, runtime = [], []
    for index in (1,2):
        now = time.monotonic()
        raw = subprocess.run([exe, RULES, "calibrate-reply-evaluator"], cwd=ROOT, input=stdin, encoding="utf8",
                             capture_output=True, check=True, timeout=210)
        runtime.append(round(time.monotonic()-now, 2))
        require("PASS evaluator distinguishes referee win, draw, and cutoff in both profiles" in raw.stderr,
                "known result controls missing")
        (output / f"run-{index}.txt").write_text(raw.stdout, encoding="utf8")
        results.append(raw.stdout)
    require(results[0] == results[1], "nondeterministic four-arm results")
    roots = parse_wire(results[0].splitlines(), read_calibration)
    require([(r["seed"],r["ply"]) for r in roots] == [(s, p) for s in range(START, START+COUNT) for p in (40,88)],
            "fresh layout schedule changed")
    report = dict(engine=file_sha(exe), rules=file_sha(RULES), prior=file_sha(args.prior), models=models,
                  prior_counts=prior, summary=summarize(roots), roots=roots, repeat_seconds=runtime,
                  elapsed_seconds=round(time.monotonic()-started, 2))
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k:report[k] for k in ("prior_counts", "summary", "elapsed_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
