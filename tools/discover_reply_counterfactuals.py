"""Discover legal alternatives to a frozen opponent reply at the nine-ply cutoff."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import time

from audit_critical_replies import parse_wire
from discover_public_actions import ARMS, COUNT, RUN, START, parse_root
from measure_deadline_opening import LABEL, ROOT, RULES, SCORE, build, file_sha, model_roster, require

SAMPLES = {"development": (START, COUNT), "confirmation": (1201980, 24)}
KINDS = {"A": "frozen_reply", "C": "known_capture", "P": "pressure_reply", "Q": "low_ranked_quiet"}
STATUSES = {"ended", "friendly", "already-chose", "unsampled-control", "test"}
DIAGNOSTICS = ("J ", "K ", "V ")


def read_root(lines):
    clean = [line for line in lines if line[:2] not in DIAGNOSTICS]
    root = parse_root(clean)
    receipts = {}
    for line in lines:
        if line.startswith("J "):
            _, world, table, arm, status, *rest = line.split()
            key = tuple(map(int, (world, table, arm)))
            require(status in STATUSES and key not in receipts, "invalid cutoff status")
            names = ("actor", "quiet_clock", "legal", "known_hits", "observation", "fallback")
            detail = dict(zip(names, map(int, rest), strict=True)) if status == "test" else {}
            require(status == "test" or not rest, "unexpected data in omitted cutoff")
            receipt = receipts[key] = dict(world=key[0], table=key[1], arm=ARMS[key[2]], status=status, choices=[], **detail)
            if detail:
                require(0 <= receipt["quiet_clock"] and 0 <= receipt["known_hits"] < receipt["legal"]
                        and (receipt["actor"]-root["seat"]) % 2 == 1, "cutoff is not an enemy choice")
        elif line.startswith("K "):
            _, kind, move, rank, old, old_ply, guard, guard_ply = line.split()
            require(receipt["status"] == "test" and kind in KINDS, "choice without cutoff")
            move, rank, old, old_ply, guard, guard_ply = map(int, (move, rank, old, old_ply, guard, guard_ply))
            require(old in SCORE and guard in SCORE and min(old_ply, guard_ply) >= root["ply"]+10, "bad outcome")
            receipt["choices"].append(dict(kind=KINDS[kind], move=move, heuristic=rank,
                                           frozen=[old, old_ply], replanning=[guard, guard_ply], repeats=[]))
        elif line.startswith("V "):
            _, kind, seed, old, old_ply, alternate, alternate_ply = line.split()
            move = receipt["choices"][-1]
            require(move["kind"] == KINDS[kind] and kind != "A", "repeat has no locked alternate")
            seed, old, old_ply, alternate, alternate_ply = map(int, (seed, old, old_ply, alternate, alternate_ply))
            require(old in SCORE and alternate in SCORE and min(old_ply, alternate_ply) >= root["ply"]+10, "bad repeat")
            move["repeats"].append(dict(seed=seed, frozen=[old, old_ply], alternate=[alternate, alternate_ply]))
    original = {(p["world"], p["table"], arm): (outcome, root["ply"]+plies)
                for p in root.get("pairs", []) for arm, (outcome, plies) in enumerate(p["results"])}
    require(set(receipts) == set(original), "cutoffs do not cover every parent continuation")
    for key, receipt in receipts.items():
        choices = receipt["choices"]
        if receipt["status"] == "test":
            require(choices[0]["kind"] == KINDS["A"] and choices[0]["move"] == receipt["fallback"] and
                    tuple(choices[0]["frozen"]) == original[key], "unmodified branch differs from the parent")
            require(len(choices) == len({m["move"] for m in choices}) == len({m["kind"] for m in choices}),
                    "duplicate reply")
            require(any(m["kind"] == KINDS["C"] for m in choices) == bool(receipt["known_hits"]), "missed required capture")
            for m in choices:
                size = 8 if 0 <= m["frozen"][0] < choices[0]["frozen"][0] else 0
                require(len(m["repeats"]) == len({v["seed"] for v in m["repeats"]}) == size, "wrong repeat budget")
    return {"seed": root["seed"], "ply": root["ply"], "seat": root["seat"], "excluded": root.get("excluded"),
            "receipts": list(receipts.values())}


def bounds(baseline, alternate):
    a, b = SCORE[baseline], SCORE[alternate]
    return [(a if a is not None else fill) - (b if b is not None else 1-fill) for fill in (0, 1)]


def grouped_interval(values):
    """Each generated opening gets equal weight, including in the resampling."""
    by_source = defaultdict(list)
    for seed, value in values:
        by_source[seed].append(value)
    groups = [[mean(column) for column in zip(*items, strict=True)] for items in by_source.values()]
    if not groups:
        return {"sources": 0, "bounds": None, "interval": None}
    rng = Random(741782)
    draws = [rng.choices(groups, k=len(groups)) for _ in range(2048)]
    low = sorted(mean(item[0] for item in row) for row in draws)
    high = sorted(mean(item[1] for item in row) for row in draws)
    return dict(sources=len(groups), bounds=[mean(x) for x in zip(*groups, strict=True)],
                interval=[low[51], high[1996]] if len(groups) > 1 else None)


def summarize_family(rows, name):
    pairs = [(root, r, r["choices"][0], m) for root, r in rows for m in r["choices"][1:] if m["kind"] == name]
    losses = {profile: Counter() for profile in ("frozen", "replanning")}
    losses_by_table = defaultdict(list)
    damage, interaction = [], []
    for root, r, a, b in pairs:
        differences = []
        for profile in losses:
            old, new = a[profile][0], b[profile][0]
            losses[profile][f"{LABEL[old]}→{LABEL[new]}"] += 1
            differences.append(bounds(old, new))
        damage.append((root["seed"], differences[0]))
        losses_by_table[r["table"]].append(damage[-1])
        interaction.append((root["seed"], [differences[0][0]-differences[1][1], differences[0][1]-differences[1][0]]))
    return dict(windows=len(pairs), distinct_observations=len({r["observation"] for _, r, _, _ in pairs}),
                transitions=losses, frozen_damage=grouped_interval(damage),
                by_table={table: grouped_interval(cells) for table, cells in losses_by_table.items()},
                extra_damage_vs_replanning=grouped_interval(interaction))


def summarize(roots):
    all_rows = [(root, r) for root in roots for r in root["receipts"]]
    rows = [(root, r) for root, r in all_rows if r["status"] == "test"]
    examples = {}
    repeats = Counter()
    robust = []
    for root, r in rows:
        for move in r["choices"][1:]:
            a = r["choices"][0]["frozen"][0]
            b = move["frozen"][0]
            if (a, b) not in ((2, 0), (2, 1), (1, 0), (0, 2), (1, 2)):
                continue
            key = (move["kind"], a, b)
            item = dict(seed=root["seed"], ply=root["ply"]+9, actor=r["actor"], table=r["table"],
                        world=r["world"], arm=r["arm"], fallback=r["choices"][0], reply=move)
            examples.setdefault(key, {}).setdefault(root["seed"], item)
            if move["repeats"]:
                trials = [(v["frozen"][0], v["alternate"][0]) for v in move["repeats"]]
                repeats[move["kind"]] += 1
                if sum(x > y for x, y in trials) >= 4 and all(x >= y >= 0 for x, y in trials):
                    robust.append(item)
    return dict(roots=len(roots), prefixes=len(all_rows), statuses=dict(Counter(r["status"] for _, r in all_rows)),
                groups={name: summarize_family(rows, name) for name in list(KINDS.values())[1:]},
                repeat_cases=dict(repeats), locally_repeated=robust,
                examples={f"{kind}/{LABEL[a]}→{LABEL[b]}": list(sources.values())[:3]
                          for (kind, a, b), sources in examples.items()})


def run(exe, first, count, models, output, sample):
    payload = " ".join(map(str, (first, count, *(w for model in models for w in model["weights"]))))
    started = time.monotonic()
    result = subprocess.run([exe, RULES, "discover-reply-counterfactuals"], cwd=ROOT, input=payload,
                            encoding="utf8", capture_output=True, check=True, timeout=210)
    (output / f"{sample}.txt").write_text(result.stdout, encoding="utf8")
    return result.stdout.splitlines(), round(time.monotonic()-started, 2)


def collect(exe, output, models, sample):
    first, count = SAMPLES[sample]
    lines, elapsed = run(exe, first, count, models, output, sample)
    repeated, repeated_seconds = run(exe, first, count, models, output, f"{sample}-repeat")
    require(lines == repeated, "repeated cutoffs, choices or terminal outcomes changed")
    roots = parse_wire(lines, read_root)
    require([(r["seed"], r["ply"]) for r in roots] == [(seed, ply) for seed in range(first, first+count) for ply in (40,88)],
            "unexpected layout or fixed sampling time")
    return lines, dict(roots=roots, elapsed_seconds=[elapsed, repeated_seconds], summary=summarize(roots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RUN / "run-1.txt")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    build(output)
    exe = output / "deadline_opening.exe"
    roster = model_roster()
    models = [roster[name] for name in ("candidate", "historical")]
    lines, development = collect(exe, output, models, "development")
    require([line for line in lines if line[:2] not in DIAGNOSTICS] ==
            args.reference.read_text(encoding="utf8").splitlines(), "development differs from prior frozen continuations")
    _, confirmation = collect(exe, output, models, "confirmation")
    require(not {r["seed"] for r in development["roots"]} & {r["seed"] for r in confirmation["roots"]},
            "the same layout seed occurs in both batches")
    receipt = dict(engine=file_sha(exe), rules=file_sha(RULES), reference=file_sha(args.reference), models=models,
                   elapsed_seconds=round(time.monotonic()-started, 2), development=development, confirmation=confirmation)
    (output / "results.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({name: receipt[name]["summary"] for name in SAMPLES}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
