"""Use a frozen training probe to compare route preparation and follow-through."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
from statistics import mean, median
import subprocess
import sys

WINDOWS = (24, 72, 120)
TABLES = {0: "random_others", 2: "heuristic_enemies"}
RESULTS = {-1: "cutoff", 0: "loss", 1: "draw", 2: "win"}
ROOT = Path(__file__).resolve().parents[1]


def gather(roots, frozen):
    from archive_action_screen import layout_group
    from audit_qq_archive import LIMIT, decode, initial_board
    from compare_qq_rules import tape

    history, gaps = {}, Counter()
    for index, directory in enumerate(roots):
        if not directory.is_dir():
            raise ValueError(f"missing replay source {index}")
        picked = {}
        for path in directory.rglob("*.jgs"):
            try:
                if path.stat().st_size > LIMIT:
                    raise ValueError("size")
                record = decode(path.read_bytes())
                key, layouts, _, _ = record
                family = layout_group(layouts).hex()
            except (OSError, ValueError):
                gaps[f"{index}/format"] += 1
                continue
            if family not in frozen and (family not in picked or key < picked[family][0]):
                picked[family] = record
        for _, (key, layouts, seats, events) in sorted(picked.items()):
            try:
                rows, ccw = tape(layouts, seats, events)
            except ValueError:
                gaps[f"{index}/collision"] += 1
                rows, ccw = [], 0
            if ccw:
                gaps[f"{index}/direction"] += 1
                rows = []
            board = [(x, y, rank) for (x, y), (_, rank) in initial_board(layouts).items()]
            history[f"qq-{index}-{key[:14]}"] = (board, rows)
    return history, dict(gaps)


def windows(sampled, sources):
    result = {}
    for index in range(sources):
        for ply in WINDOWS:
            rows = [(key, data) for key, data in sampled.items()
                    if key.startswith(f"qq-{index}-") and key.endswith(f"/{ply}")]
            aligned = [(key, data) for key, data in rows if data is not None]
            choices = [(key, data) for key, data in aligned if data["routes"]]
            result[f"{index}/{ply}"] = dict(requested=len(rows), aligned=len(aligned),
                                           route_windows=len(choices),
                                           actual_setup=[key for key, data in choices if data["played_route"]],
                                           best_route_below_top3=sum(data["routes_rank"] >= 3 for _, data in choices))
    return result


def compare(arm):
    block_bounds, transitions, score = {}, Counter(), [Counter(), Counter()]
    for layout, rows in arm.items():
        limits = []
        for a, b in rows:
            score[0][RESULTS[a]] += 1
            score[1][RESULTS[b]] += 1
            transitions[f"{RESULTS[b]}->{RESULTS[a]}"] += 1
            limits.append(((max(a, 0) - (2 if b == -1 else b))/2,
                           ((2 if a == -1 else a) - max(b, 0))/2))
        if limits:
            block_bounds[layout] = [mean(side) for side in zip(*limits)]
    return dict(layouts=len(block_bounds), pairs=sum(transitions.values()),
                both_natural=sum(count for key, count in transitions.items() if "cutoff" not in key),
                bounds_by_layout=[mean(side) for side in zip(*block_bounds.values())],
                intervention=dict(score[0]), fallback=dict(score[1]), transitions=dict(transitions),
                rescued=transitions["loss->draw"]+transitions["loss->win"],
                new_losses=transitions["draw->loss"]+transitions["win->loss"])


def outcomes(pairs):
    result = {}
    for value, name in TABLES.items():
        start, follow = defaultdict(list), defaultdict(list)
        for key, rows in pairs.items():
            layout, _, response = key.split("/")
            if int(response) != value:
                continue
            for a, b, available, c, d in rows:
                start[layout].append((a, b))
                if available and min(c, d) >= -1:
                    follow[layout].append((c, d))
        result[name] = dict(setup_vs_frozen_best=compare(start), route_vs_fallback=compare(follow))
    return result


def audited_probe():
    from compare_qq_rules import VC

    build = ROOT / "build/human-timing-accounting"
    build.mkdir(parents=True, exist_ok=True)
    sources = ("core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
               "tests/attack_candidate_screen.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{entry}"' for entry in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'/I"{ROOT.as_posix()}/core" {files} /Fe:timing_accounting.exe >&2')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=build, check=True, timeout=90)
    return build / "timing_accounting.exe"


def ranking_accounting(records):
    total = len(records)
    values = [row[-1] for row in records.values()]
    outside = sum(higher>=3 for higher, *_ in records.values())
    return dict(windows=total, strict_below_top3=outside, possible_top3=total-outside,
                all_tie_orders_top3=sum(higher+non_route<3 for higher, _, non_route, _, _ in records.values()),
                stable_order_top3=sum(stable<3 for _, _, _, stable, _ in records.values()),
                route_probability_mass_median=median(values) if values else None,
                route_probability_mass_mean=mean(values) if values else None)


def allocation_accounting(phases, pairs):
    report = {}
    phase_groups = {f"{key}/{seed}" for key, rows in pairs.items() for seed in range(len(rows))}
    if phases.keys()!=phase_groups:
        raise ValueError("phase rows do not cover all original pairs")
    for table, name in TABLES.items():
        status, targets, attrition, finals, changes = (Counter() for _ in range(5))
        excluded = []
        retired_before_follow = 0
        for key, rows in pairs.items():
            if int(key.rsplit("/", 1)[1]) != table:
                continue
            for seed, (a,b,legal,c,d) in enumerate(rows):
                reason, target, ending, retired, reached, result = phases[f"{key}/{seed}"]
                retired_before_follow += retired!=0
                status[reason]+=1
                targets[target]+=1
                if bool(legal) != (reason in ("both_actions_available", "no_alternative_action")):
                    raise ValueError("audit changed second-stage eligibility")
                if (c!=-2 and d!=-2) != (reason=="both_actions_available"):
                    raise ValueError("audit disagrees with paired second-stage outputs")
                if not legal:
                    attrition[(reason, RESULTS[a], RESULTS[b])]+=1
                    excluded.append(dict(pair=f"{key}/{seed}", reason=reason, reached_ply=reached,
                                         prepared_team=RESULTS[a], baseline_team=RESULTS[b]))
                if ending:
                    finals[(ending, RESULTS[result])]+=1
                    if result!=a:
                        raise ValueError("prepared branch did not preserve eventual team outcome")
                if a==-1 or b==-1:
                    changes["unresolved"]+=1
                else:
                    changes["improved" if a>b else "worse" if a<b else "unchanged"]+=1
        report[name] = dict(phase_status=status,
                            next_coordinate_use=targets,
                            retired_before_follow=retired_before_follow,
                            missing_terminal_pairs={"/".join(k):v for k,v in sorted(attrition.items())},
                            excluded_pairs=excluded,
                            terminal_before_follow={f"{why}/{outcome}":n for (why,outcome),n in sorted(finals.items())},
                            changes=changes)
    return report


def sampling_bounds(pairs, second=False, repetitions=4000):
    rng = random.Random(56253)
    report = {}
    for table, name in TABLES.items():
        blocks = defaultdict(list)
        for key, rows in pairs.items():
            source, _, variant = key.split("/")
            if int(variant)!=table:
                continue
            for a,b,eligible,c,d in rows:
                if second and (not eligible or min(c,d)<-1):
                    continue
                x,y = (c,d) if second else (a,b)
                blocks[source].append(((max(0,x)-(2 if y==-1 else y))/2,
                                       ((2 if x==-1 else x)-max(0,y))/2))
        scores = [tuple(mean(side) for side in zip(*rows)) for rows in blocks.values()]
        if not scores:
            continue
        lower, upper = [], []
        for _ in range(repetitions):
            sample = rng.choices(scores, k=len(scores))
            lower.append(mean(x for x,_ in sample))
            upper.append(mean(y for _,y in sample))
        report[name] = dict(source_blocks=len(scores), bootstrap_draws=repetitions,
                            outer_percentiles=[sorted(lower)[int(.025*repetitions)],
                                               sorted(upper)[int(.975*repetitions)-1]])
    return report


def account(feed, old, sampled, pairs):
    executable = audited_probe()
    probe = subprocess.run([executable, "audit-timing"], input=feed, capture_output=True,
                           text=True, check=True, timeout=100)
    standard, rankings, phases, mismatches = [], {}, {}, []
    for line in probe.stdout.splitlines():
        tag, source, ply, *fields = line.split()
        key = f"{source}/{ply}"
        if tag=="K":
            rankings[key] = (*map(int, fields[:-1]), float(fields[-1]))
        elif tag=="T":
            table, seed, reason, target, *numbers = fields
            phases[f"{key}/{table}/{seed}"] = (reason,target,*map(int,numbers))
        elif tag=="U":
            mismatches.append(key)
        else:
            standard.append(line)
    if standard!=old.splitlines():
        raise ValueError("instrumented replay differs from the frozen probe")
    if rankings.keys()!={key for key,row in sampled.items() if row and row["routes"]}:
        raise ValueError("rank accounting did not cover the original candidate windows")
    if any(rankings[k][0]!=sampled[k]["routes_rank"] for k in rankings):
        raise ValueError("ranking counts differ from the frozen probe")
    return dict(rankings=ranking_accounting(rankings), focal_outcome_mismatches=len(mismatches),
                allocation=allocation_accounting(phases,pairs), legacy_rows=len(standard),
                start_source_uncertainty=sampling_bounds(pairs),
                conditional_source_uncertainty=sampling_bounds(pairs,second=True))


def run(roots, assets, audit):
    # The historical executable and its saved roster belong to the same checkout.
    sys.path.insert(0, str(assets / "tools"))
    from partner_crossplay import model_roster
    from screen_attack_candidates import collect, send

    build = assets / "build"
    prior = json.loads((build / "archive-pressure-balanced/summary.json").read_text(encoding="utf-8-sig"))
    frozen = {item["family"] for key in ("training_sources", "evaluation_sources") for item in prior[key]}
    history, gaps = gather(roots, frozen)
    model = model_roster()["candidate"]
    feed = io.StringIO()
    print(len(history)*len(WINDOWS), file=feed)
    print(*model["weights"], file=feed)
    for key, (board, rows) in history.items():
        for ply in WINDOWS:
            send(feed, key, board, rows, ply)
    engine = build / "attack-opportunity/attack_candidate_screen.exe"
    probe = subprocess.run([engine], input=feed.getvalue(), capture_output=True, text=True, check=True, timeout=100)
    sampled, pairs = collect(probe.stdout)
    result = dict(windows=windows(sampled, len(roots)), gaps=gaps, model=model["artifact_sha256"],
                  engine=hashlib.sha256(engine.read_bytes()).hexdigest(),
                  outcomes=outcomes(pairs), follow_legal=sum(row[2] for cell in pairs.values() for row in cell))
    if audit:
        result["accounting"] = account(feed.getvalue(),probe.stdout,sampled,pairs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("art", type=Path)
    cli.add_argument("replays", type=Path)
    cli.add_argument("--assets", type=Path, required=True)
    cli.add_argument("--audit", action="store_true", help="explain historical second-stage exclusions and score ties")
    args = cli.parse_args()
    run((args.art, args.replays), args.assets.resolve(), args.audit)
