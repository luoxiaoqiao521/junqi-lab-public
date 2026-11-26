"""Use verified real replay prefixes as starting states for team-return training."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from random import Random
import shutil
from statistics import mean
import subprocess
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from archive_action_screen import select
from audit_qq_archive import INPUT, ROOT, initial_board, public_records, local_records
from compare_archive_cycle_exit_values import interval
from evidence_model import load, save, sha
from learn_short import EXE
from partner_crossplay import ARCHIVE, FILES

RULES = ROOT / "config/ruleset-selfplay.json"
DEFAULT_OUT = ROOT / "build/archive-pressure-balanced"
PARENT = ARCHIVE / FILES["candidate"]
PREFIX = 96
SPLIT_SEED, TRAIN_SEED, EVAL_SEED = 842991, 843200, 843801
GROUPS = ("ordinary", "pressure")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_records():
    locator = read(INPUT)
    corpus = Path(locator["corpus"])
    local = select(local_records(Path(locator["local"])))
    public = select(public_records(corpus))
    families = sorted(public.keys() - local.keys())
    rng = Random(SPLIT_SEED)
    rng.shuffle(families)
    families = families[:3500]
    cutoff = len(families) * 2 // 3
    filtered, cases = Counter(), []
    for ordinal, family in enumerate(families):
        key, layouts, _, events = public[family]
        prefix = events[:PREFIX]
        fold = "train" if ordinal < cutoff else "holdout"
        if len(prefix) != PREFIX or any(event[0] != 95 for event in prefix):
            filtered[fold] += 1
            continue
        cases.append((dict(family=family.hex(), key=key, fold=fold, rotation=family[0] % 4), layouts, prefix))
    return cases, dict(filtered), sha(corpus)


def normalized_xy(x, y, rotation):
    # Mirror the archive's clockwise coordinates and rotate each family once.
    x = 16 - x
    for _ in range(rotation):
        x, y = 16 - y, x
    return x, y


def feed(cases, weights=None, seed=None):
    rows = [str(len(cases))]
    if weights is not None:
        rows.append(" ".join(map(str, (seed, *weights))))
    for meta, layouts, prefix in cases:
        rotation = meta["rotation"]
        rows.append(str(len(prefix)))
        for (x, y), (_, rank) in initial_board(layouts).items():
            a, b = normalized_xy(x, y, rotation)
            rows.append(f"{a} {b} {rank}")
        for event in prefix:
            a, b = normalized_xy(event[2], 16-event[3], rotation)
            c, d = normalized_xy(event[4], 16-event[5], rotation)
            rows.append(f"{a} {b} {c} {d} {event[1]&3}")
    return "\n".join(rows) + "\n"


def choose(cases, findings, fold, limit):
    by_seat = defaultdict(list)
    for case, finding in zip(cases, findings, strict=True):
        if case[0]["fold"] == fold and finding["group"] in GROUPS:
            meta, layouts, prefix = case
            by_seat[finding["seat"], finding["group"]].append((dict(meta, **finding), layouts, prefix))
    choices = {group: [] for group in GROUPS}
    for seat in range(4):
        n = min(limit, *(len(by_seat[seat, group]) for group in GROUPS))
        for group in GROUPS:
            choices[group].extend(by_seat[seat, group][:n])
    return choices


def verify_rejections(engine_call, rules, cases, findings):
    valid = next((case for case, row in zip(cases, findings, strict=True)
                  if row["group"] in (*GROUPS, "unmatched")), None)
    if valid is None:
        raise ValueError("no strictly aligned replay available for the rejection fixture")
    meta, layouts, prefix = valid
    wrong = bytes((prefix[0][0], prefix[0][1] ^ 1)) + prefix[0][2:]
    bad = [(meta, layouts, [wrong, *prefix[1:]]),
           (meta, layouts, [prefix[0], prefix[0], *prefix[2:]])]
    reasons = [row["group"] for row in engine_call("archive-curriculum", rules, "scan", input_text=feed(bad))]
    if reasons != ["collision_mismatch", "illegal_move_or_turn"]:
        raise ValueError(f"malformed replay prefix was not rejected: {reasons}")
    return reasons


def validate_training(training):
    counts = training["settlement"]["terminal_reasons"]
    assert training["natural"] + training["truncated"] == 512 == sum(counts.values())
    assert training["settlement"]["scored_with_decisions"] == training["natural"] - training["no_decisions"]
    assert counts["team_eliminated"] + counts["no_progress_draw"] == training["natural"]
    assert len(training["weights"]) == 5


def compare_rows(reference, candidate, metadata, normal=False):
    def key(r):
        return (r["seed"], r["seat"], r["first"]) if normal else (r["root"], r["seat"], r["table"], r["rep"])

    blocks, report = defaultdict(list), Counter()
    for a, b in zip(reference, candidate, strict=True):
        if key(a) != key(b):
            raise ValueError("unpaired source, seed, table or seat")
        for row in (a, b):
            if row["score"] not in (None, 0, .5, 1):
                raise ValueError("unknown team score")
        old, new = a["score"], b["score"]
        group = a["seed"] if normal else metadata[a["root"]]["family"]
        blocks[group].append(((new or 0) - (old if old is not None else 1),
                              (new if new is not None else 1) - (old or 0)))
        report["pairs"] += 1
        for label, row in (("reference", a), ("candidate", b)):
            verdict = {None: "cutoff", 0: "loss", .5: "draw", 1: "win"}[row["score"]]
            report[f"{label}_{verdict}"] += 1
            report[f"{label}_first_hq_pressure"] += row.get("hq_pressure", 0)
            report[f"{label}_first_contact"] += row.get("contact", 0)
        report["rescued"] += old == 0 and new in (.5, 1)
        report["new_losses"] += old in (.5, 1) and new == 0
    lows = [mean(x[0] for x in cells) for cells in blocks.values()]
    highs = [mean(x[1] for x in cells) for cells in blocks.values()]
    return dict(report, blocks=len(blocks), bounds=[mean(lows), mean(highs)],
                conservative_interval=[interval(lows)[0], interval(highs)[1]])


def summaries(results, meta):
    panels = {}
    for label, table in (("random", 0), ("contact_enemy", 4)):
        for group in GROUPS:
            keys = {i for i, r in enumerate(meta) if r["group"] == group}
            subset = {name: [row for row in rows if row["table"] == table and row["root"] in keys]
                      for name, rows in results["archive"].items()}
            panels[f"{label}/{group}"] = dict(
                ordinary_vs_parent=compare_rows(subset["parent"], subset["ordinary"], meta),
                pressure_vs_parent=compare_rows(subset["parent"], subset["pressure"], meta),
                pressure_vs_ordinary=compare_rows(subset["ordinary"], subset["pressure"], meta))
    for label, rows in results["normal"].items():
        panels[f"{label}/generated"] = dict(
            ordinary_vs_parent=compare_rows(rows["parent"], rows["ordinary"], [], normal=True),
            pressure_vs_parent=compare_rows(rows["parent"], rows["pressure"], [], normal=True),
            pressure_vs_ordinary=compare_rows(rows["ordinary"], rows["pressure"], [], normal=True))
    return panels


def audit(directory):
    report = read(directory / "summary.json")
    engine, rules = directory / "junqi.exe", directory / "ruleset.json"
    assert report["engine_sha256"] == sha(engine) and report["rules_sha256"] == sha(rules)
    assert report["profile"] == "archive-pressure"
    assert report["rejected_fixture_reasons"] == ["collision_mismatch", "illegal_move_or_turn"]
    assert report["status"] in ("complete", "insufficient_coverage")
    if report["status"] == "complete":
        train = report["training_sources"]
        holdout = report["evaluation_sources"]
        assert {r["family"] for r in train}.isdisjoint(r["family"] for r in holdout)
        assert len({r["family"] for r in train + holdout}) == len(train + holdout)
        assert all(r["fold"] == "train" for r in train) and all(r["fold"] == "holdout" for r in holdout)
        assert [r["seat"] for r in train if r["group"] == "ordinary"] == [
            r["seat"] for r in train if r["group"] == "pressure"]
        for name, cell in report["training"].items():
            path = directory / f"{name}.json"
            model = load(path, engine, rules, "short-v1")
            assert model["weights"] == cell["weights"] and sha(path) == cell["model_sha256"]
            assert model["parent_sha256"] == report["parent_sha256"]
            validate_training(cell)
        assert report["panels"] == summaries(report["evaluation"], holdout)
    print(f"PASS archive trainer audit: {report['status']}; {directory / 'summary.json'}", flush=True)


def train_group(engine_call, directory, cases, name, weights, parent_hash):
    engine, rules = directory / "junqi.exe", directory / "ruleset.json"
    model_path = directory / f"{name}.json"
    report, = engine_call("archive-curriculum", rules, "train", input_text=feed(cases, weights, TRAIN_SEED))
    validate_training(report)
    if model_path.exists():
        model = load(model_path, engine, rules, "short-v1")
        if model["weights"] != report["weights"] or model["parent_sha256"] != parent_hash:
            raise ValueError("archived checkpoint differs from deterministic training")
    else:
        save(model_path, report["weights"], parent_hash, engine, rules, "short-v1")
    report["model_sha256"] = sha(model_path)
    print(f"trained {name}: W/D/L {report['wins']}/{report['draws']}/{report['losses']}, "
          f"cutoff {report['truncated']}, HQ approach contacts {report['first_hq_pressure']}", flush=True)
    return report


def evaluate(engine_call, rules, cases, roster):
    archive = {}
    for name, weights in roster.items():
        archive[name] = engine_call("archive-curriculum", rules, "assess",
                                    input_text=feed(cases, weights, EVAL_SEED))
        expected = [(i, case[0]["seat"], table, rep) for i, case in enumerate(cases)
                    for table in (0, 4) for rep in range(6)]
        assert [(r["root"], r["seat"], r["table"], r["rep"]) for r in archive[name]] == expected
    normal = {}
    for label, enemy in (("random", "random"), ("contact_enemy", "contact")):
        normal[label] = {name: engine_call(
            "cross-play", rules, EVAL_SEED, 32, "short", "random", enemy, "random", *weights, *([0]*15))
            for name, weights in roster.items()}
    print("evaluated frozen parent and both reloaded children on held-out prefixes and generated layouts", flush=True)
    return dict(archive=archive, normal=normal)


class ArchiveTrainingStopped(RuntimeError):
    pass


def run(directory, hours, audit_only, command, write):
    if audit_only or (directory / "summary.json").exists():
        audit(directory)
        return
    began = time.time()
    phase = None

    def progress(status, step, error=None):
        nonlocal phase
        if status == "running" and phase == step:
            return
        phase = step
        write(directory / "progress.json",
              dict(profile="archive-pressure", status=status, phase=step, started_utc=began,
                   hours_limit=hours, error=error))

    progress("running", "locate")
    try:
        if (directory / "STOP").exists():
            raise ArchiveTrainingStopped("archive training stopped before launch")
        train_archive_run(directory, hours, command, write, progress)
    except ArchiveTrainingStopped as error:
        progress("stopped", phase, str(error))
        raise
    except Exception as error:
        progress("failed", phase, str(error))
        raise


def train_archive_run(directory, hours, command, write, progress):
    started = time.monotonic()
    deadline = started + hours * 3600
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)

    def engine_call(*args, **kw):
        if (directory / "STOP").exists():
            raise ArchiveTrainingStopped("archive training stopped at phase boundary")
        phase = args[2] if args[0] == "archive-curriculum" else "assess"
        progress("running", phase)
        return command(directory, deadline, metrics, *args, **kw)

    engine, rules = directory / "junqi.exe", directory / "ruleset.json"
    for original, target in ((EXE, engine), (RULES, rules)):
        if not target.exists():
            shutil.copyfile(original,target)
    parent = load(PARENT, ARCHIVE / "junqi.exe", ARCHIVE / "ruleset.json", "short-v1")
    cases, filtered, corpus_hash = source_records()
    manifest = dict(engine_sha256=sha(engine), rules_sha256=sha(rules), parent_sha256=sha(PARENT),
                    parent_engine_sha256=sha(ARCHIVE / "junqi.exe"),
                    parent_rules_sha256=sha(ARCHIVE / "ruleset.json"),
                    corpus_sha256=corpus_hash, locator_sha256=sha(INPUT), prefix_plies=PREFIX,
                    source_transform="reflect_global_x_then_rotate_each_family_by_first_hash_byte_mod_4",
                    split_seed=SPLIT_SEED, training_seed=TRAIN_SEED, evaluation_seed=EVAL_SEED)
    frozen = directory / "manifest.json"
    if frozen.exists() and read(frozen) != manifest:
        raise ValueError("frozen replay training run inputs changed")
    if not frozen.exists():
        write(frozen, manifest)
    findings = engine_call("archive-curriculum", rules, "scan", input_text=feed(cases))
    rejected = verify_rejections(engine_call, rules, cases, findings)
    training, holdout = (choose(cases, findings, fold, limit)
                         for fold, limit in (("train", 8), ("holdout", 6)))
    counts = {fold: dict(Counter(row["group"] for case,row in zip(cases,findings,strict=True)
                                 if case[0]["fold"] == fold)) for fold in ("train","holdout")}
    print(f"strict prefix scan {len(cases)}; stage counts {counts}", flush=True)
    report = dict(**manifest, profile="archive-pressure", screened=counts, external_or_short=filtered,
                  rejected_fixture_reasons=rejected,
                  status="insufficient_coverage", training_sources=[], evaluation_sources=[])
    if len(training["pressure"]) >= 12 and len(holdout["pressure"]) >= 8:
        report.update(train_and_assess(engine_call, directory, parent["weights"], training, holdout, manifest))
    report.update(seconds=round(time.monotonic()-started,3), engine_metrics=metrics)
    write(directory / "summary.json", report)
    audit(directory)
    progress(report["status"], "done")


def train_and_assess(engine_call, directory, weights, training, holdout, manifest):
    trained = {name: train_group(engine_call,directory,training[name],name,weights,manifest["parent_sha256"])
               for name in GROUPS}
    targets = holdout["ordinary"] + holdout["pressure"]
    engine, rules = directory / "junqi.exe", directory / "ruleset.json"
    roster = dict(parent=weights, **{name:load(directory / f"{name}.json",engine,rules,"short-v1")["weights"]
                                    for name in GROUPS})
    results = evaluate(engine_call, rules, targets, roster)
    meta = [row[0] for row in targets]
    return dict(status="complete", training=trained, evaluation=results, evaluation_sources=meta,
                training_sources=[case[0] for name in GROUPS for case in training[name]],
                panels=summaries(results,meta))


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        sys.exit("usage: archive_pressure_training.py [audit]")
    cli = [sys.executable, "-X", "utf8", str(ROOT / "tools/bounded_training.py"),
           "--profile", "archive-pressure", "--directory", str(DEFAULT_OUT), "--hours", "1"]
    if sys.argv[1:]:
        cli.append("--audit")
    subprocess.run(cli, check=True, timeout=210)
