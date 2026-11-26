"""Hold the parent, slots and game count fixed while changing the continuation rate."""
from collections import Counter
from pathlib import Path
import json
import shutil
import sys
import time

from bounded_training import command
from confirm_tuned import COUNT, crossed
from evaluate_transfer import compare, totals
from evidence_model import load, save, sha
from learn_short import EXE, RULES
from partner_crossplay import evaluate, model_roster
from rank_experiment_families import REPORTS, audit as audit_ranking, digest, read

RUN = REPORTS.parents[1]/"build/continuation-rate-trial-v1"
REPORT = REPORTS/"continuation-rate-trial.json"
SEEDS = (450000, 460000, 470000, 480000)
RATES = {"prior": .15, "smaller": .015}
BATCHES, START = 64, 299000
PARTNERS = {"normal": "random", "new_partner": "historical"}
MIN_GAIN, PARTNER_TOLERANCE = .01, -.01


def frozen():
    roster = model_roster()
    profiles = {name: roster[name] for name in ("candidate", "random", "historical")}
    ranking = audit_ranking()["evidence"]
    if ranking["selected"] != "whole-game-update":
        raise ValueError("frozen evidence did not authorize the continuation trial")
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"),
                roster=profiles, ranking=digest(ranking))


def validate_training(result, rate):
    games = BATCHES*16
    if (result["rate"] != rate or result["return_baseline"] or result["batches"] != BATCHES
            or result["training_profile"] != 0
            or result["natural"] + result["truncated"] != games
            or sum(result[key] for key in ("wins", "draws", "losses")) != result["natural"]):
        raise ValueError("training changed a frozen variable or budget")


def train(seed, rate, parent, start, metrics, phase_seconds):
    training, = command(RUN, start+phase_seconds, metrics, "train-general", RUN/"ruleset.json",
                        seed, rate, 0, BATCHES, "short", 0, *parent["weights"])
    validate_training(training, rate)
    return training


def score_context(profiles, weights, context):
    roster = dict(profiles, tested=dict(policy="short", weights=weights))
    return evaluate(RUN/"junqi.exe", RUN/"ruleset.json", roster, "tested", PARTNERS[context],
                    "random", START, COUNT, later_enemy="random")


def validate_rows(rows):
    expected = [(seed, seat) for seed in range(START, START+COUNT) for seat in range(4)]
    if ([(r["seed"], r["seat"]) for r in rows] != expected or
            any(r["score"] not in (0, .5, 1, None) for r in rows)):
        raise ValueError("evaluation layout coverage or terminal semantics changed")


def summarize(parent, streams):
    if [s["seed"] for s in streams] != list(SEEDS):
        raise ValueError("training stream coverage changed")
    contexts = {}
    for label in PARTNERS:
        base = parent[label]
        validate_rows(base)
        pairs = {name: [] for name in ("smaller_minus_prior", "smaller_minus_parent", "prior_minus_parent")}
        by_seed, transitions = [], Counter()
        for stream in streams:
            prior, smaller = (stream[name]["evaluation"][label] for name in RATES)
            for rows in (prior, smaller):
                validate_rows(rows)
            grouped = dict(smaller_minus_prior=(prior, smaller), smaller_minus_parent=(base, smaller),
                           prior_minus_parent=(base, prior))
            for key, games in grouped.items():
                pairs[key].append(games)
            transitions.update(f"{a['score']}->{b['score']}" for a, b in zip(prior, smaller, strict=True))
            by_seed.append(dict(seed=stream["seed"], prior=totals(prior), smaller=totals(smaller),
                                contrasts={key: compare(*games) for key, games in grouped.items()}))
        contexts[label] = dict(frozen_parent=totals(base), by_seed=by_seed, transitions=dict(transitions),
                               crossed={name: crossed(games) for name, games in pairs.items()})
    normal, partner = (contexts[label]["crossed"] for label in PARTNERS)
    eligible = (all(normal[key]["difference_bounds"][0] >= MIN_GAIN and normal[key]["crossed_interval"][0] > 0
                    for key in ("smaller_minus_prior", "smaller_minus_parent"))
                and partner["smaller_minus_parent"]["difference_bounds"][0] >= 0
                and partner["smaller_minus_parent"]["crossed_interval"][0] >= PARTNER_TOLERANCE)
    next_step = "replicate-conservative-continuation" if eligible else "audit-frozen-parent-update-stability"
    return dict(contexts=contexts, eligible_for_replication=eligible, next=next_step)


def replay_without_timing(rows):
    return [{k: v for k, v in row.items() if k != "learner_ns"} for row in rows]


def artifact(seed, name):
    return RUN/f"{seed}-{name}.json"


def audit():
    report = read(REPORT)
    inputs = frozen()
    if (report["frozen"] != inputs or report["budget"] != dict(episodes_per_arm=BATCHES*16, rates=RATES,
            primary="smaller_minus_prior/normal", min_gain=MIN_GAIN, partner_tolerance=PARTNER_TOLERANCE)):
        raise ValueError("trial contract changed")
    streams = report["streams"]
    parent = inputs["roster"]["candidate"]
    for stream in streams:
        for name, rate in RATES.items():
            cell = stream[name]
            validate_training(cell["training"], rate)
            path = artifact(stream["seed"], name)
            saved = load(path, RUN/"junqi.exe", RUN/"ruleset.json", "short-v1")
            if (cell["model_sha256"] != sha(path) or saved["weights"] != cell["training"]["weights"]
                    or saved["parent_sha256"] != parent["artifact_sha256"]
                    or set(cell["evaluation"]) != set(PARTNERS)):
                raise ValueError("incomplete stored weights or evaluation")
    if set(report["parent"]) != set(PARTNERS) or report["summary"] != summarize(report["parent"], streams):
        raise ValueError("paired difference or independent context changed")
    first = streams[0]["smaller"]
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    if first["training"] != train(SEEDS[0], RATES["smaller"], parent, time.monotonic(), metrics, 60):
        raise ValueError("frozen training replay differed")
    sample = score_context(inputs["roster"], first["training"]["weights"], "new_partner")
    if replay_without_timing(sample) != replay_without_timing(first["evaluation"]["new_partner"]):
        raise ValueError("frozen cross-play replay differed")
    print("PASS equal budgets, declared contexts, paired effects and frozen replays")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("continuation trial exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE, RUN/"junqi.exe")
    shutil.copyfile(RULES, RUN/"ruleset.json")
    inputs = frozen()
    parent = inputs["roster"]["candidate"]
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    started, training_seconds = time.monotonic(), 0.
    parent_rows = {label: score_context(inputs["roster"], parent["weights"], label) for label in PARTNERS}
    streams = []
    for seed in SEEDS:
        stream = dict(seed=seed)
        for name, rate in RATES.items():
            before = time.monotonic()
            training = train(seed, rate, parent, started, metrics, 225)
            training_seconds += time.monotonic()-before
            path = artifact(seed, name)
            model = save(path, training["weights"], parent["artifact_sha256"],
                         RUN/"junqi.exe", RUN/"ruleset.json", "short-v1")
            evaluation = {label: score_context(inputs["roster"], model["weights"], label) for label in PARTNERS}
            stream[name] = dict(training=training, model_sha256=sha(path), evaluation=evaluation)
            print(f"completed seed={seed} rate={rate}", flush=True)
        streams.append(stream)
    summary = summarize(parent_rows, streams)
    data = dict(frozen=inputs, parent=parent_rows, streams=streams, summary=summary, metrics=metrics,
                seconds=time.monotonic()-started, training_seconds=training_seconds,
                budget=dict(episodes_per_arm=BATCHES*16, rates=RATES, primary="smaller_minus_prior/normal",
                            min_gain=MIN_GAIN, partner_tolerance=PARTNER_TOLERANCE))
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(content) > 2*1024**2:
        raise ValueError("paired continuation exceeded its artifact budget")
    with REPORT.open("xb") as output:
        output.write(content)
    print(json.dumps(dict(next=summary["next"])))


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
