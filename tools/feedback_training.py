"""Paired pilot of two continuation distributions from one frozen parent."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evidence_model import load, save, sha
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES
from partner_crossplay import evaluate, model_roster

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/feedback-training-mixture-v1"
REPORT = ROOT / "docs/REPORTS/feedback-training-mixture.json"
LAYOUT, COUNT = 285000, 32
SEEDS = (400000, 410000)
PROFILES = (0, 4)
BATCHES = 64
CONTEXTS = {
    "random": ("random", "random", "random"),
    "heldout-partner": ("historical", "random", "random"),
    "heuristic-enemies": ("random", "heuristic", "heuristic"),
    "contact-diagnostic": ("random", "contact", "random"),
}


def run_training(engine, rules, weights, seed, profile):
    args = [engine, "train-general", rules, seed, .15, 0, BATCHES, "short", profile, *weights]
    run = subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True,
                         timeout=80, creationflags=0x40 | 0x08000000)
    result = json.loads(run.stdout)
    validate_training(result, profile)
    return result


def validate_training(report, profile):
    games = BATCHES * 16
    if (report["rate"] != .15 or report["return_baseline"] or report["batches"] != BATCHES or
            report["training_profile"] != profile or report["natural"] + report["truncated"] != games or
            sum(report[k] for k in ("wins", "draws", "losses")) != report["natural"]):
        raise ValueError("training result differs from declared controls")
    if profile == 4 and report["contexts"] != [games//2, 0, 0, 0, games//4, games//4]:
        raise ValueError("mixed sampling budget differs from plan")


def score_contexts(engine, rules, profiles, weights):
    players = dict(profiles, tested=dict(policy="short", weights=weights))
    return {label: evaluate(engine, rules, players, "tested", partner, before, LAYOUT,
                            COUNT, later_enemy=after)
            for label, (partner, before, after) in CONTEXTS.items()}


def summarize(parent, cells):
    if [(c["seed"], c["profile"]) for c in cells] != [(s, p) for s in SEEDS for p in PROFILES]:
        raise ValueError("invalid paired continuation grid")
    expected = [(seed, seat) for seed in range(LAYOUT, LAYOUT+COUNT) for seat in range(4)]
    for evaluation in [parent, *(c["evaluation"] for c in cells)]:
        if set(evaluation) != set(CONTEXTS):
            raise ValueError("incomplete frozen context grid")
        for rows in evaluation.values():
            if ([(r["seed"], r["seat"]) for r in rows] != expected or
                    any(r["score"] not in (0, .5, 1, None) for r in rows)):
                raise ValueError("invalid matched continuation rows")
    result = []
    for control, mixed in zip(cells[::2], cells[1::2], strict=True):
        comparisons = {}
        for context in CONTEXTS:
            a, b = (c["evaluation"][context] for c in (control, mixed))
            comparisons[context] = dict(parent=totals(parent[context]), control=totals(a),
                mixed=totals(b), mixed_vs_control=compare(a, b),
                control_vs_parent=compare(parent[context], a), mixed_vs_parent=compare(parent[context], b))
        result.append(dict(seed=mixed["seed"], comparisons=comparisons))
    return result


def frozen_roster():
    return dict(model_roster(), contact=dict(policy="contact", weights=[0]*5))


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    profiles = frozen_roster()
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), profiles=profiles):
        raise ValueError("frozen artifacts or profiles changed")
    for cell in data["cells"]:
        seed, profile = cell["seed"], cell["profile"]
        artifact = RUN/f"{seed}-{profile}.json"
        model = load(artifact, engine, rules, "short-v1")
        validate_training(cell["training"], profile)
        if (cell["model_sha256"] != sha(artifact) or model["weights"] != cell["training"]["weights"] or
                model["parent_sha256"] != profiles["candidate"]["artifact_sha256"]):
            raise ValueError("reloaded continuation differs from its receipt")
    if data["summary"] != summarize(data["parent"], data["cells"]):
        raise ValueError("continuation summaries differ from raw paired results")
    print("PASS equal training budget, frozen parent hashes, reloaded children and paired evaluations")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("continuation archive exists; use audit")
    profiles = frozen_roster()
    parent = profiles["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    reference = score_contexts(engine, rules, profiles, parent["weights"])
    cells = []
    for seed in SEEDS:
        for profile in PROFILES:
            training = run_training(engine, rules, parent["weights"], seed, profile)
            artifact = RUN/f"{seed}-{profile}.json"
            saved = save(artifact, training["weights"], parent["artifact_sha256"],
                         engine, rules, "short-v1")
            evaluation = score_contexts(engine, rules, profiles, saved["weights"])
            cells.append(dict(seed=seed, profile=profile, training=training,
                              model_sha256=sha(artifact), evaluation=evaluation))
            print(f"Completed seed={seed} profile={profile}", flush=True)
    report = dict(parent=reference, cells=cells, summary=summarize(reference, cells),
                  seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), profiles=profiles))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 2*1024**2:
        raise ValueError("continuation output budget exceeded")
    with REPORT.open("xb") as output:
        output.write(data)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
