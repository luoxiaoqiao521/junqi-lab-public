"""Test representation and frozen training-population interactions at two budgets."""
import json
from pathlib import Path
import shutil
import sys
import time

from bounded_training import command
from evidence_model import load, save, sha
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "build/representation-distribution-learning-20261118"
REPORT = ROOT / "docs/REPORTS/representation-distribution-learning.json"
SEEDS, POLICIES, BUDGETS = (200000, 220000), ("short", "position", "evidence"), (4096, 16384)
ENCODINGS = dict(short="short-v1", position="position-v1", evidence="public-engineer-v1")
LAYOUT, COUNT = 250000, 64


def write(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + chr(10)).encode()
    if sum(p.stat().st_size for p in DIRECTORY.iterdir() if p.is_file()) + len(data) > 32 * 1024**2:
        raise RuntimeError("distribution study storage limit")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def evaluate(deadline, metrics, weights, policy, variant):
    return command(DIRECTORY, deadline, metrics, "transfer-probe", DIRECTORY / "ruleset.json",
                   *weights, LAYOUT, COUNT, policy, variant)


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["hashes"] != dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")):
        raise ValueError("frozen distribution study changed")
    expected = {(s, p, f, b) for s in SEEDS for p in POLICIES for f in (0, 3) for b in BUDGETS}
    actual = [(c["seed"], c["policy"], c["profile"], c["budget"]) for c in report["cells"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("distribution study design incomplete")
    controls = {(c["seed"], c["profile"], c["budget"]): c
                for c in report["cells"] if c["policy"] == "short"}
    for cell in report["cells"]:
        model = load(cell["path"], DIRECTORY / "junqi.exe", DIRECTORY / "ruleset.json",
                     ENCODINGS[cell["policy"]])
        if model["weights"] != cell["training"]["weights"]:
            raise ValueError("persisted distribution model differs")
        reference = controls[cell["seed"], cell["profile"], cell["budget"]]
        if [context["variant"] for context in cell["contexts"]] != [0, 1, 2]:
            raise ValueError("incomplete or duplicate evaluation contexts")
        for context in cell["contexts"]:
            variant = context["variant"]
            for name in ("initial", "random", "heuristic"):
                if context["versus_" + name] != compare(report["baselines"][f"{name}-{variant}"], context["rows"]):
                    raise ValueError("paired baseline report differs")
            if context["versus_short"] != compare(reference["contexts"][variant]["rows"], context["rows"]):
                raise ValueError("paired representation comparison differs")
            if totals(context["rows"]) != context["totals"]:
                raise ValueError("distribution totals differ")
    print("PASS 24 typed models, frozen populations and raw paired comparisons")


def run():
    if DIRECTORY.exists() or REPORT.exists():
        raise FileExistsError("distribution study exists; audit instead")
    DIRECTORY.mkdir(parents=True)
    shutil.copyfile(EXE, DIRECTORY / "junqi.exe")
    shutil.copyfile(RULES, DIRECTORY / "ruleset.json")
    write(DIRECTORY / "manifest.json", dict(total_batches=24, training_seeds=SEEDS,
                                           budgets=BUDGETS, profiles=[0, 3]))
    write(DIRECTORY / "summary.json", dict(status="running", started_utc=time.time(), error=None))
    started, metrics = time.monotonic(), dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    deadline = started + 6 * 3600
    baselines = {f"{name}-{v}": evaluate(deadline, metrics, [0.0]*5, policy, v)
                 for v in range(3) for name, policy in
                 (("initial", "short"), ("random", "random"), ("heuristic", "heuristic"))}
    cells, controls = [], {}
    for stream, seed in enumerate(SEEDS):
        for policy in POLICIES:
            for profile in (0, 3):
                for budget in BUDGETS:
                    began = time.monotonic()
                    training, = command(DIRECTORY, deadline, metrics, "train-general",
                                        DIRECTORY / "ruleset.json", seed, .15, 0, budget//16, policy, profile)
                    path = DIRECTORY / f"model-{seed}-{policy}-{profile}-{budget}.json"
                    save(path, training["weights"], executable=DIRECTORY / "junqi.exe",
                         rules=DIRECTORY / "ruleset.json", encoding=ENCODINGS[policy])
                    model = load(path, DIRECTORY / "junqi.exe", DIRECTORY / "ruleset.json", ENCODINGS[policy])
                    contexts = []
                    for variant in range(3):
                        rows = evaluate(deadline, metrics, model["weights"], policy, variant)
                        context = dict(variant=variant, rows=rows, totals=totals(rows))
                        for name in ("initial", "random", "heuristic"):
                            context["versus_" + name] = compare(baselines[f"{name}-{variant}"], rows)
                        if policy == "short":
                            controls[seed, profile, budget, variant] = rows
                        context["versus_short"] = compare(controls[seed, profile, budget, variant], rows)
                        contexts.append(context)
                    cell = dict(seed=seed, policy=policy, profile=profile, budget=budget, path=str(path),
                                training=training, contexts=contexts, seconds=time.monotonic()-began)
                    write(DIRECTORY / f"round-{stream}-{len(cells)}.json", cell)
                    cells.append(cell)
                    print(json.dumps(dict(seed=seed, policy=policy, profile=profile, budget=budget,
                                          differences=[c["versus_heuristic"]["difference_bounds"][0]
                                                       for c in contexts])), flush=True)
    report = dict(cells=cells, baselines=baselines, seconds=time.monotonic()-started, metrics=metrics,
                  hashes=dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")))
    write(REPORT, report)
    write(DIRECTORY / "summary.json", dict(status="complete", completed_rounds=24,
                                          seconds=report["seconds"], error=None))
    audit()


if __name__ == "__main__":
    try:
        audit() if len(sys.argv) > 1 and sys.argv[1] == "audit" else run()
    except Exception as failure:
        if DIRECTORY.exists():
            write(DIRECTORY / "summary.json", dict(status="failed", error=str(failure)))
        raise
