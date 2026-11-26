"""Predeclared learning-rate and return-estimator development comparison."""
import json
from pathlib import Path
import shutil
import sys
import time

from bounded_training import command
from evidence_model import sha
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "build/training-update-screen-20261118"
REPORT = ROOT / "docs/REPORTS/training-update-screen.json"
SEEDS = (130000, 136000)
RATES = (.0015, .015, .15)
LAYOUT = 145000
COUNT = 64


def write(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + chr(10)).encode()
    if sum(p.stat().st_size for p in DIRECTORY.iterdir() if p.is_file()) + len(data) > 32 * 1024**2:
        raise RuntimeError("update screen output budget")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def evaluate(deadline, metrics, weights, policy, variant):
    return command(DIRECTORY, deadline, metrics, "transfer-probe", DIRECTORY / "ruleset.json",
                   *weights, LAYOUT, COUNT, policy, variant)


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["hashes"] != dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")):
        raise ValueError("update screen engine changed")
    expected = {(s, r, b, n) for s in SEEDS for r in RATES for b in (0, 1) for n in (512, 4096)}
    actual = [(c["seed"], c["rate"], c["baseline"], c["budget"]) for c in report["cells"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("update factorial design incomplete")
    for cell in report["cells"]:
        if cell["training"]["natural"] + cell["training"]["truncated"] != cell["budget"]:
            raise ValueError("training budget mismatch")
        if [context["variant"] for context in cell["contexts"]] != [0, 1]:
            raise ValueError("incomplete or duplicate evaluation contexts")
        for context in cell["contexts"]:
            variant = context["variant"]
            for key in ("initial", "random", "heuristic"):
                if compare(report["baselines"][f"{key}-{variant}"], context["rows"]) != context["versus_" + key]:
                    raise ValueError("update paired difference mismatch")
            if totals(context["rows"]) != context["totals"]:
                raise ValueError("update totals mismatch")
    print("PASS 24 update configurations, declared budgets and raw comparisons")


def run():
    if DIRECTORY.exists() or REPORT.exists():
        raise FileExistsError("existing update screen; audit it")
    DIRECTORY.mkdir(parents=True)
    shutil.copyfile(EXE, DIRECTORY / "junqi.exe")
    shutil.copyfile(RULES, DIRECTORY / "ruleset.json")
    write(DIRECTORY / "manifest.json", dict(total_batches=24, rates=RATES, starts=SEEDS))
    started = time.monotonic()
    deadline = started + 6 * 3600
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    write(DIRECTORY / "summary.json", dict(status="running", started_utc=time.time(), error=None))
    baselines = {f"{name}-{variant}": evaluate(deadline, metrics, [0.0]*5, policy, variant)
                 for variant in (0, 1)
                 for name, policy in (("initial", "short"), ("random", "random"), ("heuristic", "heuristic"))}
    cells = []
    for seed in SEEDS:
        for rate in RATES:
            for baseline in (0, 1):
                for budget in (512, 4096):
                    began = time.monotonic()
                    training, = command(DIRECTORY, deadline, metrics, "train-tuned",
                                        DIRECTORY / "ruleset.json", seed, rate, baseline, budget//16)
                    contexts = []
                    for variant in (0, 1):
                        rows = evaluate(deadline, metrics, training["weights"], "short", variant)
                        context = dict(variant=variant, rows=rows, totals=totals(rows))
                        for name in ("initial", "random", "heuristic"):
                            context["versus_" + name] = compare(baselines[f"{name}-{variant}"], rows)
                        contexts.append(context)
                    cell = dict(seed=seed, rate=rate, baseline=baseline, budget=budget, training=training,
                                contexts=contexts, seconds=time.monotonic()-began)
                    write(DIRECTORY / f"round-{len(cells)}-1.json", cell)
                    cells.append(cell)
                    print(json.dumps(dict(seed=seed, rate=rate, baseline=baseline, budget=budget,
                                          differences=[c["versus_random"]["difference_bounds"][0]
                                                       for c in contexts])), flush=True)
    report = dict(cells=cells, baselines=baselines, seconds=time.monotonic()-started,
                  metrics=metrics, hashes=dict(engine=sha(DIRECTORY / "junqi.exe"),
                                               rules=sha(DIRECTORY / "ruleset.json")))
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
