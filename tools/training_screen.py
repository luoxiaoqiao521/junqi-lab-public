"""Development screening across representation, training budget and seat context."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evidence_model import sha
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "build/multi-axis-training-screen-20261118"
REPORT = ROOT / "docs/REPORTS/multi-axis-training-screen.json"
SEEDS = (120000, 122000)
EVAL_START = 125000
EVAL_COUNT = 32
POLICIES = ("short", "position", "evidence")


def write(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + chr(10)).encode()
    if sum(p.stat().st_size for p in DIRECTORY.iterdir() if p.is_file()) + len(data) > 32 * 1024**2:
        raise RuntimeError("screen output budget reached")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def command(deadline, *arguments):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or (DIRECTORY / "STOP").exists():
        raise RuntimeError("screen stopped or deadline reached")
    result = subprocess.run([str(DIRECTORY / "junqi.exe"), *map(str, arguments)], check=True,
                            capture_output=True, text=True, timeout=min(180, remaining),
                            creationflags=0x40 | 0x08000000)
    return [json.loads(line) for line in result.stdout.splitlines()]


def evaluate(deadline, weights, policy, variant):
    return command(deadline, "transfer-probe", DIRECTORY / "ruleset.json", *weights,
                   EVAL_START, EVAL_COUNT, policy, variant)


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["hashes"] != dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")):
        raise ValueError("screen executable or rules changed")
    expected = {(seed, policy, budget) for seed in SEEDS for policy in POLICIES for budget in (512, 1024)}
    actual = [(cell["seed"], cell["policy"], cell["budget"]) for cell in report["cells"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("screen design is incomplete or duplicated")
    for cell in report["cells"]:
        width = 5 if cell["policy"] == "short" else 6
        if len(cell["training"]["weights"]) != width:
            raise ValueError("representation width mismatch")
        for context in cell["contexts"]:
            variant, rows = context["variant"], context["rows"]
            for baseline in ("initial", "random", "heuristic"):
                if context["versus_" + baseline] != compare(report["baselines"][f"{baseline}-{variant}"], rows):
                    raise ValueError("comparison differs from raw blocks")
            keys = [(r["seed"], r["seat"]) for r in rows]
            if keys != [(seed, seat) for seed in range(EVAL_START, EVAL_START + EVAL_COUNT) for seat in range(4)]:
                raise ValueError("evaluation distribution changed")
            if totals(rows) != context["totals"]:
                raise ValueError("totals differ")
    print("PASS 12 frozen candidates x 3 contexts; raw paired comparisons audited")


def train_cell(deadline, seed, policy, parent, budget):
    names = {"short": ("train-openings", "train-openings"),
             "position": ("train-position", "resume-position"),
             "evidence": ("train-evidence", "resume-evidence")}
    initial = [0.0] * (6 if policy == "evidence" else 5)
    weights = initial if parent is None else parent["weights"]
    name = names[policy][0 if parent is None else 1]
    if policy == "evidence" and parent is None:
        name = "resume-evidence"
    training, = command(deadline, name, DIRECTORY / "ruleset.json", *weights,
                        seed + budget - 512)
    return training


def run():
    if REPORT.exists() or DIRECTORY.exists():
        raise FileExistsError("screen exists; use audit instead of rerunning")
    DIRECTORY.mkdir(parents=True)
    shutil.copyfile(EXE, DIRECTORY / "junqi.exe")
    shutil.copyfile(RULES, DIRECTORY / "ruleset.json")
    write(DIRECTORY / "manifest.json", dict(total_batches=12, contexts=3))
    started = time.monotonic()
    deadline = started + 1200
    write(DIRECTORY / "summary.json", dict(status="running", started_utc=time.time(), error=None))
    baselines = {f"{name}-{variant}": evaluate(deadline, [0.0] * 5, policy, variant)
                 for variant in range(3)
                 for name, policy in (("initial", "short"), ("random", "random"), ("heuristic", "heuristic"))}
    cells = []
    for stream, seed in enumerate(SEEDS):
        for policy in POLICIES:
            parent = None
            for budget in (512, 1024):
                began = time.monotonic()
                training = train_cell(deadline, seed, policy, parent, budget)
                contexts = []
                for variant in range(3):
                    rows = evaluate(deadline, training["weights"], policy, variant)
                    context = dict(variant=variant, rows=rows, totals=totals(rows))
                    for name in ("initial", "random", "heuristic"):
                        context["versus_" + name] = compare(baselines[f"{name}-{variant}"], rows)
                    contexts.append(context)
                cell = dict(seed=seed, policy=policy, budget=budget, training=training,
                            contexts=contexts, seconds=time.monotonic()-began)
                write(DIRECTORY / f"round-{stream * 3 + POLICIES.index(policy)}-{budget//512}.json", cell)
                cells.append(cell)
                parent = training
                print(json.dumps(dict(seed=seed, policy=policy, budget=budget,
                                      differences=[c["versus_initial"]["difference_bounds"][0] for c in contexts])),
                      flush=True)
    report = dict(cells=cells, baselines=baselines, seconds=time.monotonic()-started,
                  contexts=["all others random", "only partner heuristic", "only opponents heuristic"],
                  hashes=dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")))
    write(REPORT, report)
    write(DIRECTORY / "summary.json", dict(status="complete", completed_rounds=12,
                                          seconds=report["seconds"], error=None))
    audit()


if __name__ == "__main__":
    try:
        audit() if len(sys.argv) > 1 and sys.argv[1] == "audit" else run()
    except Exception as failure:
        if DIRECTORY.exists():
            write(DIRECTORY / "summary.json", dict(status="failed", error=str(failure)))
        raise
