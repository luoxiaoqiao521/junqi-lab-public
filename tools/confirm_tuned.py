"""Frozen confirmation of the selected update method on new seeds and layouts."""
import json
from pathlib import Path
import shutil
import sys
import time

from bounded_training import command
from evidence_model import load, save, sha
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES, np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "build/tuned-policy-confirmation-20261118"
REPORT = ROOT / "docs/REPORTS/tuned-policy-confirmation.json"
SEEDS = (150000, 156000, 162000, 168000)
LAYOUT, COUNT = 180000, 128


def write(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + chr(10)).encode()
    if sum(p.stat().st_size for p in DIRECTORY.iterdir() if p.is_file()) + len(data) > 32 * 1024**2:
        raise RuntimeError("confirmation output budget")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def evaluate(deadline, metrics, weights, policy, variant):
    return command(DIRECTORY, deadline, metrics, "transfer-probe", DIRECTORY / "ruleset.json",
                   *weights, LAYOUT, COUNT, policy, variant)


def crossed(pairs):
    low, high = [], []
    for baseline, candidate in pairs:
        compare(baseline, candidate)
        low.append([(b["score"] or 0) - (a["score"] if a["score"] is not None else 1)
                    for a, b in zip(baseline, candidate)])
        high.append([(b["score"] if b["score"] is not None else 1) - (a["score"] or 0)
                     for a, b in zip(baseline, candidate)])
    lo = np.asarray(low).reshape(len(pairs), COUNT, 4).mean(axis=2)
    hi = np.asarray(high).reshape(len(pairs), COUNT, 4).mean(axis=2)
    rng = np.random.default_rng(180500)
    streams = rng.integers(0, len(pairs), size=(4096, len(pairs)))
    layouts = rng.integers(0, COUNT, size=(4096, COUNT))
    lo_samples = lo[streams[:, :, None], layouts[:, None, :]].mean(axis=(1, 2))
    hi_samples = hi[streams[:, :, None], layouts[:, None, :]].mean(axis=(1, 2))
    return dict(difference_bounds=[float(lo.mean()), float(hi.mean())],
                crossed_interval=[float(np.quantile(lo_samples, .025)),
                                  float(np.quantile(hi_samples, .975))])


def aggregate(streams, baselines):
    results = {}
    for variant in range(3):
        for name in ("random", "heuristic", "default"):
            pairs = [(s["default"][variant] if name == "default" else baselines[f"{name}-{variant}"],
                      s["candidate"][variant]) for s in streams]
            results[f"{name}-{variant}"] = crossed(pairs)
    return results


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["hashes"] != dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")):
        raise ValueError("frozen confirmation engine changed")
    if [s["seed"] for s in report["streams"]] != list(SEEDS):
        raise ValueError("confirmation streams differ")
    for stream in report["streams"]:
        for kind in ("candidate", "default"):
            model = load(stream[kind + "_path"], DIRECTORY / "junqi.exe",
                         DIRECTORY / "ruleset.json", "short-v1")
            if model["weights"] != stream[kind + "_training"]["weights"]:
                raise ValueError("saved model differs from trained model")
            for rows in stream[kind]:
                if [(r["seed"], r["seat"]) for r in rows] != [
                        (seed, seat) for seed in range(LAYOUT, LAYOUT + COUNT) for seat in range(4)]:
                    raise ValueError("confirmation layout range differs")
    if aggregate(report["streams"], report["baselines"]) != report["aggregate"]:
        raise ValueError("confirmation inference differs from raw results")
    print("PASS frozen model identities, fresh seed ranges and crossed-block confirmation")


def run():
    if DIRECTORY.exists() or REPORT.exists():
        raise FileExistsError("confirmation exists; audit instead")
    DIRECTORY.mkdir(parents=True)
    shutil.copyfile(EXE, DIRECTORY / "junqi.exe")
    shutil.copyfile(RULES, DIRECTORY / "ruleset.json")
    write(DIRECTORY / "manifest.json", dict(total_batches=8, training_seeds=SEEDS,
                                           primary="heuristic-0", candidate_rate=.15))
    write(DIRECTORY / "summary.json", dict(status="running", started_utc=time.time(), error=None))
    started = time.monotonic()
    deadline = started + 6 * 3600
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    baselines = {f"{name}-{variant}": evaluate(deadline, metrics, [0.0]*5, policy, variant)
                 for variant in range(3) for name, policy in
                 (("initial", "short"), ("random", "random"), ("heuristic", "heuristic"))}
    streams = []
    for index, seed in enumerate(SEEDS):
        stream = dict(seed=seed)
        for number, (kind, rate) in enumerate((("candidate", .15), ("default", .015))):
            began = time.monotonic()
            training, = command(DIRECTORY, deadline, metrics, "train-tuned",
                                DIRECTORY / "ruleset.json", seed, rate, 0, 256)
            path = DIRECTORY / f"{kind}-{seed}.json"
            save(path, training["weights"], executable=DIRECTORY / "junqi.exe",
                 rules=DIRECTORY / "ruleset.json", encoding="short-v1")
            model = load(path, DIRECTORY / "junqi.exe", DIRECTORY / "ruleset.json", "short-v1")
            rows = [evaluate(deadline, metrics, model["weights"], "short", v) for v in range(3)]
            stream[kind], stream[kind + "_training"], stream[kind + "_path"] = rows, training, str(path)
            write(DIRECTORY / f"round-{index}-{number}.json",
                  dict(seed=seed, kind=kind, totals=[totals(group) for group in rows],
                       seconds=time.monotonic()-began))
        streams.append(stream)
        print(json.dumps(dict(seed=seed, candidate=[totals(rows) for rows in stream["candidate"]],
                              default=[totals(rows) for rows in stream["default"]])), flush=True)
    results = aggregate(streams, baselines)
    report = dict(streams=streams, baselines=baselines, aggregate=results,
                  seconds=time.monotonic()-started, metrics=metrics,
                  hashes=dict(engine=sha(DIRECTORY / "junqi.exe"), rules=sha(DIRECTORY / "ruleset.json")))
    write(REPORT, report)
    primary = results["heuristic-0"]
    write(DIRECTORY / "summary.json", dict(status="complete", completed_rounds=8,
          primary=primary, primary_positive=primary["crossed_interval"][0] > 0, error=None))
    audit()


if __name__ == "__main__":
    try:
        audit() if len(sys.argv) > 1 and sys.argv[1] == "audit" else run()
    except Exception as failure:
        if DIRECTORY.exists():
            write(DIRECTORY / "summary.json", dict(status="failed", error=str(failure)))
        raise
