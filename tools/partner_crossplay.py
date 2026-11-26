"""Frozen single-seat comparisons across explicitly paired partners and opponents."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evidence_model import load, sha
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES, np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "build/tuned-policy-confirmation-20261118"
RUN = ROOT / "build/partner-crossplay-v1"
REPORT = ROOT / "docs/REPORTS/partner-crossplay.json"
START, COUNT = 280000, 64
LEARNERS = ("candidate", "initial", "heuristic")
PARTNERS = ("random", "heuristic", "independent", "historical")
ENEMIES = ("random", "heuristic", "learned_enemy")
FILES = {"candidate": "candidate-150000.json", "independent": "candidate-156000.json",
         "historical": "default-162000.json", "learned_enemy": "candidate-168000.json"}


def model_roster():
    if sha(RULES) != sha(ARCHIVE / "ruleset.json"):
        raise ValueError("rules differ from the frozen model archive")
    roster = {name: dict(policy=name if name != "initial" else "short", weights=[0]*5)
              for name in ("random", "heuristic", "initial")}
    for name, filename in FILES.items():
        path = ARCHIVE / filename
        model = load(path, ARCHIVE / "junqi.exe", ARCHIVE / "ruleset.json", "short-v1")
        roster[name] = dict(policy="short", weights=model["weights"], artifact_sha256=sha(path))
    return roster


def evaluate(engine, rules, roster, learner, partner, enemy, start, count, later_enemy=None):
    order = (learner, partner, enemy) if later_enemy is None else (learner, partner, enemy, later_enemy)
    seats = [roster[name] for name in order]
    args = [str(engine), "cross-play", str(rules), str(start), str(count)]
    args.extend(seat["policy"] for seat in seats)
    args.extend(str(value) for seat in seats for value in seat["weights"])
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60,
                            creationflags=0x40 | 0x08000000)
    return [json.loads(line) for line in result.stdout.splitlines()]


def check_legacy(engine, rules, roster):
    for variant, partner, enemy in ((0, "random", "random"), (1, "heuristic", "random"),
                                    (2, "random", "heuristic")):
        new = evaluate(engine, rules, roster, "candidate", partner, enemy, START, 2)
        weights = map(str, roster["candidate"]["weights"])
        result = subprocess.run([str(engine), "transfer-probe", str(rules), *weights,
                                 str(START), "2", "short", str(variant)], check=True,
                                capture_output=True, text=True, timeout=15)
        old = [json.loads(line) for line in result.stdout.splitlines()]
        stable = lambda rows: [{k: v for k, v in row.items() if k != "learner_ns"} for row in rows]
        if stable(new) != stable(old):
            raise ValueError("cross-play changed a legacy seat configuration")


def interaction(reference, candidate, reference_random, candidate_random):
    # Paired difference of differences. Every layout is kept as a four-seat block.
    low, high = [], []
    for a, b, c, d in zip(reference, candidate, reference_random, candidate_random, strict=True):
        values = [r["score"] for r in (b, a, d, c)]
        low.append(sum((value if value is not None else int(sign < 0)) * sign
                       for value, sign in zip(values, (1, -1, -1, 1))))
        high.append(sum((value if value is not None else int(sign > 0)) * sign
                        for value, sign in zip(values, (1, -1, -1, 1))))
    layout_lo = np.asarray(low).reshape(COUNT, 4).mean(axis=1)
    layout_hi = np.asarray(high).reshape(COUNT, 4).mean(axis=1)
    blocks = np.random.default_rng(280100).integers(0, COUNT, (4096, COUNT))
    return dict(bounds=[float(layout_lo.mean()), float(layout_hi.mean())],
                exploratory_interval=[float(np.quantile(layout_lo[blocks].mean(axis=1), .025)),
                                      float(np.quantile(layout_hi[blocks].mean(axis=1), .975))])


def summarize(cells):
    output = {}
    expected = [(seed, seat) for seed in range(START, START + COUNT) for seat in range(4)]
    for enemy in ENEMIES:
        fixed_first = [r["first"] for r in cells[enemy]["random"]["heuristic"]]
        for partner in PARTNERS:
            context = cells[enemy][partner]
            random_partner = cells[enemy]["random"]
            if set(context) != set(LEARNERS):
                raise ValueError("learner matrix is incomplete")
            for rows in context.values():
                if ([(r["seed"], r["seat"]) for r in rows] != expected or
                        [r["first"] for r in rows] != fixed_first or
                        any(r["score"] not in (0, .5, 1, None) for r in rows)):
                    raise ValueError("layout/seat matrix is incomplete")
            output[f"{enemy}/{partner}"] = dict(
                totals={name: totals(rows) for name, rows in context.items()},
                versus_heuristic=compare(context["heuristic"], context["candidate"]),
                versus_initial=compare(context["initial"], context["candidate"]),
                partner_interaction=interaction(context["heuristic"], context["candidate"],
                                                random_partner["heuristic"], random_partner["candidate"]))
    return output


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["hashes"] != dict(engine=sha(RUN / "junqi.exe"), rules=sha(RUN / "ruleset.json")):
        raise ValueError("cross-play engine or rules changed")
    if report["roster"] != model_roster() or report["summary"] != summarize(report["cells"]):
        raise ValueError("cross-play report does not match frozen roster and raw results")
    print("PASS frozen seat profiles, 12 paired contexts, and partner interaction contrasts")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("cross-play exists; use audit")
    roster = model_roster()
    RUN.mkdir()
    engine, rules = RUN / "junqi.exe", RUN / "ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    check_legacy(engine, rules, roster)
    cells = {}
    for enemy in ENEMIES:
        cells[enemy] = {}
        for partner in PARTNERS:
            if time.monotonic() - started > 900:
                raise TimeoutError("cross-play runtime budget")
            cells[enemy][partner] = {
                learner: evaluate(engine, rules, roster, learner, partner, enemy, START, COUNT)
                for learner in LEARNERS}
            print(f"Completed paired context {enemy}/{partner}", flush=True)
    report = dict(roster=roster, cells=cells, summary=summarize(cells),
                  hashes=dict(engine=sha(engine), rules=sha(rules)), seconds=time.monotonic()-started)
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 8 * 1024**2:
        raise ValueError("cross-play output budget")
    with REPORT.open("xb") as output:
        output.write(data)
    audit()


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
