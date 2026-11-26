"""Measure delayed paired action contrasts from legal normal-opening games."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/feedback-credit-window-v1"
REPORT = ROOT / "docs/REPORTS/feedback-credit-window.json"
COHORTS = {"course-source": 286000, "development": 286200}
LAYOUTS = 32
BANDS = ((4, 16), (17, 48))
SOURCE_VARIANTS = (0, 2)
SAMPLES = 8
PAIR_KINDS = ("same-five", "different-five")
MIN_REPLICATED_LAYOUTS = 4
SCORES = (-1, 0, 1, 2)


def collect(engine, rules, parent, seed, seat, variant, reseed=False, prefer_live=False):
    command = "credit-windows"
    if reseed:
        command = "credit-windows-reseed"
    if prefer_live:
        if reseed:
            raise ValueError("vary only one factor in a credit-window replay")
        command = "credit-windows-live"
    result = subprocess.run(
        [str(engine), command, str(rules), *map(str, (seed, seat, variant, *parent["weights"]))],
        check=True, capture_output=True, text=True, timeout=15, creationflags=0x40 | 0x08000000)
    return json.loads(result.stdout)


def action_key(action):
    if (len(action) != 11 or
            any(type(node) is not int or not 0 <= node < 129 for node in action[:2]) or
            action[0] == action[1]):
        raise ValueError("invalid action record")
    return action[:2]


def classify(cells, worlds):
    if len(cells) != worlds * 2 or any(len(cell) != SAMPLES for cell in cells):
        raise ValueError("incomplete or duplicate response budget")
    signs = []
    supports = []
    any_truncated = any_immediate = False
    all_draws = all_tied = True
    for cell in cells:
        deltas = []
        for a, b, a_plies, b_plies in cell:
            if a not in SCORES or b not in SCORES or min(a_plies, b_plies) < 1:
                raise ValueError("invalid branch score or duration")
            any_truncated |= -1 in (a, b)
            any_immediate |= 1 in (a_plies, b_plies)
            all_draws &= a == b == 1
            all_tied &= a == b
            if -1 not in (a, b):
                deltas.append(b - a)
        delta = sum(deltas)
        signs.append((delta > 0) - (delta < 0))
        supports.append(sum(d > 0 for d in deltas) - sum(d < 0 for d in deltas))
    if worlds < 2:
        return "no-live-alternate", 0
    if any_truncated:
        return "truncated", 0
    if any_immediate:
        return "immediate-end", 0
    if all_draws:
        return "all-draw", 0
    if all_tied:
        return "tied", 0
    if -1 in signs and 1 in signs:
        return "reversal", 0
    if min(supports) >= 2 and min(signs) == 1:
        return "consistent", 1
    if max(supports) <= -2 and max(signs) == -1:
        return "consistent", -1
    return "weak-or-zero-cell", 0


def count_window(window, kind, metrics, examples, row):
    played = window["played"]
    action_key(played)
    alt = window["alternatives"][PAIR_KINDS.index(kind)]
    if not alt:
        metrics["missing-pair"] += 1
        return
    action = alt["action"]
    if action_key(played) == action_key(action) or (played[2:7] == action[2:7]) != (kind == "same-five"):
        raise ValueError("invalid paired action encoding")
    metrics["rollout_pairs"] += sum(map(len, alt["cells"]))
    status, direction = classify(alt["cells"], window["worlds"])
    metrics[status] += 1
    if direction:
        examples[kind].append(dict(seed=row["seed"], seat=row["seat"], variant=row["variant"],
            ply=window["ply"], remaining=window["remaining"], source_score=row["score"],
            better="alternative" if direction > 0 else "played",
            position_distinguishes=played[7] != action[7], legacy_distinguishes=played[8:] != action[8:]))


def expected_rows(start):
    return [(seed, (seed-start) % 4, variant) for seed in range(start, start+LAYOUTS)
            for variant in SOURCE_VARIANTS]


def cohort_summary(rows, start):
    actual = [(row["seed"], row["seat"], row["variant"]) for row in rows]
    if actual != expected_rows(start):
        raise ValueError("missing, duplicate or reordered source layouts")
    sources = defaultdict(Counter)
    metrics = defaultdict(Counter)
    examples = defaultdict(list)
    for row in rows:
        if row["score"] not in SCORES or row["plies"] < 1 or len(row["windows"]) != len(BANDS):
            raise ValueError("invalid source outcome or window grid")
        sources[str(row["variant"])][str(row["score"])] += 1
        for (lower, upper), window in zip(BANDS, row["windows"], strict=True):
            for kind in PAIR_KINDS:
                counts = metrics[f'{row["variant"]}/{lower}-{upper}/{kind}']
                if window is None:
                    counts["missing-window"] += 1
                    continue
                if (window["worlds"] not in (1, 2) or not lower <= window["remaining"] <= upper or
                        window["ply"] + window["remaining"] != row["plies"] or row["score"] == -1):
                    raise ValueError("invalid window timing or world count")
                count_window(window, kind, counts, examples, row)
    return dict(sources={k: dict(v) for k, v in sources.items()},
                cells={k: dict(v) for k, v in metrics.items()},
                examples={k: examples[k] for k in PAIR_KINDS})


def summarize(cohorts):
    if set(cohorts) != set(COHORTS):
        raise ValueError("unexpected cohort grid")
    results = {name: cohort_summary(cohorts[name], start) for name, start in COHORTS.items()}
    layouts = {name: {kind: len({example["seed"] for example in summary["examples"][kind]})
                      for kind in PAIR_KINDS} for name, summary in results.items()}
    candidates = [kind for kind in PAIR_KINDS
                  if all(group[kind] >= MIN_REPLICATED_LAYOUTS for group in layouts.values())]
    return dict(cohorts=results, consistent_layouts=layouts, candidates=candidates)


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if report["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("frozen credit-window inputs changed")
    if report["summary"] != summarize(report["cohorts"]):
        raise ValueError("credit-window classification differs from paired rows")
    bad = dict(report["cohorts"])
    bad["development"] = bad["development"][:-1]
    try:
        summarize(bad)
    except ValueError:
        pass
    else:
        raise ValueError("missing layout row passed audit")
    for rows in report["cohorts"].values():
        for row in (rows[0], rows[-1]):
            if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"]):
                raise ValueError("frozen source game or counterfactual rows did not reproduce")
    print("PASS source and response grids, hidden-world checks, blocked missing layout, frozen replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("credit-window archive exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    cohorts = {}
    for name, start in COHORTS.items():
        cohorts[name] = [collect(engine, rules, parent, seed, seat, variant)
                         for seed, seat, variant in expected_rows(start)]
        print(f"Completed {name} {len(cohorts[name])} source games", flush=True)
    report = dict(cohorts=cohorts, summary=summarize(cohorts), seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 2*1024**2:
        raise ValueError("credit-window output byte budget exceeded")
    with REPORT.open("xb") as output:
        output.write(data)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
