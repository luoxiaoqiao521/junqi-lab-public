"""Recount frozen outcomes and run an offline decision contrast on new layouts."""
from collections import Counter
import json
from math import comb, isclose
from pathlib import Path
import statistics
import subprocess
import sys
import time

from evidence_model import load, sha
from evaluate_transfer import compare
from learn_short import EXE, RULES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/REPORTS/tuned-policy-confirmation.json"
DESTINATION = ROOT / "docs/REPORTS/decision-review.json"
SCORES = {0: "loss", .5: "draw", 1: "win", None: "truncated"}
LAYOUT_START = 271000
LAYOUT_COUNT = 48


def recount(report):
    output = {}
    for variant in range(3):
        for name in ("heuristic", "initial", "default"):
            transitions = Counter()
            for stream in report["streams"]:
                before = stream["default"][variant] if name == "default" else report["baselines"][f"{name}-{variant}"]
                after = stream["candidate"][variant]
                compare(before, after)
                transitions.update(f"{SCORES[a['score']]}->{SCORES[b['score']]}" for a, b in zip(before, after))
            output[f"{name}-{variant}"] = dict(transitions)
    return output


def direction(pairs):
    if len(pairs) != 12 or any(a not in (-1, 0, 1, 2) or b not in (-1, 0, 1, 2) for a, b in pairs):
        raise ValueError("invalid rollout scores")
    if any(a == -1 or b == -1 for a, b in pairs):
        return 0, None
    differences = [a - b for a, b in pairs]
    win = sum(value > 0 for value in differences)
    loss = sum(value < 0 for value in differences)
    count = win + loss
    # A descriptive screen across every tested world and continuation. The
    # nominal sign test is not a multiplicity-controlled confirmation.
    p = min(1, 2 * sum(comb(count, k) for k in range(min(win, loss) + 1)) / 2**count)
    sign = (win > loss) - (win < loss) if p <= .05 else 0
    return sign, statistics.mean(differences) / 2


def case_summary(row):
    before = [action[2] for action in row["actions"]]
    after = [action[3] for action in row["actions"]]
    if (not isclose(sum(before), 1) or not isclose(sum(after), 1) or
            not isclose(sum(abs(b - a) for a, b in zip(before, after)) / 2, row["tv"], abs_tol=1e-12)):
        raise ValueError("invalid policy distribution")
    cells = row["cells"]
    worlds = {c["world"] for c in cells}
    keys = [(c["world"], c["response"], c["continuation"]) for c in cells]
    if cells and (len(keys) != len(set(keys)) or len(keys) != len(worlds) * 4):
        raise ValueError("incomplete rollout response cells")
    estimates = [direction(c["pairs"]) for c in cells]
    signs = {sign for sign, _ in estimates}
    robust = signs.pop() if len(signs) == 1 and len(worlds) >= 2 else 0
    means = {(c["world"], c["response"], c["continuation"]): mean
             for c, (_, mean) in zip(cells, estimates)}
    reverses = sum(mean is not None and other is not None and mean * other < 0
                   for (world, response, continuation), mean in means.items() if continuation == 0
                   for other in (means[world, response, 1],))
    return dict(robust_direction=robust, continuation_sign_reversals=reverses,
                alternative_worlds=max(0, len(worlds)-1),
                rollout_pairs=sum(len(cell["pairs"]) for cell in cells))


def summarize(games):
    output = {}
    for variant in range(3):
        groups = [g for g in games if g["variant"] == variant]
        points = [row for g in groups for row in g["rows"] if "actions" in row]
        derived = [case_summary(row) for row in points]
        output[str(variant)] = dict(
            layout_count=len(groups), decision_count=len(points),
            source_games=Counter(str(row["score"]) for g in groups for row in g["rows"] if "game_plies" in row),
            supported_better=sum(c["robust_direction"] == 1 for c in derived),
            supported_worse=sum(c["robust_direction"] == -1 for c in derived),
            unresolved=sum(c["robust_direction"] == 0 for c in derived),
            continuation_sign_reversals=sum(c["continuation_sign_reversals"] for c in derived),
            decisions_with_continuation_reversal=sum(c["continuation_sign_reversals"] > 0 for c in derived),
            decision_samples_with_alternate_world=sum(c["alternative_worlds"] > 0 for c in derived),
            rollout_pairs=sum(c["rollout_pairs"] for c in derived),
            mean_tv=statistics.mean(row["tv"] for row in points) if points else None)
    return output


def audit():
    report = json.loads(DESTINATION.read_text(encoding="utf8"))
    frozen = json.loads(SOURCE.read_text(encoding="utf8"))
    if report["source_sha256"] != sha(SOURCE) or report["transitions"] != recount(frozen):
        raise ValueError("frozen transitions mismatch")
    expected = source_selection(report["source_pool"])
    actual = [(g["variant"], g["seed"], g["seat"]) for g in report["games"]]
    if list(expected) != actual:
        raise ValueError("layout selection differs from frozen source pool")
    for game in report["games"]:
        source_key = (game["variant"], game["seed"], game["seat"])
        observed = [r for r in game["rows"] if "game_plies" in r]
        for generated, recorded in zip(observed, expected[source_key], strict=True):
            score = -1 if recorded["score"] is None else int(2 * recorded["score"])
            if (generated["score"], generated["game_plies"], generated["first"]) != (
                    score, recorded["plies"], recorded["first"]):
                raise ValueError("natural game differs from the source pool")
    if report["decision_summary"] != summarize(report["games"]):
        raise ValueError("decision contrast summaries mismatch")
    print("PASS archived paired transitions and offline decision summaries")


def engine(*arguments):
    result = subprocess.run([str(EXE), *map(str, arguments)], capture_output=True, text=True,
                            check=True, timeout=60, creationflags=0x40 | 0x08000000)
    return [json.loads(line) for line in result.stdout.splitlines()]


def source_selection(pool):
    selected = {}
    for variant in range(3):
        baseline, candidate = pool[str(variant)]
        compare(baseline, candidate)
        if [(r["seed"], r["seat"]) for r in baseline] != [
                (seed, seat) for seed in range(LAYOUT_START, LAYOUT_START + LAYOUT_COUNT)
                for seat in range(4)]:
            raise ValueError("unexpected development source pool")
        used_layouts = set()
        for source in (baseline, candidate):
            for score in (0, .5, 1, None):
                count = 0
                for a, b, row in zip(baseline, candidate, source, strict=True):
                    if row["score"] != score or row["seed"] in used_layouts:
                        continue
                    selected[variant, row["seed"], row["seat"]] = (a, b)
                    used_layouts.add(row["seed"])
                    count += 1
                    if count == 2:
                        break
    return selected


def run():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    frozen = json.loads(SOURCE.read_text(encoding="utf8"))
    parent = frozen["streams"][0]
    path = Path(parent["candidate_path"])
    model = load(path, path.parent / "junqi.exe", path.parent / "ruleset.json", "short-v1")
    started = time.monotonic()
    pool = {str(variant): [
        engine("transfer-probe", RULES, *weights, LAYOUT_START, LAYOUT_COUNT, policy, variant)
        for policy, weights in (("heuristic", [0.0]*5), ("short", model["weights"]))]
        for variant in range(3)}
    games = []
    for variant, seed, seat in source_selection(pool):
        rows = engine("decision-branches", RULES, seed, seat, variant, *model["weights"])
        if [r["source"] for r in rows if "game_plies" in r] != [0, 1]:
            raise ValueError("missing natural trajectories")
        games.append(dict(seed=seed, seat=seat, variant=variant, rows=rows))
    report = dict(source_sha256=sha(SOURCE), executable_sha256=sha(EXE), rules_sha256=sha(RULES),
                  model_sha256=sha(path), training_seed=parent["seed"], transitions=recount(frozen),
                  source_pool=pool, games=games, decision_summary=summarize(games),
                  seconds=time.monotonic()-started)
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("decision review output exceeds budget")
    with DESTINATION.open("xb") as output:
        output.write(data)
    audit()
    print(json.dumps(report["decision_summary"], ensure_ascii=False))


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
