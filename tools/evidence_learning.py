"""Bounded development comparison for the optional public-evidence policy."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_transfer import compare, totals
from learn_short import EXE, RULES

TRAIN_SEEDS = (73000, 74000)
EVAL_START = 75000
EVAL_COUNT = 32


def command(*args):
    result = subprocess.run([str(EXE), *map(str, args)], check=True,
                            capture_output=True, text=True, timeout=180)
    return [json.loads(line) for line in result.stdout.splitlines()]


def evaluate(weights, policy):
    return command("transfer-probe", RULES, *weights, EVAL_START, EVAL_COUNT, policy, 0)


def main(path):
    start = time.monotonic()
    heuristic = evaluate([0] * 5, "heuristic")
    results = []
    for seed in TRAIN_SEEDS:
        control, = command("train-openings", RULES, *([0] * 5), seed)
        candidate, = command("train-evidence", RULES, *([0] * 5), seed)
        if len(candidate["weights"]) != 6 or not candidate["weights"][5]:
            raise ValueError("evidence feature received no learned update")
        rows = {
            "control": evaluate(control["weights"], "short"),
            "candidate": evaluate(candidate["weights"], "evidence"),
            "masked": evaluate(candidate["weights"][:5], "short"),
        }
        result = dict(seed=seed, control_training=control, candidate_training=candidate,
                      contact_turns=sum(row["evidence_contacts"] for row in rows["candidate"]),
                      totals={key: totals(value) for key, value in rows.items()},
                      versus_control=compare(rows["control"], rows["candidate"]),
                      versus_masked=compare(rows["masked"], rows["candidate"]),
                      versus_heuristic=compare(heuristic, rows["candidate"]),
                      rows=rows)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    Path(path).write_text(json.dumps(dict(seconds=time.monotonic()-start, results=results,
                                         hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (EXE, RULES)},
                                         heuristic=heuristic, heuristic_totals=totals(heuristic)),
                                     indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main(sys.argv[1])
