"""Bounded continuation experiment with persisted parent and child artifacts."""
import copy
import json
import shutil
from pathlib import Path
import sys
import time

from evidence_learning import command
from evidence_model import ENCODING, load, save, sha, validate
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES

SEEDS = (76000, 78000)
EVAL_START = 81000
EVAL_COUNT = 64


def evaluate(weights, policy):
    return command("transfer-probe", RULES, *weights, EVAL_START, EVAL_COUNT, policy, 0)


def stable(rows):
    return [{key: value for key, value in row.items() if key != "learner_ns"} for row in rows]


def audit(path):
    report = json.loads(Path(path).read_text(encoding="utf8"))
    executable, rules = report["executable_path"], report["rules_path"]
    if report["hashes"] != dict(engine=sha(executable), rules=sha(rules)):
        raise ValueError("experiment provenance mismatch")
    if [stream["seed"] for stream in report["streams"]] != list(SEEDS):
        raise ValueError("unexpected training streams")
    if report["rejected_invalid_models"] != 5:
        raise ValueError("model rejection checks incomplete")
    initial = Path(report["artifact_dir"]) / "initial.json"
    load(initial, executable, rules)
    for stream in report["streams"]:
        parent = load(stream["parent_path"], executable, rules)
        child = load(stream["child_path"], executable, rules)
        if parent["parent_sha256"] != sha(initial):
            raise ValueError("parent did not name the persisted initialization")
        if child["parent_sha256"] != sha(stream["parent_path"]):
            raise ValueError("child did not name its persisted parent")
        if parent["weights"] != stream["parent_training"]["weights"]:
            raise ValueError("persisted parent differs from training")
        if child["weights"] != stream["child_training"]["weights"]:
            raise ValueError("persisted child differs from training")
        rows = stream["rows"]
        expected = [(seed, seat) for seed in range(EVAL_START, EVAL_START + EVAL_COUNT)
                    for seat in range(4)]
        for group in rows.values():
            if [(row["seed"], row["seat"]) for row in group] != expected:
                raise ValueError("unexpected evaluation blocks")
        for key, baseline in (("versus_parent", "parent"), ("versus_control", "control"),
                              ("versus_masked", "masked")):
            if stream[key] != compare(rows[baseline], rows["child"]):
                raise ValueError("comparison does not match raw results")
        if stream["totals"] != {key: totals(value) for key, value in rows.items()}:
            raise ValueError("reported totals differ")
        if stable(rows["child"]) != stream["reload_rows"]:
            raise ValueError("reloaded policy outcomes or decision summaries changed")
    print("PASS persisted parents, encodings, reloads and paired comparisons")


def main(path):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    artifact_dir = destination.parent.parent.parent / "build" / ("evidence-models-" + str(time.time_ns()))
    artifact_dir.mkdir(parents=True)
    archived_executable = artifact_dir / "junqi.exe"
    archived_rules = artifact_dir / "ruleset.json"
    shutil.copyfile(EXE, archived_executable)
    shutil.copyfile(RULES, archived_rules)
    initial_path = artifact_dir / "initial.json"
    initial = save(initial_path, [0.0] * 6)
    rejected = 0
    for field, value in (("encoding", "position-v1"), ("rules_sha256", "wrong"),
                         ("engine_sha256", "wrong"), ("weights", [0.0] * 5),
                         ("weights", [float("nan")] * 6)):
        invalid = copy.deepcopy(initial)
        invalid[field] = value
        try:
            validate(invalid)
        except ValueError:
            rejected += 1
        else:
            raise ValueError("invalid model accepted")
    started = time.monotonic()
    streams = []
    for seed in SEEDS:
        parent_training, = command("resume-evidence", RULES, *load(initial_path)["weights"], seed)
        parent_path = artifact_dir / f"parent-{seed}.json"
        parent = save(parent_path, parent_training["weights"], sha(initial_path))
        child_training, = command("resume-evidence", RULES, *load(parent_path)["weights"], seed + 512)
        child_path = artifact_dir / f"child-{seed}.json"
        child = save(child_path, child_training["weights"], sha(parent_path))
        first, = command("train-openings", RULES, *([0] * 5), seed)
        control, = command("train-openings", RULES, *first["weights"], seed + 512)
        rows = dict(parent=evaluate(parent["weights"], "evidence"),
                    child=evaluate(child["weights"], "evidence"),
                    control=evaluate(control["weights"], "short"),
                    masked=evaluate(child["weights"][:5], "short"))
        reloaded = evaluate(load(child_path)["weights"], "evidence")
        if stable(rows["child"]) != stable(reloaded):
            raise ValueError("reload changed outcome or decisions")
        stream = dict(seed=seed, parent_path=str(parent_path), child_path=str(child_path),
                      parent_training=parent_training, child_training=child_training,
                      control_training=[first, control], rows=rows, reload_rows=stable(reloaded),
                      totals={key: totals(value) for key, value in rows.items()},
                      versus_parent=compare(rows["parent"], rows["child"]),
                      versus_control=compare(rows["control"], rows["child"]),
                      versus_masked=compare(rows["masked"], rows["child"]))
        streams.append(stream)
        print(json.dumps({k: v for k, v in stream.items()
                          if k in ("seed", "totals", "versus_parent", "versus_control", "versus_masked")}),
              flush=True)
    report = dict(encoding=ENCODING, rejected_invalid_models=rejected,
                  artifact_dir=str(artifact_dir), executable_path=str(archived_executable),
                  rules_path=str(archived_rules), streams=streams,
                  seconds=time.monotonic()-started, hashes=dict(engine=sha(EXE), rules=sha(RULES)))
    with destination.open("x", encoding="utf8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write(chr(10))
    audit(path)


if __name__ == "__main__":
    if sys.argv[1] == "audit":
        audit(sys.argv[2])
    else:
        main(sys.argv[1])
