"""Rank the three proposed experiments using their frozen source receipts."""
import hashlib
import json
from pathlib import Path
import sys
import time

REPORTS = Path(__file__).resolve().parents[1]/"docs/REPORTS"
OUTPUT = REPORTS/"rank-independent-experiment-families.json"
SOURCES = ("public-flag-guard-transfer", "feedback-coverage",
           "feedback-training-mixture", "tuned-policy-confirmation")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evidence():
    inputs = {name: read(REPORTS/f"{name}.json") for name in SOURCES}
    tactics, feedback, mixture, update = (inputs[name] for name in SOURCES)
    guard = tactics["summary"]["contexts"]
    changes = {key: guard[key]["changed_games"] for key in ("0", "1")}
    opportunities = {key: guard[key]["control"]["opportunities"] for key in ("0", "1")}
    decisive = {name: sum(row["unequal_pairs"] for row in feedback["summary"] if row["context"][2] == name)
                for name in ("heuristic", "contact")}
    mix = [row["comparisons"]["random"]["mixed_vs_control"] for row in mixture["summary"]]
    update_effect = update["aggregate"]["default-0"]
    families = {
        "tactical-representation": dict(observed=opportunities, switched=changes,
            whole_game={key: guard[key]["comparison"] for key in ("0", "1")},
            seconds=tactics["seconds"], admissible=tactics["summary"]["next"] == "replicate-urgent-guard"),
        "practice-distribution": dict(unequal_pairs=decisive, mixed_vs_control=mix,
            feedback_seconds=feedback["seconds"], training_and_evaluation_seconds=mixture["seconds"],
            admissible=len(mix) >= 2 and all(c["exploratory_block_interval"][0] > 0 for c in mix)),
        "whole-game-update": dict(from_zero_rate_effect=update_effect, independent_streams=len(update["streams"]),
            seconds=update["seconds"], admissible=len(update["streams"]) >= 4 and update_effect["crossed_interval"][0] > 0),
    }
    eligible = [name for name, family in families.items() if family["admissible"]]
    return dict(families=families, selected=eligible[0] if len(eligible) == 1 else None,
                source_digests={name: digest(obj) for name, obj in inputs.items()})


def audit():
    document = read(OUTPUT)
    if document["evidence"] != evidence():
        raise ValueError("experiment ranking no longer matches the source receipts")
    return document


def main():
    if sys.argv[1:] == ["audit"]:
        document = audit()
    else:
        started = time.monotonic()
        measured = evidence()
        document = dict(evidence=measured, seconds=time.monotonic()-started)
        with OUTPUT.open("x", encoding="utf8") as output:
            output.write(json.dumps(document, indent=2)+"\n")
    print(json.dumps(dict(selected=document["evidence"]["selected"])))


if __name__ == "__main__":
    main()
