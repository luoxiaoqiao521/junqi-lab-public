"""Verify visibility partitions in the current executable against frozen replays."""
import json

from goal_followthrough import EXE, GRID, REPORT, RULES, collect, model_roster, sha


def audit_current_engine():
    result = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    if parent != result["frozen"]["parent"] or sha(RULES) != result["frozen"]["rules"]:
        raise ValueError("ordinary replay requires the original model and rules")
    for key, expected in zip(GRID, result["rows"], strict=True):
        if collect(EXE, RULES, parent, *key, mode="goal-followthrough") != expected:
            raise ValueError(f"visibility change affected four-dark rows at {key}")
    print(f"PASS native visibility groups and unchanged four-dark results in {len(GRID)} source games")


if __name__ == "__main__":
    audit_current_engine()
