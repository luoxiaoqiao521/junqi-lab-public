"""Evaluate the first-retirement phase with the post-retirement policy held fixed."""
import sys

from measure_deadline_rollout import main


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(audit=sys.argv[1:] == ["audit"], mode="first-exit")
