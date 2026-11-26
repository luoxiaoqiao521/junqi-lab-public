"""Evaluate prioritizing the first retirement, then using the remaining routes."""
import sys

from compare_deadline_two_targets import main


if __name__ == "__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(audit=sys.argv[1:]==["audit"],staged=True)
