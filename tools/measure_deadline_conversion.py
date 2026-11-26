"""Run the matched first-enemy-retirement continuation experiment."""
import sys

from measure_deadline_opening import main


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(audit=sys.argv[1:] == ["audit"], conversion=True)
