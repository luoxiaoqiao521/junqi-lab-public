"""Falsify fixed continuations while protecting indistinguishable hidden worlds."""
import json
import os
import subprocess
import sys

from check_public_reply_representation import result as previous_probe
from measure_deadline_opening import ROOT, RULES, build, require


PREVIOUS = (ROOT / "build/deadline-opening/compare-reply-qualified-route-d8e57fb2ce45a443cb97"
            / "summary.json")


def calculate(directory):
    run = subprocess.run([directory / "deadline_opening.exe", RULES, "conditional-reply-proof"],
                         capture_output=True, encoding="utf8", errors="replace", timeout=45)
    require(run.returncode == 0, f"conditional proof failed: {run.stderr}")
    rows = {}
    fields = ("common_finish", "observation_conditioned_finish", "public_branches", "unresolved")
    for line in run.stdout.splitlines():
        name, *values = line.split()
        rows[name] = dict(zip(fields, map(int, values), strict=True))
    require(set(rows) == {"different_public_blocks", "identical_view_hidden_flag_swap"},
            "conditional proof matrix incomplete")
    good = rows["different_public_blocks"]
    dark = rows["identical_view_hidden_flag_swap"]
    require(good["observation_conditioned_finish"] == 1 and good["common_finish"] == 0
            and good["unresolved"] == 0, "different visible answers did not allow different continuations")
    require(dark["observation_conditioned_finish"] == dark["common_finish"] == 0 and dark["unresolved"] > 0,
            "hidden truth was used to select a branch")
    require(previous_probe(directory) == json.loads(PREVIOUS.read_text(encoding="utf8")),
            "previous frozen public representation changed")
    return {"evidence": rows, "preserves_previous_diagnostic": True}


def main(audit):
    directory = ROOT / "build/comeback" / os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True, exist_ok=True)
        build(directory)
    wire = json.dumps(calculate(directory), sort_keys=True, indent=2) + "\n"
    receipt = directory / "summary.json"
    if audit:
        require(receipt.read_bytes() == wire.encode(), "conditional-reply proof did not reproduce")
    else:
        pending = receipt.with_suffix(".tmp")
        pending.write_bytes(wire.encode())
        pending.replace(receipt)
    print(wire)
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
