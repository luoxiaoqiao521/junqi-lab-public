"""Compare frozen short-v1 ties with the same public reply evidence."""
import json
import os
import subprocess
import sys

from measure_deadline_opening import ROOT, RULES, build, model_roster, require


EXPECTED = {
    "dark_flag_counter_on_far_wing": 1,
    "dark_flag_counter_on_near_wing": 0,
    "decoy_same_observation": -1,
    "decoy_hidden_swap": -1,
    "probe_and_keep_finisher": 1,
}


def result(directory):
    roster = model_roster()
    weights = [v for name in ("candidate", "historical") for v in roster[name]["weights"]]
    run = subprocess.run([directory / "deadline_opening.exe", RULES, "public-reply-proof"],
                         input=" ".join(map(str, (1, 1, *weights))), capture_output=True,
                         encoding="utf8", errors="replace", timeout=45)
    require(run.returncode == 0, f"public decision probe failed: {run.stderr}")
    rows = {}
    for line in run.stdout.splitlines():
        name, *values = line.split()
        choice, flags, *witnesses = map(int, values)
        require(name not in rows and len(witnesses) == 4 and set(witnesses) <= {0, 1},
                "malformed or repeated proof")
        rows[name] = {"short_v1_tied": True, "selected": choice if choice >= 0 else None,
                      "near_far_flag_mask": flags, "after_reply": [
                          {"sample_common_finish": bool(witnesses[i]),
                           "sample_common_refutation": bool(witnesses[i + 1])} for i in (0, 2)]}
    require(set(rows) == set(EXPECTED) and all(rows[k]["selected"] ==
            (v if v >= 0 else None) for k, v in EXPECTED.items()), "wrong selection or incomplete matrix")
    require(rows["decoy_same_observation"] == rows["decoy_hidden_swap"],
            "replacing only the referee's hidden identities changed the public decision")
    return {"source": "controlled legal positions", "samples_per_root": 8, "rows": rows,
            "frozen_research_model": roster["candidate"]["artifact_sha256"]}


def main(audit):
    directory = ROOT / "build/deadline-opening" / os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True, exist_ok=True)
        build(directory)
    wire = json.dumps(result(directory), sort_keys=True, indent=2) + "\n"
    receipt = directory / "summary.json"
    if audit:
        require(receipt.read_bytes() == wire.encode(), "ranking or public isolation failed to reproduce")
    else:
        temporary = receipt.with_suffix(".tmp")
        temporary.write_bytes(wire.encode())
        temporary.replace(receipt)
    print(wire)
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
