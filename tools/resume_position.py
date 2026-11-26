"""Verify six-dimensional checkpoint continuation against a zeroed-axis control."""
import argparse
import json
from pathlib import Path
import subprocess

from compare_generations import call, sha
from learn_short import EXE, RULES, np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent = np.load(args.parent, allow_pickle=False)
    if parent.shape != (6,) or not np.isfinite(parent).all() or parent[-1] == 0:
        raise ValueError("expected finite six-weight position checkpoint")
    control = parent.copy()
    control[-1] = 0
    resumed = json.loads(call("resume-position", parent, "68000"))
    zeroed = json.loads(call("resume-position", control, "68000"))
    weighted = np.asarray(resumed["weights"])
    if weighted.shape != (6,) or not np.isfinite(weighted).all():
        raise ValueError("invalid resumed weights")
    if np.array_equal(weighted,parent) or np.array_equal(weighted,zeroed["weights"]):
        raise ValueError("resume did not update or silently ignored initial sixth weight")
    malformed = subprocess.run([str(EXE), "resume-position", str(RULES),
                                 *map(str, parent[:5]), "nan", "68000"],
                               capture_output=True, text=True, timeout=10)
    if malformed.returncode == 0 or "invalid weight" not in malformed.stderr:
        raise ValueError("non-finite weight was not rejected")
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.out / "resumed.npy"
    np.save(checkpoint, weighted, allow_pickle=False)
    if not np.array_equal(weighted, np.load(checkpoint, allow_pickle=False)):
        raise ValueError("resumed checkpoint reload mismatch")
    result = dict(seed_range=[68000, 68511], initial_weights=parent.tolist(),
                  resumed=resumed, zeroed_control=zeroed, parent_sha256=sha(args.parent),
                  resumed_sha256=sha(checkpoint), rules_sha256=sha(RULES), exe_sha256=sha(EXE))
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print("Six-weight checkpoint accepted, updated, distinguished from zero-axis control, and reloaded.")


if __name__ == "__main__":
    main()
