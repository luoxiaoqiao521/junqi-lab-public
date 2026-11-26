"""Audit the maximum probability effect of the binary public-engineer feature."""
import json
import math
from pathlib import Path

from evidence_model import load, sha

SOURCE = Path("docs/REPORTS/evidence-model-contract.json")
DESTINATION = Path("docs/REPORTS/evidence-probability-bound.json")


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    report = json.loads(SOURCE.read_text(encoding="utf8"))
    rows = []
    for stream in report["streams"]:
        for generation in ("parent", "child"):
            path = stream[generation + "_path"]
            coefficient = load(path, report["executable_path"], report["rules_path"])["weights"][5]
            # A binary feature tilts one action group by exp(coefficient).
            # Max_p |exp(w)*p/(1-p+exp(w)*p)-p| = tanh(abs(w)/4).
            bound = math.tanh(abs(coefficient) / 4)
            rows.append(dict(seed=stream["seed"], generation=generation,
                             artifact_sha256=sha(path), coefficient=coefficient,
                             maximum_total_variation=bound))
    result = dict(source_sha256=sha(SOURCE), scope="same visible decision, all other weights fixed",
                  formula="tanh(abs(w6)/4)", rows=rows,
                  limitations=["Upper bound, not observed action-change rate",
                               "Does not bound accumulated team-score changes across a game",
                               "Requires the current binary sixth feature and linear softmax"])
    with DESTINATION.open("x", encoding="utf8") as output:
        json.dump(result, output, indent=2)
        output.write(chr(10))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
