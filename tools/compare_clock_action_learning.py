"""Train a public clock/action interaction and exercise it from natural openings."""
from collections import Counter
import json
import math
import os
from statistics import mean
import subprocess
import sys

from measure_deadline_opening import ROOT, RULES, build, model_roster, require
from evidence_model import sha


TRAIN = 1091700
VALIDATION = 1091800
COUNT = 48
ENCODER = "short-v1+quiet-clock-contact-v1"
OUTCOMES = {-1: "unresolved", 0: "loss", 1: "draw", 2: "win"}


def run(directory, roster, start, coefficient=None, contact_control=False):
    mode = ("clock-action-pairs" if coefficient is None else
            "clock-contact-control" if contact_control else "clock-action-apply")
    weights = [x for n in ("candidate", "historical") for x in roster[n]["weights"]]
    args = [start, COUNT, *weights]
    if coefficient is not None:
        args.append(coefficient)
    process = subprocess.run([directory / "deadline_opening.exe", RULES, mode],
                             input=" ".join(map(str, args)), capture_output=True, text=True,
                             encoding="utf8", errors="replace", timeout=60)
    require(process.returncode == 0, process.stderr)
    rows = []
    for line in process.stdout.splitlines():
        columns = line.split()
        seed, seat, table, ply, clock = map(int, columns[:5])
        delta, interaction = map(float, columns[5:7])
        tail = list(map(int, columns[7:]))
        require(len(tail) == (8 if coefficient is None else 5), "malformed clock row")
        require((ply < 0 and clock == interaction == 0) or
                (60 <= clock < 70 and abs(interaction) == clock / 70), "invalid clock observation")
        rows.append(((seed, seat, table), ply, clock, delta, interaction, tail))
    grid = [(s,j,t) for s in range(start,start+COUNT) for t in range(2) for j in range(4)]
    require([r[0] for r in rows] == grid, "incomplete opening matrix")
    return rows


def train(rows, contact_control=False):
    labels = [(delta,math.copysign(1.,f) if contact_control else f,.5 + (tail[4] - tail[0]) / 4)
              for _,ply,_,delta,f,tail in rows if ply >= 0 and tail[0] >= 0 and tail[4] >= 0]
    require(labels and any(y!=.5 for _,_,y in labels), "clock comparison has no informative team outcomes")
    coefficient = 0.
    for _ in range(300):
        grads = []
        for delta, feature, target in labels:
            advantage = max(-32, min(32, delta + coefficient * feature))
            prob = 1/(1 + math.exp(-advantage))
            grads.append((target - prob) * feature)
        coefficient += .35 * (mean(grads) - .02 * coefficient)
    return coefficient, len(labels)


def model_value(artifact, roster):
    require(artifact["encoding"] == ENCODER and artifact["rules"] == sha(RULES)
            and artifact["parent"] == roster["candidate"]["artifact_sha256"], "incompatible clock model")
    weight = artifact["coefficient"]
    require(type(weight) is float and math.isfinite(weight), "invalid coefficient")
    return weight


def verify_application(paired, applied, coefficient, contact_control=False):
    for row, actual in zip(paired, applied, strict=True):
        _,ply,_,delta,feature,options = row
        value = math.copysign(1., feature) if contact_control else feature
        switched = ply >= 0 and delta + coefficient * value > 0
        expected = options[4:] if switched else options[:4]
        require(row[:-1] == actual[:-1] and actual[-1][0] == switched and actual[-1][1:] == expected,
                "applied move changed the root or frozen continuation")


def metrics(rows):
    scores = [r[-1][1] for r in rows if r[-1][1] >= 0]
    return {"known_mean": mean(scores)/2 if scores else None,
            "result": dict(Counter(OUTCOMES[r[-1][1]] for r in rows)),
            "wins_by_140": sum(r[-1][1]==2 and r[-1][2]<=140 for r in rows),
            "loss_by_140": sum(r[-1][1]==0 and r[-1][2]<=140 for r in rows),
            "complete_mean_plies": mean(r[-1][2] for r in rows) if len(scores)==len(rows) else None}


def summarize(base, learned):
    report = {}
    for table in (0,1):
        a,b = ([r for r in panel if r[0][2]==table] for panel in (base,learned))
        pairs = list(zip(a,b,strict=True))
        changed = [(x,y) for x,y in pairs if x[-1][0]!=y[-1][0]]
        known = [(x[-1][1],y[-1][1]) for x,y in changed if min(x[-1][1],y[-1][1])>=0]
        report[str(table)] = {
            "roots": len(a), "eligible": sum(r[1]>=0 for r in a), "different_choices": len(changed),
            "families_different": len({x[0][0] for x,_ in changed}), "unknown_changes": len(changed)-len(known),
            "pair_transitions": dict(Counter(f"{OUTCOMES[x]}/{OUTCOMES[y]}" for x,y in known)),
            "before": metrics(a), "after": metrics(b)}
    return report


def main(audit):
    root = ROOT / "build/deadline-opening" / os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        root.mkdir(parents=True, exist_ok=True)
        build(root)
    roster = model_roster()
    training = run(root, roster, TRAIN)
    beta, cases = train(training)
    contact_weight, _ = train(training, contact_control=True)
    artifact = {"encoding": ENCODER, "coefficient": beta, "rules": sha(RULES),
                "parent": roster["candidate"]["artifact_sha256"], "paired_known_train": cases}
    model_wire = json.dumps(artifact, sort_keys=True, allow_nan=False) + "\n"
    loaded = json.loads(model_wire)
    weight = model_value(loaded, roster)
    for incompatible in ("position-v1", "public-engineer-v1", "short-v1"):
        try:
            model_value(dict(loaded, encoding=incompatible), roster)
        except ValueError:
            continue
        raise ValueError("incompatible model was accepted")
    paired = run(root, roster, VALIDATION)
    base = run(root, roster, VALIDATION, 0.)
    learned = run(root, roster, VALIDATION, weight)
    contact = run(root, roster, VALIDATION, contact_weight, contact_control=True)
    verify_application(paired, base, 0.)
    verify_application(paired, learned, weight)
    verify_application(paired, contact, contact_weight, contact_control=True)
    result = {"model": artifact, "contact_control_coefficient": contact_weight,
              "tables": summarize(base, learned), "clock_vs_contact_control": summarize(contact, learned),
              "validation": learned}
    for name, wire in (("model.json", model_wire), ("summary.json", json.dumps(result, sort_keys=True) + "\n")):
        path = root / name
        if audit:
            require(path.read_bytes() == wire.encode(), f"{name} did not reproduce")
        else:
            pending = path.with_suffix(".tmp")
            pending.write_bytes(wire.encode())
            pending.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k!="validation"}, sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
