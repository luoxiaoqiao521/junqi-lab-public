"""Falsify a rollout budget rule and verify exact reference controls."""
import argparse
from itertools import combinations
import json
from math import ceil, log
from pathlib import Path
import subprocess
import time

from calibrate_reply_evaluator import direction, stable_signal
from measure_deadline_opening import ROOT, RULES, build, file_sha, require

PRIOR = ROOT / "build/tactic-discovery/evaluator-calibration-final/results.json"


def audit_early_stop(path):
    data = json.loads(path.read_text(encoding="utf8"))
    rows = []
    for root in data["roots"]:
        at = root["ply"]+9
        for item in root["calibration"]:
            trials = item["repeats"]
            require(len(trials) == len({t["seed"] for t in trials}) == 8, "expected eight distinct tail streams")
            results = [[direction(t[a],t[b]) for t in trials] for a,b in ("AB", "CD")]
            signals = [any(series[i] in ("damage", "unresolved") for series in results) for i in range(8)]
            costs = [sum(t[k]["plies"]-at for k in "ABCD") for t in trials]
            rows.append(dict(source=root["seed"], cutoff=at, world=item["world"], challenger=item["challenger"],
                             costs=costs, signals=signals, supported=any(map(stable_signal,results))))
    runs = []
    for first in combinations(range(8),4):
        deferred = [r for r in rows if not any(r["signals"][i] for i in first)]
        saved = sum(r["costs"][i] for r in deferred for i in range(8) if i not in first)
        missed = [{k:r[k] for k in ("source","cutoff","world","challenger")} for r in deferred if r["supported"]]
        runs.append(dict(first_streams=[i+1 for i in first], deferred=len(deferred), saved_plies=saved, missed=missed))
    gate = data["summary"]["all"]["gate"]
    require((runs[0]["deferred"],runs[0]["saved_plies"],len(runs[0]["missed"])) ==
            (gate["stop"],gate["deferred_tail_plies"],gate["would_miss_stable"]), "original stopping rule failed to replay")
    total = sum(sum(r["costs"]) for r in rows)
    require(total == gate["all_tail_plies"], "cost unit changed")
    return dict(cutoffs=len(rows), source_layouts=len({r["source"] for r in rows}),
                local_signals=sum(r["supported"] for r in rows), total_tail_plies=total,
                replay_schedules=len(runs), schedules_with_miss=[r for r in runs if r["missed"]],
                saved_plies_range=[min(r["saved_plies"] for r in runs),max(r["saved_plies"] for r in runs)],
                original=runs[0], schedules=runs)


def parse_proofs(stdout):
    proofs = {}
    for line in stdout.splitlines():
        key, depth, lo, hi, nodes = line.split()
        require(key not in proofs and key.startswith(("E1_","E2_")), "unexpected or duplicate proof")
        proofs[key] = dict(depth=int(depth), lower=int(lo)/2, upper=int(hi)/2, expanded=int(nodes))
    require(len(proofs) == 12, "incomplete exact and unresolved controls")
    return proofs


def failure_controls():
    # The later action must also be shared across worlds until new information arrives.
    concealed = [[int(world == guess) for guess in range(3)] for world in range(3)]
    informed = sum(map(max, concealed))/3
    uninformed = max(sum(per_action)/3 for per_action in zip(*concealed, strict=True))
    require(informed > .5 > uninformed, "counterexample did not reverse the action preference")
    return dict(hidden_followup=dict(omniscient_attack=informed, legal_attack=uninformed, safe=.5),
                iid_binomial=dict(zero_in_4_upper_95=1-.05**.25, miss_p10_in_4=.9**4,
                                  independent_zero_events_needed={str(p):ceil(log(.05)/log(1-p)) for p in (.1,.05,.01)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=PRIOR)
    args = parser.parse_args()
    early_stop = audit_early_stop(args.prior)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    build(output)
    engine = output / "deadline_opening.exe"
    raw = [subprocess.run([engine, RULES, "prove-evaluator-controls"], cwd=ROOT, encoding="utf8",
                          capture_output=True, timeout=45, check=True).stdout for _ in range(2)]
    require(raw[0] == raw[1], "reference controls are not deterministic")
    proofs = parse_proofs(raw[0])
    take = [proofs[f"E2_{world}_take"]["lower"] for world in ("weak","strong")]
    wait = [proofs[f"E2_{world}_quiet"]["lower"] for world in ("weak","strong")]
    weights = [0,.25,.5,.75,1]
    matrix = [dict(weight_weak=p, take=p*take[0]+(1-p)*take[1], wait=p*wait[0]+(1-p)*wait[1],
                   omniscient=p*max(take[0],wait[0])+(1-p)*max(take[1],wait[1])) for p in weights]
    receipt = dict(engine=file_sha(engine), rules=file_sha(RULES), prior=file_sha(args.prior),
                   elapsed_seconds=round(time.monotonic()-started,2), proofs=proofs, e2=matrix,
                   counterexamples=failure_controls(), early_stop=early_stop)
    (output / "results.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf8")
    early_stop = {k:v for k,v in early_stop.items() if k != "schedules"}
    print(json.dumps(dict(proofs=proofs, early_stop=early_stop),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
