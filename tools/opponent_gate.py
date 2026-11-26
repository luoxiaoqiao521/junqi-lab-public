"""Run a frozen opponent-only contrast and independently audit its admission gate."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import tempfile
import time

from compare_generations import sha
from learn_short import EXE, RULES
from pressure_ranking import probe, sign

SEEDS = range(70000, 70016)


def gate(cases, condition="hq"):
    strata = defaultdict(set)
    total = defaultdict(int)
    for case in cases:
        cells = {(cell, profile, (source,target)): counts
                 for cell,profile,source,target,*counts in case["cells"]}
        for pair in case["comparisons"]:
            q, a = tuple(pair["quiet"]), tuple(pair["attack"])
            if condition == "movement":
                group = "seen_moving" if pair["target_moved"] else "no_move_observed"
            else:
                group = "enemy_hq" if pair["attack_hq"] else "other_target"
            total[group] += 1
            directions = [[sign(cells[2*w+offset,p,a], cells[2*w+offset,p,q])
                           for offset in (0,1)]
                          for w,p in product(range(case["worlds"]),range(2))]
            if directions != [list(d) for d in pair["directions"]]:
                raise ValueError("recorded directions differ from outcomes")
            for reversal in ([-1,1],[1,-1]):
                if case["worlds"] >= 2 and all(d == reversal for d in directions):
                    strata[f"{group}:{reversal[0]}->{reversal[1]}"].add(case["source_group"])
    supported = {key: sorted(value) for key,value in strata.items()}
    return dict(pairs_by_public_group=dict(total), supporting_source_groups=supported,
                advance_to_feature_test=any(len(sources)>=2 for sources in supported.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--handoff", action="store_true")
    parser.add_argument("--condition", choices=("hq","movement"), default="hq")
    args = parser.parse_args()
    movement = args.condition == "movement"
    seeds = range(71000,71024) if movement else SEEDS
    if args.mode == "run":
        if args.analysis.exists():
            raise FileExistsError(args.analysis)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="junqi-opponent-") as directory:
            cases = [probe(seed,Path(directory),"pressure-moved" if movement else "pressure-opponents")
                     for seed in seeds]
        report = dict(seed_range=[seeds.start,seeds.stop-1], condition=args.condition, role="development-diagnostic",
                      eligible_as_training_label=False, profiles=["all-random","opponents-heuristic"],
                      cases=[case for case in cases if case], exe_sha256=sha(EXE),
                      rules_sha256=sha(RULES), seconds=time.monotonic()-started)
        with args.analysis.open("x",encoding="utf8") as out:
            json.dump(report,out,indent=2)
            out.write("\n")
        print(f"Opponent contrast stored: {len(report['cases'])} matched prefixes.")
        if args.handoff:
            print("\n```agent-autoloop\n" + json.dumps({"events":[{
                "id":"junqi-opponent-sensitivity:"+sha(args.analysis),
                "type":"junqi.movement.audit.ready" if movement else "junqi.opponent.audit.ready",
                "subject":"junqi-movement" if movement else "junqi-opponent",
                "summary":"Audit frozen opponent contrast","detail":sha(args.analysis)
            }]}) + "\n```")
    else:
        report = json.loads(args.analysis.read_text(encoding="utf8"))
        if report["exe_sha256"] != sha(EXE) or report["rules_sha256"] != sha(RULES):
            raise ValueError("executable or rules changed before gate audit")
        if report.get("condition","hq") != args.condition:
            raise ValueError("analysis condition mismatch")
        decision = gate(report["cases"],args.condition)
        decision.update(analysis_sha256=sha(args.analysis), matched_prefixes=len(report["cases"]))
        with args.decision.open("x",encoding="utf8") as out:
            json.dump(decision,out,indent=2)
            out.write("\n")
        print(f"Gate verdict: advance={decision['advance_to_feature_test']}; "
              f"pairs={decision['pairs_by_public_group']}")


if __name__ == "__main__":
    main()
