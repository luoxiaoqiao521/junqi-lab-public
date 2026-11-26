"""Measure useful reply coverage when exactly one rival policy changes."""
from collections import Counter
import json
import os
import sys

from compare_replay_first_followthrough import ROOT, ARMS, compile_native, evidence, games, report, require, write_result


CONTROL = ROOT/"build/replay-curriculum/compare-replay-first-and-followthrough-ca13c1f1c0bde56f4f6e/summary.json"


def coverage(results, table):
    groups = [(k,a) for k,a in results.items() if k[2]==table]
    phases = {}
    for at in (48,96,144):
        items = [(k,a) for k,a in groups if k[1]==at]
        phases[at] = {"first_enemy_retired_80":sum(0<r["enemy_exit"]<=80 for _,a in items for r in a.values()),
                      "first_friendly_retired_80":sum(0<r["friendly_exit"]<=80 for _,a in items for r in a.values())}
    return dict(by_phase=phases,
        early_exit_families=len({k[0] for k,a in groups if any(0<r["enemy_exit"]<=80 or
                                  0<r["friendly_exit"]<=80 for r in a.values())}),
        opposite_families=len({k[0] for k,a in groups if {0,2}<={r["outcome"] for r in a.values()}}))


def matched(results):
    changes = {}
    for arm in ARMS:
        counts = Counter()
        for (family,ply,table,rep), arms in results.items():
            if table!=1:
                continue
            a,b = arms[arm],results[family,ply,2,rep][arm]
            require(a["first"]==b["first"],"pressure changed the first decision")
            counts["reply_changed"] += a["reply"]!=b["reply"]
            counts["loss_to_nonloss"] += a["outcome"]==0 and b["outcome"] in (1,2)
            counts["nonloss_to_loss"] += a["outcome"] in (1,2) and b["outcome"]==0
            counts["different_end"] += a["outcome"]!=b["outcome"]
        changes[str(arm)] = dict(counts)
    return changes


def main(audit):
    directory=ROOT/"build/replay-curriculum"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        compile_native(directory)
    results,provenance=evidence(directory,"curriculum-pressure",(1,2))
    old=json.loads(CONTROL.read_bytes())
    control=[g for g in old["games"] if g[0][2]==1]
    require([g for g in games(results) if g[0][2]==1]==control and provenance==old["provenance"],
            "historical comparator did not reproduce")
    counts={t:coverage(results,t) for t in (1,2)}
    metrics=dict(coverage=counts,matched=matched(results),panels=report(results,(1,2)))
    write_result(directory,dict(provenance=provenance,metrics=metrics,games=games(results)),audit)
    print(json.dumps(metrics,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
