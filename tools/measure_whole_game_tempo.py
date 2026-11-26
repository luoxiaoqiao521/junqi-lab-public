"""Measure team score and tempo from paired full selfplay openings."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean, median
import sys

from archive_pressure_training import RULES
from compare_archive_cycle_exit_values import interval
from partner_crossplay import evaluate, model_roster
from pilot_relay_candidate_budget import file_sha

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build/junqi.exe"
TABLES = ("random", "historical")
START = 964200
COUNT = 40
THRESHOLD = 140
HORIZONS = (70, THRESHOLD, 280, 560)
LABEL = {None:"unknown", 0:"loss", .5:"draw", 1:"win"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(roster, arm, table):
    observed = evaluate(ENGINE,RULES,roster,arm,table,table,START,COUNT)
    expected = [(seed,seat) for seed in range(START,START+COUNT) for seat in range(4)]
    require([(r["seed"],r["seat"]) for r in observed]==expected,"incomplete layout/seat grid")
    for row in observed:
        require(row["score"] in LABEL and 1<=row["plies"]<=3000,"invalid terminal score or plies")
        del row["learner_ns"]
    return observed


def measures(games):
    terminal = [r["plies"] for r in games if r["score"] is not None]
    wins = [r["plies"] for r in games if r["score"]==1]
    return dict(
        n=len(games), outcomes=dict(Counter(LABEL[r["score"]] for r in games)),
        mean_observed_plies=mean(r["plies"] for r in games),
        mean_terminal_plies=mean(terminal) if terminal else None,
        median_win_plies=median(wins) if wins else None,
        max_observed_plies=max(r["plies"] for r in games),
        observed_by_horizon={str(h):dict(Counter(LABEL[r["score"]] if r["plies"]<=h else "unresolved"
                                               for r in games)) for h in HORIZONS},
        cap_cutoffs=sum(r["score"] is None and r["plies"]==3000 for r in games),
        rule_undefined=sum(r["score"] is None and r["plies"]!=3000 for r in games),
    )


def compare(a,b):
    groups=defaultdict(list)
    for old,new in zip(a,b,strict=True):
        require(all(old[k]==new[k] for k in ("seed","seat","first")),"layout or first actor changed")
        sa,sb=old["score"],new["score"]
        low=(sb if sb is not None else 0)-(sa if sa is not None else 1)
        high=(sb if sb is not None else 1)-(sa if sa is not None else 0)
        groups[old["seed"]].append((low,high,new["plies"]-old["plies"]))
    low,high,delta=map(list,zip(*(tuple(map(mean,zip(*block))) for block in groups.values())))
    old_summary,new_summary=map(measures,(a,b))
    w_a,w_b=[panel["observed_by_horizon"][str(THRESHOLD)].get("win",0) for panel in (old_summary,new_summary)]
    score_range=[interval(low)[0],interval(high)[1]]
    accepted=(not new_summary["outcomes"].get("unknown") and
              new_summary["mean_terminal_plies"] is not None and
              new_summary["mean_terminal_plies"]<=THRESHOLD and score_range[0]>=0 and w_b>w_a)
    return dict(frozen=old_summary,experiment=new_summary, paired_score_bounds=[mean(low),mean(high)],
                paired_score_outer_interval=score_range,observed_plies_delta=mean(delta),
                observed_plies_delta_interval=interval(delta),wins_by_140_delta=w_b-w_a,target_pass=accepted,
                transitions=dict(Counter(f"{LABEL[x['score']]}->{LABEL[y['score']]}" for x,y in zip(a,b))),
                new_losses=sum(x["score"] in (.5,1) and y["score"]==0 for x,y in zip(a,b)),
                both_win_speed=dict(Counter("earlier" if y["plies"]<x["plies"] else
                    "later" if y["plies"]>x["plies"] else "equal" for x,y in zip(a,b)
                    if x["score"]==y["score"]==1)))


def run():
    roster=model_roster()
    weights=list(roster["candidate"]["weights"])
    require(len(weights)==5 and weights[0]<0,"unexpected input encoding or contact coefficient")
    modified=list(weights)
    modified[0]*=.25
    roster["tempo_contact_experiment"]=dict(policy="short",weights=modified)
    fingerprint=dict(engine_sha256=file_sha(ENGINE),rules_sha256=file_sha(RULES),
                     frozen_sha256=roster["candidate"]["artifact_sha256"],
                     historical_sha256=roster["historical"]["artifact_sha256"],
                     original=weights, experiment=modified,start=START,count=COUNT,horizons=list(HORIZONS))
    data={}
    for table in TABLES:
        original=rows(roster,"candidate",table)
        study=rows(roster,"tempo_contact_experiment",table)
        data[table]=dict(report=compare(original,study),frozen=original,experiment=study)
    return dict(inputs=fingerprint,tables=data)


def main(audit):
    key=os.environ.get("JUNQI_SERIAL_KEY","manual")
    require(key and all(ch.isalnum() or ch=="-" for ch in key),"invalid serial key")
    out=ROOT / "build/whole-game-tempo" / key
    evidence=out / "summary.json"
    data=run()
    if audit:
        require(json.loads(evidence.read_text(encoding="utf-8"))==data,"paired opening trials changed")
    else:
        out.mkdir(parents=True,exist_ok=True)
        temporary=evidence.with_suffix(".tmp")
        temporary.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        temporary.replace(evidence)
    print(json.dumps({key:table["report"] for key,table in data["tables"].items()},sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
