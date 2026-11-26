"""Test whether one frozen partner change unmasks an earlier intervention."""
from collections import defaultdict
from hashlib import sha256
import json
from random import Random
from statistics import mean
import sys

from audit_public_attack_windows import write_json
from audit_selfplay_stall_decisions import read_output
from calibrate_selfplay_evaluation_signal import OUT, compile_probe, execute
from compare_selfplay_stall_timing import RESULT as USED, SEED, frozen_sources
from partner_crossplay import model_roster
from validate_layout_source_transfer import stat_bounds

RESULT = OUT / "carryover.json"
FIELDS = ("block","table","rep","actual","alternative","actual_split","alternative_split")


def reference():
    raw=USED.read_bytes()
    base=json.loads(raw)
    current=base["windows"]["earlier"]
    return base, sha256(raw).hexdigest(), {k:[r for r in current[k] if r["table"]==2]
                                           for k in ("games","late","offers","pairs")}


def contrast(before, after):
    totals = defaultdict(list)
    for old,new in zip(before,after,strict=True):
        a,b=stat_bounds(old["alternative"],old["actual"])
        c,d=stat_bounds(new["alternative"],new["actual"])
        totals[new["block"]].append((c-b,d-a))
    return [tuple(map(mean,zip(*outcomes))) for outcomes in totals.values()]


def summarize(before,after):
    blocks = contrast(before,after)
    rng=Random(853333)
    samples=[rng.choices(blocks,k=len(blocks)) for _ in range(1600)]
    low=sorted(mean(r[0] for r in sample) for sample in samples)
    high=sorted(mean(r[1] for r in sample) for sample in samples)
    return dict(layouts=len(blocks),pairs=len(after),
                changed_actual_replies=sum(r["actual_split"]>0 for r in after),
                changed_alternative_replies=sum(r["alternative_split"]>0 for r in after),
                improved=sum(0<=r["actual"]<r["alternative"] for r in after),
                harmed=sum(0<=r["alternative"]<r["actual"] for r in after),
                cutoffs=sum(min(r["actual"],r["alternative"])<0 for r in after),
                positive_layouts=sum(lo>0 for lo,_ in blocks),
                negative_layouts=sum(hi<0 for _,hi in blocks),
                bounds=list(map(mean,zip(*blocks))), block_interval=[low[39],high[1559]])


def run():
    base,digest,expected=reference()
    _,_,selected,_=frozen_sources()
    assert base["archive"]==[key for _,key,_ in selected if key]
    history=model_roster()["historical"]
    answer,elapsed=execute(selected,"carryover",seed=SEED,partner=history["weights"])
    baseline_lines, changed = [], []
    for line in answer.stdout.splitlines():
        if line.startswith("X "):
            changed.append(dict(zip(FIELDS,map(int,line.split()[1:]),strict=True)))
        else:
            baseline_lines.append(line)
    rule,games,late,offers,pairs=read_output("\n".join(baseline_lines))
    assert rule==base["windows"]["earlier"]["rule_hash"]
    assert dict(games=games,late=late,offers=offers,pairs=pairs)==expected
    write_json(RESULT,dict(timing_sha256=digest, rule_hash=rule, partner=history["artifact_sha256"],
                           elapsed_seconds=elapsed, baseline=pairs,changed=changed,
                           summary=summarize(pairs,changed)))
    audit()


def audit():
    base,digest,expected=reference()
    result=json.loads(RESULT.read_text(encoding="utf8"))
    assert result["timing_sha256"]==digest and result["rule_hash"]==base["windows"]["earlier"]["rule_hash"]
    assert result["partner"]==model_roster()["historical"]["artifact_sha256"]
    assert result["baseline"]==expected["pairs"]
    lookup={(r["block"],r["table"]):r["ply"] for r in expected["offers"]}
    for a,b in zip(result["baseline"],result["changed"],strict=True):
        assert all(a[k]==b[k] for k in FIELDS[:3])
        for arm in ("actual","alternative"):
            first=b[arm+"_split"]
            assert b[arm] in (-1,0,1,2)
            if first:
                assert lookup[b["block"],b["table"]]<first<=1600
            else:
                assert b[arm]==a[arm]
    assert result["summary"]==summarize(result["baseline"],result["changed"])
    print("PASS same roots and original continuation; one frozen partner contrast")
    print(json.dumps(result["summary"],sort_keys=True))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
