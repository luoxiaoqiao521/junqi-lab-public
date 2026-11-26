"""Audit missed public options near the selfplay quiet-move deadline."""
from collections import Counter
import hashlib
import json
from statistics import mean
import sys

from audit_public_attack_windows import write_json
from calibrate_selfplay_evaluation_signal import (
    GAME, NAMES, OUT, RESULT as BASE, compile_probe, execute, sources,
)
from validate_layout_source_transfer import stat_bounds

RESULT = OUT / "stall.json"
LATE = ("block","table","turns","quiet","reverse","contact","safe","winning","learner_quiet",
        "learner_has_contact")
OFFER = ("block","table","ply","clock","kind","actual_from","actual_to","alternative_from","alternative_to")
PAIR = ("block","table","rep","actual","alternative")


def read_output(stdout):
    games, late, offers, pairs = [], [], [], []
    formats = dict(G=(games,GAME), L=(late,LATE), O=(offers,OFFER), P=(pairs,PAIR))
    lines = stdout.splitlines()
    prefix, hash_id = lines[0].split()
    if prefix!="R":
        raise ValueError("missing rule identity")
    for text in lines[1:]:
        kind,*values=text.split()
        dest,fields = formats[kind]
        dest.append(dict(zip(fields,map(int,values),strict=True)))
    return hash_id, games, late, offers, pairs


def summary(late,offers,pairs):
    result={}
    for table,name in enumerate(NAMES):
        a=[r for r in late if r["table"]==table]
        b=[r for r in offers if r["table"]==table]
        c=[r for r in pairs if r["table"]==table]
        d=[stat_bounds(r["alternative"],r["actual"]) for r in c]
        block_bounds=[]
        for offer in b:
            matched=[stat_bounds(r["alternative"],r["actual"]) for r in c if r["block"]==offer["block"]]
            block_bounds.append(tuple(map(mean,zip(*matched))))
        result[name]=dict(visited=sum(r["turns"]>0 for r in a),
                          counts={field:sum(r[field] for r in a) for field in LATE[2:]},
                          blocks_with_contact=sum(r["contact"]>0 for r in a),
                          blocks_with_safe=sum(r["safe"]>0 for r in a),
                          blocks_with_win=sum(r["winning"]>0 for r in a),
                          offer_counts=dict(Counter(str(r["kind"]) for r in b)), pairs=len(d),
                          improved=sum(0<=r["actual"]<r["alternative"] for r in c),
                          harmed=sum(0<=r["alternative"]<r["actual"] for r in c),
                          cutoffs=sum(min(r["actual"],r["alternative"])<0 for r in c),
                          positive_layouts=sum(lo>0 for lo,_ in block_bounds),
                          negative_layouts=sum(hi<0 for _,hi in block_bounds),
                          delta=list(map(mean,zip(*d))) if d else None)
    return result


def audit():
    raw = BASE.read_bytes()
    base=json.loads(raw)
    r=json.loads(RESULT.read_text(encoding="utf8"))
    assert r["baseline"]==hashlib.sha256(raw).hexdigest() and r["rule_hash"]==base["rule_hash"]
    assert r["games"]==base["games"]
    ids=[(row["block"],row["table"]) for row in r["games"]]
    assert [(row["block"],row["table"]) for row in r["late"]]==ids
    late=dict(zip(ids,r["late"],strict=True))
    for row in late.values():
        assert 0<=row["reverse"]<=row["quiet"]<=row["turns"]
        assert 0<=row["winning"]+row["safe"]<=row["contact"]<=row["quiet"]
        assert 0<=row["learner_has_contact"]<=min(row["contact"],row["learner_quiet"])
    offers={(row["block"],row["table"]):row for row in r["offers"]}
    assert len(offers)==len(r["offers"])
    expected=[(*key,rep) for key in offers for rep in range(12)]
    assert [(row["block"],row["table"],row["rep"]) for row in r["pairs"]]==expected
    for key,row in offers.items():
        assert row["kind"] in (1,2) and 60<=row["clock"]<70 and late[key]["learner_has_contact"]>0
        assert (row["actual_from"],row["actual_to"]) != (row["alternative_from"],row["alternative_to"])
    for row in r["pairs"]:
        assert row["actual"] in (-1,0,1,2) and row["alternative"] in (-1,0,1,2)
        assert offers[(row["block"],row["table"])]["kind"]!=2 or row["alternative"]==2
    assert r["summary"]==summary(r["late"],r["offers"],r["pairs"])
    print("PASS unchanged natural games",len(ids),"missed options",len(offers),"paired branches",len(expected))
    print(json.dumps(r["summary"],sort_keys=True))


def run():
    raw=BASE.read_bytes()
    selected,_=sources()
    answer,seconds=execute(selected,"stall")
    rule_hash,games,late,offers,pairs=read_output(answer.stdout)
    write_json(RESULT,dict(baseline=hashlib.sha256(raw).hexdigest(),rule_hash=rule_hash,
                           elapsed_seconds=seconds,games=games,late=late,offers=offers,pairs=pairs,
                           summary=summary(late,offers,pairs)))
    audit()


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
