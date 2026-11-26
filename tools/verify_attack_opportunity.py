"""Verify the two decisions around 000433 against the archive and C++ referee."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import subprocess
import sys
import tarfile
import tempfile

from audit_qq_archive import INPUT, LIMIT, ROOT, decode, initial_board
from audit_public_attack_windows import OUT, audit as audit_coverage, write_json
from compare_qq_rules import VC, tape
from local_archive_inputs import archive_value
from partner_crossplay import model_roster

EXE=OUT/"attack_opportunity_probe.exe"
RESULT=OUT/"opportunity.json"


def compile_probe():
    OUT.mkdir(parents=True,exist_ok=True)
    project=ROOT.as_posix()
    files=" ".join(f'"{project}/{name}"' for name in
                   ("core/board.cpp","core/rules.cpp","core/state.cpp","core/layout.cpp",
                    "tests/attack_opportunity_probe.cpp"))
    command=(f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
             f'/I"{project}/core" {files} /Fe:attack_opportunity_probe.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"',cwd=OUT,check=True,timeout=75)


def source():
    corpus=json.loads(INPUT.read_text(encoding="utf-8-sig"))["corpus"]
    name=archive_value("attack_member")
    with tarfile.open(Path(corpus),"r:gz") as archive:
        try:
            entry=archive.getmember(name)
        except (KeyError, TypeError):
            raise ValueError("Selected archive member is unavailable") from None
        if not entry.isfile() or not 0<entry.size<=LIMIT:
            raise ValueError("Expected a regular bounded archive member")
        with archive.extractfile(entry) as replay:
            return decode(replay.read(LIMIT+1))


def parse(text):
    evidence,replies,pairs={}, {}, defaultdict(list)
    for line in text.splitlines():
        tag,*args=line.split()
        if tag=="C":
            world,outcome,defender,start,start_other,follow,switch=args
            evidence[world]=dict(outcome=int(outcome),defender_rank=int(defender),start=start,start_other=start_other,
                                 follow=follow,switch=switch)
        elif tag=="R":
            world,root,*stats=args
            fields=("legal","front_removed","allows_front_win","counter_flag",
                    "recorded_ally_reply_legal","side_legal","side_survives",
                    "side_available_after_block","side_survives_after_block")
            replies[f"{world}/{root}"]=dict(zip(fields,map(int,stats),strict=True))
        elif tag=="P":
            phase,world,table,seed,one,two=args
            label=f"{phase}/{world}/{table}"
            if int(seed)!=len(pairs[label]):
                raise ValueError("incomplete pair stream")
            pairs[label].append([int(one),int(two)])
        else:
            raise ValueError("unknown witness output")
    return evidence,replies,dict(pairs)


def summary(pairs):
    result={}
    for cell,rows in pairs.items():
        counts=[Counter(row[i] for row in rows) for i in range(2)]
        low=sum((a if a>=0 else 0)-(b if b>=0 else 2) for a,b in rows)/(2*len(rows))
        high=sum((a if a>=0 else 2)-(b if b>=0 else 0) for a,b in rows)/(2*len(rows))
        result[cell]=dict(recorded=dict(counts[0]),alternative=dict(counts[1]),bounds=[low,high])
    return result


def run():
    key,layouts,seats,events=source()
    rows,ccw=tape(layouts,seats,events)
    if ccw or sum(row[0]>=0 for row in rows) != 119:
        raise ValueError("unexpected archive direction or ending")
    compile_probe()
    model=model_roster()["candidate"]
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as script:
        print(len(rows),file=script)
        print(*model["weights"],file=script)
        for (x,y),(_,rank) in initial_board(layouts).items():
            print(x,y,rank,file=script)
        for entry in rows:
            print(*entry,file=script)
        script.seek(0)
        proc=subprocess.run([EXE],stdin=script,capture_output=True,text=True,timeout=60)
        if proc.returncode:
            raise ValueError(proc.stderr.strip() or f"archive probe failed with {proc.returncode}")
    evidence,replies,pairs=parse(proc.stdout)
    write_json(RESULT,dict(source=key[:16],model=model["artifact_sha256"],evidence=evidence,
                           replies=replies,pairs=pairs,summary=summary(pairs)))
    audit()


def audit():
    r=json.loads(RESULT.read_text(encoding="utf-8"))
    if r["model"]!=model_roster()["candidate"]["artifact_sha256"]:
        raise ValueError("frozen model changed")
    evidence=r["evidence"]
    if [evidence[w]["outcome"] for w in ("0","1")] != [1,2]:
        raise ValueError("the same follow-up did not change from mine to stronger mobile piece")
    if any(evidence["0"][name]!=evidence["1"][name] for name in ("start","start_other","follow","switch")):
        raise ValueError("action choice changed when hidden identities did")
    if set(r["replies"])!={f"{world}/{root}" for world in range(2) for root in range(2)}:
        raise ValueError("one defensive response set was not enumerated")
    for cell in r["replies"].values():
        if (not 0<=cell["side_survives_after_block"]<=cell["side_available_after_block"]<=cell["front_removed"]
                or not 0<=cell["side_survives"]<=cell["side_legal"]<=cell["recorded_ally_reply_legal"]<=cell["legal"]):
            raise ValueError("defensive subgroups did not reconcile")
    expected={f"{phase}/{world}/{table}" for phase in ("start","continue")
              for world in range(2) for table in range(3)}
    pairs=r["pairs"]
    if set(pairs)!=expected or any(len(rows)!=32 for rows in pairs.values()):
        raise ValueError("a paired response cell is incomplete")
    if r["summary"]!=json.loads(json.dumps(summary(pairs))):
        raise ValueError("summary did not reproduce")
    audit_coverage()
    print("compatible follow-up",json.dumps(evidence,ensure_ascii=False))
    print("legal defensive replies",json.dumps(r["replies"]))
    print("paired deltas",json.dumps({k:v["bounds"] for k,v in r["summary"].items()}))


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
