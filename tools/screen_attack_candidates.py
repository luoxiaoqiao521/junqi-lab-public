"""Screen proposed quiet attack routes at fixed pre-move replay windows."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from archive_action_screen import layout_group, select
from audit_qq_archive import INPUT, ROOT, initial_board, local_records, public_records
from audit_public_attack_windows import OUT, write_json
from compare_qq_rules import VC, tape
from evidence_model import sha
from partner_crossplay import model_roster
from verify_attack_opportunity import source as anchor_source

EXE=OUT/"attack_candidate_screen.exe"
RESULT=OUT/"candidate-screen.json"
JQH_EXPORT=ROOT/"build/private/jqh-candidate-input.json"
PLIES=(24,72,120)
FAMILIES=("jqh","qq")


def jqs_sources():
    try:
        with JQH_EXPORT.open("rb") as source:
            raw=source.read(16_777_217)
    except OSError:
        raise ValueError("A local passive JQH candidate export is required") from None
    if len(raw)>16_777_216:
        raise ValueError("JQH candidate export size exceeded")
    bundle=json.loads(raw)
    if bundle.get("schema")!=1 or not isinstance(bundle.get("cases"),list):
        raise ValueError("Invalid JQH candidate schema")
    for case in bundle["cases"]:
        key,pieces,events=case["key"],case["pieces"],case["events"]
        if (not isinstance(key,str) or not key or len(key)>80 or
                not all(c.isascii() and (c.isalnum() or c in "-_") for c in key)):
            raise ValueError("Invalid anonymous JQH case key")
        if not 1<=len(pieces)<=100 or not 1<=len(events)<=2000:
            raise ValueError("Unexpected JQH candidate record count")
        for rows,width in ((pieces,3),(events,5)):
            if any(not isinstance(row,list) or len(row)!=width or
                   any(type(value)is not int or not 0<=value<=16 for value in row) for row in rows):
                raise ValueError("Expected bounded JQH integers")
        yield "jqh-"+key,pieces,events


def qq_sources():
    manifest={k:Path(v) for k,v in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    public=select(public_records(manifest["corpus"]))
    local=set(select(local_records(manifest["local"])))
    anchor=layout_group(anchor_source()[1])
    pool=sorted(public.keys()-local-{anchor})
    old_open=set(random.Random(31415).sample(sorted(public.keys()-local),192))
    chosen=random.Random(993505).sample(sorted(set(pool)-old_open),64)
    for group in chosen:
        key,layouts,seats,events=public[group]
        board=[(x,y,rank) for (x,y),(_,rank) in initial_board(layouts).items()]
        rows,direction=tape(layouts,seats,events)
        if direction:
            rows=[]
        yield "qq-"+key[:14],board,rows


def compile_probe():
    project=ROOT.as_posix()
    files=" ".join(f'"{project}/{name}"' for name in
                   ("core/board.cpp","core/rules.cpp","core/state.cpp","core/layout.cpp",
                    "tests/attack_candidate_screen.cpp"))
    args=(f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT '
          f'/utf-8 /I"{project}/core" {files} /Fe:attack_candidate_screen.exe')
    subprocess.run(f'cmd.exe /d /s /c "{args}"',cwd=OUT,check=True,timeout=75)


def send(output,group,board,events,ply):
    seen=0
    prefix=[]
    for row in events:
        prefix.append(row)
        seen+=row[0]>=0
        if seen==ply:
            break
    else:
        prefix=[]
    print(group,int(group.startswith("qq")),ply,len(prefix),file=output)
    for piece in board:
        print(*piece,file=output)
    for row in prefix:
        print(*row,file=output)


def collect(output):
    sampled={}
    pairs=defaultdict(list)
    for line in output.splitlines():
        tag,group,p,*data=line.split()
        key=f"{group}/{p}"
        values=list(map(int,data))
        if tag=="P":
            response,seed,*scores=values
            label=f"{key}/{response}"
            if seed!=len(pairs[label]):
                raise ValueError("broken matched sequence")
            pairs[label].append(scores)
        elif tag=="X":
            sampled[key]=None
        elif tag=="D":
            names=("legal","routes","played_route","routes_rank","paired")
            sampled[key]=dict(zip(names,values,strict=True))
        else:
            raise ValueError("unexpected evaluation line")
    return sampled,dict(pairs)


def summary(sampled,pairs):
    result={}
    for family in FAMILIES:
        all_rows=[r for k,r in sampled.items() if k.startswith(family)]
        aligned=[r for r in all_rows if r is not None]
        choices=[r for r in aligned if r["routes"]]
        cells=[rows for name,rows in pairs.items() if name.startswith(family)]
        paired=[pair for rows in cells for pair in rows]
        follow=[p for p in paired if p[3]!=-2]
        result[family]=dict(requested=len(all_rows),aligned=len(aligned),candidate_windows=len(choices),
                            recorded_route=sum(r["played_route"] for r in choices),
                            rank_below_first_three=sum(r["routes_rank"]>=3 for r in choices),
                            paired_layouts=sum(r["paired"] for r in aligned),pairs=len(paired),
                            still_legal=sum(p[2] for p in paired),follow_with_alternative=len(follow),
                            start_cutoffs=sum(v<0 for p in paired for v in p[:2]),
                            start_natural_delta=dict(Counter(p[0]-p[1] for p in paired if min(p[:2])>=0)),
                            follow_cutoffs=sum(v<0 for p in follow for v in p[3:]),
                            follow_natural_delta=dict(Counter(p[3]-p[4] for p in follow if min(p[3:])>=0)))
    return result


def run():
    selected=list(jqs_sources())+list(qq_sources())
    if Counter(row[0].split("-")[0] for row in selected)!={"jqh":61,"qq":64}:
        raise ValueError("source manifest failed to cover all scheduled layouts")
    compile_probe()
    model=model_roster()["candidate"]
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as stream:
        print(len(selected)*len(PLIES),file=stream)
        print(*model["weights"],file=stream)
        for group,board,events in selected:
            for ply in PLIES:
                send(stream,group,board,events,ply)
        stream.seek(0)
        process=subprocess.run([EXE],stdin=stream,capture_output=True,text=True,check=True,timeout=130)
    sampled,pairs=collect(process.stdout)
    write_json(RESULT,dict(model=model["artifact_sha256"],jh_sidecar=sha(JQH_EXPORT),sampled=sampled,
                           pairs=pairs,summary=summary(sampled,pairs)))
    audit()


def audit():
    data=json.loads(RESULT.read_text(encoding="utf-8"))
    sampled,pairs=data["sampled"],data["pairs"]
    counts=Counter(key.split("-")[0] for key in sampled)
    if (counts!={"jqh":183,"qq":192} or data["jh_sidecar"]!=sha(JQH_EXPORT) or
            data["model"]!=model_roster()["candidate"]["artifact_sha256"]):
        raise ValueError("the source groups or frozen weights changed")
    positive={key for key,val in sampled.items() if val and val["paired"]}
    if (set(pairs)!={f"{key}/{response}" for key in positive for response in (0,2)} or
            any(len(rows)!=16 for rows in pairs.values())):
        raise ValueError("paired coverage did not reconcile")
    if data["summary"]!=json.loads(json.dumps(summary(sampled,pairs))):
        raise ValueError("summary did not reproduce")
    print(json.dumps(data["summary"]))


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
