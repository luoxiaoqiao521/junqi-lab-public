"""Test one legal bomb-to-platoon opening swap on paired layouts."""
from collections import Counter
import json
import random
import subprocess
import sys
from audit_qq_archive import ROOT
from compare_qq_rules import VC

OUT=ROOT/"build/qq-evidence"
RESULT=OUT/"mobile-bomb-pairs.json"
EXE=OUT/"mobile_bomb_probe.exe"


def summarize(rows):
    summaries={}
    for context in (0,1):
        group=[r for r in rows if r[2]==context]
        if len(group)!=256 or len({r[0] for r in group})!=256:
            raise ValueError("invalid paired layout count")
        if any(a[7] or a[1] not in range(4) or min(a[5:7])<1 for a in group):
            raise ValueError("stationary headquarters bomb moved or invalid outcome")
        known=[(r[4]-r[3])/2 for r in group if min(r[3:5])>=0]
        band=None
        if len(known)==len(group):
            rng=random.Random(8314+context)
            samples=sorted(sum(rng.choices(known,k=len(known)))/len(known) for _ in range(2000))
            band=[samples[50],samples[1949]]
        outcome={key:dict(Counter(r[i] for r in group)) for key,i in (("baseline",3),("swap",4))}
        summaries[str(context)]=dict(complete=len(known), unknown=len(group)-len(known),
            delta=sum(known)/len(known) if known else None, interval=band, outcome=outcome,
            moved=sum(r[8] for r in group))
    if [r[0:2] for r in rows if r[2]==0] != [r[0:2] for r in rows if r[2]==1]:
        raise ValueError("contexts used different layouts")
    return summaries


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    project=ROOT.as_posix()
    sources=" ".join(f'"{project}/{p}"' for p in ("core/board.cpp","core/rules.cpp",
                       "core/state.cpp","core/layout.cpp","tests/mobile_bomb_probe.cpp"))
    command=f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /utf-8 /I"{project}/core" {sources} /Fe:mobile_bomb_probe.exe'
    subprocess.run(f'cmd.exe /d /s /c "{command}"',cwd=OUT,check=True,timeout=75)
    raw=subprocess.run([EXE],capture_output=True,text=True,check=True,timeout=50).stdout
    rows=[list(map(int,r.split())) for r in raw.splitlines()]
    stats=summarize(rows)
    temp=RESULT.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(rows=rows,summary=stats),indent=2)+"\n",encoding="utf8")
    temp.replace(RESULT)
    print("paired mobile bomb",json.dumps(stats,sort_keys=True))


def audit():
    results=json.loads(RESULT.read_text(encoding="utf8"))
    if json.loads(json.dumps(summarize(results["rows"])))!=results["summary"]:
        raise ValueError("paired layout results changed")
    print("PASS mobile bomb pairing",json.dumps(results["summary"],sort_keys=True))


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
