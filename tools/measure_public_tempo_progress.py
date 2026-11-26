"""Check whether public, flag-safe won contacts improve complete-opening tempo."""
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_qq_rules import VC
from measure_whole_game_tempo import COUNT, ROOT, RULES, START, TABLES, compare, require
from partner_crossplay import model_roster
from pilot_relay_candidate_budget import file_sha

PREVIOUS = ROOT / "build/whole-game-tempo/screen-whole-game-tempo-0ca1a377fc250ed19da2/summary.json"
COLUMNS = "seed seat first table variant plies outcome ending overrides enemy_retired friendly_retired".split()
SCORE = {0:0,1:.5,2:1,-1:None}


def compile_probe(out):
    paths=("core/board.cpp","core/layout.cpp","core/rules.cpp","core/state.cpp","tests/public_tempo_progress.cpp")
    files=" ".join(f'"{ROOT.as_posix()}/{name}"' for name in paths)
    cmd=(f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX /utf-8 '
         f'/I"{ROOT.as_posix()}/core" {files} /Fe:public_tempo.exe')
    subprocess.run(f'cmd.exe /d /s /c "{cmd}"',cwd=out,check=True,timeout=100)


def summarize(rows,previous):
    output={}
    for table,label in enumerate(TABLES):
        old,new=([row for row in rows if row["table"]==table and row["variant"]==variant] for variant in (0,1))
        keys=("seed","seat","first","plies","score")
        require(all(tuple(a[k] for k in keys)==tuple(b[k] for k in keys) for a,b in
                    zip(old,previous["tables"][label]["frozen"],strict=True)),
                "frozen games differ from the preceding complete-opening receipt")
        require(len(new)==COUNT*4 and all(0<=r["overrides"]<=r["plies"] for r in new),
                "incomplete or invalid activity for the public variant")
        require(all(a["enemy_retired"]==b["enemy_retired"] and a["plies"]==b["plies"] and a["score"]==b["score"]
                    for a,b in zip(old,new,strict=True) if not b["overrides"]),
                "variant modified a game without an override")
        output[label]=dict(scoreboard=compare(old,new),
                           applied_games=sum(r["overrides"]>0 for r in new),
                           overrides=sum(r["overrides"] for r in new),
                           enemy_retired_by_140=[sum(0<r["enemy_retired"]<=140 for r in side) for side in (old,new)],
                           friendly_retired_by_140=[sum(0<r["friendly_retired"]<=140 for r in side) for side in (old,new)])
    return output


def compute(out):
    old=json.loads(PREVIOUS.read_text(encoding="utf8"))
    roster=model_roster()
    feed=" ".join(map(str,(START,COUNT,*roster["candidate"]["weights"],*roster["historical"]["weights"])))
    result=subprocess.run([out/"public_tempo.exe",RULES],input=feed,text=True,capture_output=True,
                          encoding="utf8",errors="replace",timeout=160)
    require(result.returncode==0 and result.stderr.splitlines()==[
            "PASS quiet window, guaranteed collision, bomb uncertainty, own flag exposure"],
            f"native progress screen failed: {result.stderr.strip()}")
    rows=[]
    for line in result.stdout.splitlines():
        row=dict(zip(COLUMNS,map(int,line.split()),strict=True))
        row["score"]=SCORE[row["outcome"]]
        rows.append(row)
    expected=[(table,arm,seed,seat) for table in range(2) for arm in range(2)
              for seed in range(START,START+COUNT) for seat in range(4)]
    require([(r["table"],r["variant"],r["seed"],r["seat"]) for r in rows]==expected,"wrong paired order")
    return dict(engine=file_sha(out/"public_tempo.exe"),rules=file_sha(RULES),
                prior=file_sha(PREVIOUS),findings=summarize(rows,old),rows=rows)


def main(audit):
    key=os.environ.get("JUNQI_SERIAL_KEY","manual")
    require(key and all(c.isalnum() or c=="-" for c in key),"invalid serial key")
    out=ROOT/"build/public-tempo-progress"/key
    path=out/"summary.json"
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        compile_probe(out)
    data=compute(out)
    if audit:
        require(json.loads(path.read_text(encoding="utf8"))==data,"public progress trials changed")
    else:
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf8")
        tmp.replace(path)
    print(json.dumps(data["findings"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
