"""Calibrate frozen selfplay tables on fresh layouts and public HQ decision controls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
import tempfile
import time

from archive_action_screen import select
from audit_public_attack_windows import write_json
from audit_qq_archive import INPUT, ROOT, local_records, public_records
from compare_qq_rules import VC
from partner_crossplay import model_roster
from replay_deadline_transfer import encode_layout
from validate_layout_source_transfer import RESULT as USED, previously_sampled, stat_bounds

OUT = ROOT / "build/selfplay-evaluation-signal"
EXE = OUT / "signal.exe"
RESULT = OUT / "calibration.json"
RULES = ROOT / "config/ruleset-selfplay.json"
N = 96
SEED = 839400
NAMES = ("random", "one_enemy", "symmetric")
GAME = ("block","source","seat","first","seed","table","score","plies")
TACTIC = ("seat","hidden","world","table","rep","a","a_plies","b","b_plies")


def sources(sample_seed=839391, exclude=()):
    manifest = {name:Path(path) for name,path in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    public = select(public_records(manifest["corpus"]))
    local = set(select(local_records(manifest["local"])))
    excluded = previously_sampled(public,local) | set(exclude)
    excluded.update(json.loads(USED.read_text(encoding="utf8"))["source"]["archive_groups"])
    keys = sorted(key for key in public if key.hex()[:16] not in excluded)
    rng = Random(sample_seed)
    picked = rng.sample(keys,N)
    generated = [(0,None,None)]*N
    archived = [(1,key.hex()[:16],public[key][1]) for key in picked]
    return generated+archived, len(keys)


def compile_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    files = ("core/board.cpp","core/rules.cpp","core/state.cpp","core/layout.cpp",
             "tests/selfplay_evaluation_signal.cpp")
    units = " ".join(f'"{ROOT.as_posix()}/{file}"' for file in files)
    cmd = f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I"{ROOT.as_posix()}/core" {units} /Fe:{EXE.name}'
    subprocess.run(f'cmd.exe /d /s /c "{cmd}"', cwd=OUT,check=True,timeout=75)


def natural(rows):
    count = Counter(r["score"] for r in rows)
    total_plies = sum(r["plies"] for r in rows)
    return dict(wins=count[2],draws=count[1],losses=count[0],cutoffs=count[-1],plies=total_plies,
                decisive_per_10000=10000*(count[0]+count[2])/total_plies,
                score_bounds=[(count[2]+count[1]*.5)/len(rows),
                              (count[2]+count[1]*.5+count[-1])/len(rows)])


def tactical(rows):
    score = [(r["b"],r["a"]) if r["world"] else (r["a"],r["b"]) for r in rows]
    intervals = [stat_bounds(best,other) for best,other in score]
    return dict(known_wins=sum(a==2 for a,_ in score), missed_play_losses=sum(b==0 for _,b in score),
                missed_play_draws=sum(b==1 for _,b in score), missed_play_wins=sum(b==2 for _,b in score),
                unknown=sum(b<0 for _,b in score), gap_bounds=list(map(mean, zip(*intervals))))


def summarize(games, controls):
    result = {}
    for table,name in enumerate(NAMES):
        group = [r for r in games if r["table"]==table]
        cases = [r for r in controls if r["table"]==table]
        result[name] = dict(generated=natural([r for r in group if r["source"]==0]),
                            archive=natural([r for r in group if r["source"]==1]),
                            pooled=natural(group),
                            public_hq=[tactical([r for r in cases if r["hidden"]==0 and r["world"]==w])
                                       for w in (0,1)],
                            hidden_hq=[tactical([r for r in cases if r["hidden"]==1 and r["world"]==w])
                                       for w in (0,1)])
    return result


def read_output(stdout):
    lines=stdout.splitlines()
    kind,hash_id = lines[0].split()
    if kind!="R":
        raise ValueError("missing rule identity")
    games, controls = [], []
    for line in lines[1:]:
        code,*rest=line.split()
        if code=="G":
            games.append(dict(zip(GAME,map(int,rest),strict=True)))
        elif code=="T":
            controls.append(dict(zip(TACTIC,map(int,rest),strict=True)))
        else:
            raise ValueError("unrecognized result line")
    return hash_id, games, controls


def execute(selected,*options,seed=SEED,partner=None):
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as feed:
        print(len(selected),file=feed)
        print(*model_roster()["candidate"]["weights"],file=feed)
        if partner is not None:
            print(*partner,file=feed)
        for block,(source,_,layouts) in enumerate(selected):
            print(source,seed+block,file=feed)
            if layouts:
                for row in layouts:
                    print(*encode_layout(row),file=feed)
        feed.seek(0)
        start=time.perf_counter()
        answer=subprocess.run([EXE,RULES,*options],stdin=feed,capture_output=True,text=True,check=True,timeout=105)
        seconds=time.perf_counter()-start
    return answer, seconds


def run():
    selected, pool=sources()
    roster=model_roster()["candidate"]
    answer, seconds = execute(selected)
    rule_hash, games, controls = read_output(answer.stdout)
    write_json(RESULT,dict(rule_hash=rule_hash, config=hashlib.sha256(RULES.read_bytes()).hexdigest(),
                          model=roster["artifact_sha256"], pool=pool, archive=[k for _,k,_ in selected if k],
                          games=games,controls=controls,elapsed_seconds=seconds,
                          summary=summarize(games,controls)))
    audit(selected,pool)


def audit(selected=None,pool=None):
    if selected is None:
        selected,pool=sources()
    r=json.loads(RESULT.read_text(encoding="utf8"))
    assert r["archive"]==[key for _,key,_ in selected if key] and r["pool"]==pool
    assert len(set(r["archive"]))==N
    assert r["config"]==hashlib.sha256(RULES.read_bytes()).hexdigest() and r["rule_hash"].isdigit()
    assert r["model"]==model_roster()["candidate"]["artifact_sha256"]
    games, controls = r["games"],r["controls"]
    expected = [(b,source,b%4,b//4%4,SEED+b,t)
                for b,(source,_,_) in enumerate(selected) for t in range(3)]
    assert [tuple(g[k] for k in GAME[:6]) for g in games]==expected
    assert all(g["score"] in (-1,0,1,2) and 0<g["plies"]<=1600 for g in games)
    script=[(seed,h,w,t,r) for seed in range(4) for h in (0,1) for w in (0,1)
            for t in range(3) for r in range(24)]
    assert [tuple(r[k] for k in TACTIC[:5]) for r in controls]==script
    for row in controls:
        finish,other=("b","a") if row["world"] else ("a","b")
        assert row[finish]==2 and row[finish+"_plies"]==1
        assert row[other] in (-1,0,1,2) and 2<=row[other+"_plies"]<=1600
    assert r["summary"]==summarize(games,controls)
    print("PASS fresh layout blocks",len(selected),"complete game rows",len(games),
          "paired tactical rows",len(controls))
    print(json.dumps({name:metrics["pooled"] for name,metrics in r["summary"].items()},sort_keys=True))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
