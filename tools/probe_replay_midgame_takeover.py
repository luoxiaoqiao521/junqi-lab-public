"""Measure single-seat takeover from strict local JGS midgame positions."""
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

from archive_pressure_training import ARCHIVE, PARENT, RULES, read
from audit_human_timing import sanitized_tape
from audit_qq_archive import INPUT, decode
from compare_qq_rules import VC
from evidence_model import load
from partner_crossplay import model_roster
from pilot_relay_candidate_budget import ROOTS, corpus_families, development_paths, file_sha, setup, sha

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "build/archive-pressure-balanced"
FROZEN = ROOT / "build/relay-candidate-budget/pilot-relay-candidate-budget-8a8fc209f912748189c5/pilot.json"
CONFIG = ROOT / "config/ruleset.json"
HORIZONS = (16, 40, 80, 160, 240)
MODES = {0: "parent", 1: "trained", 2: "human_first"}
ENDS = {-1: "unknown", 0: "loss", 1: "draw", 2: "win"}
FIELDS = {
    "R": "seat clock legal contacts known_flag human_from human_to human_contact mass retired",
    "H": "result plies contact enemy_loss enemy_exit own_exit",
    "T": "table rep mode result plies ending first_from first_to first_contact turns contact enemy_loss enemy_exit own_exit",
}
DIFF = "illegal turn retired reveal clock result first external".split()


def weights():
    parent = load(PARENT, ARCHIVE / "junqi.exe", ARCHIVE / "ruleset.json", "short-v1")
    trained = load(COURSE / "pressure.json", COURSE / "junqi.exe", COURSE / "ruleset.json", "short-v1")
    if trained.get("parent_sha256") != file_sha(PARENT):
        raise ValueError("training parent differs from frozen comparison parent")
    history = model_roster()["historical"]
    return ((parent["weights"], trained["weights"], history["weights"]),
            dict(parent=file_sha(PARENT), trained=file_sha(COURSE / "pressure.json"),
                 peer=history["artifact_sha256"]))


def prepare():
    family_hash, blocks, valid_grid, _ = setup(ROOT)
    corpus = Path(read(INPUT)["corpus"])
    excluded = corpus_families(corpus,family_hash,blocks,valid_grid)
    development, counts = development_paths(ROOTS,excluded,family_hash,blocks,valid_grid)
    frozen = read(FROZEN)["evidence"]["families"]
    if {key.hex() for key in sorted(development)[:64]} != frozen.keys():
        raise ValueError("development family frame differs from prior selection")
    models, hashes = weights()
    feed = [" ".join(" ".join(map(str, model)) for model in models)]
    sources = {}
    for family in sorted(development)[:64]:
        source,path,_,_ = development[family]
        raw = path.read_bytes()
        key,layouts,_,events = decode(raw)
        if family_hash(layouts) != family:
            raise ValueError("development source changed after selection")
        tape = sanitized_tape(raw,events)
        sources[family.hex()] = dict(source=source, replay=key[:16], events=len(events), tape=sha(bytes.fromhex(tape)))
        feed.append(f"{family.hex()} {tape}")
    data = "\n".join(feed)+"\n"
    return data, dict(sources=sources, model_sha256=hashes, source_selection=counts, feed_sha256=sha(data.encode("ascii")),
                      archive_config_sha256=file_sha(CONFIG), shadow_rules_sha256=file_sha(RULES),
                      independent_confirmation=False)


def compile_native(out):
    sources = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
               "tests/replay_midgame_takeover.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{source}"' for source in sources)
    cmd = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX /utf-8 '
           f'/I"{ROOT.as_posix()}/core" {files} /Fe:takeover.exe')
    subprocess.run(f'cmd.exe /d /s /c "{cmd}"',cwd=out,check=True,timeout=100)


def parse(output,sources,cap=240):
    roots, human, trials, diff = {}, {}, {}, {}
    for line in output.splitlines():
        tag,key,*values = line.split()
        if key not in sources:
            raise ValueError("native returned an unregistered source")
        if tag == "D":
            if key in diff or len(values) != 20:
                raise ValueError("wrong integrity row width or duplicate source")
            numbers = list(map(int,values))
            diff[key] = dict(zip(("verified","sequence","failed","missing_end"),numbers[:4],strict=True))
            diff[key].update(archive=dict(zip(DIFF,numbers[4:12],strict=True)),
                             selfplay=dict(zip(DIFF,numbers[12:20],strict=True)))
            continue
        root = key,int(values[0])
        row = dict(zip(FIELDS[tag].split(),map(int,values[1:]),strict=True))
        if tag == "T":
            root += row["table"],row["rep"],row["mode"]
        index = {"R":roots,"H":human,"T":trials}[tag]
        if root in index:
            raise ValueError("duplicate root, human record, or trial")
        index[root] = row
    grid = {k+(table,rep,mode) for k in roots for table in range(2) for rep in range(4) for mode in range(3)}
    if roots.keys()!=human.keys() or trials.keys()!=grid or diff.keys()!=sources.keys():
        raise ValueError("missing root, trial, or rule integrity record")
    for (key,ply,_,_,mode),row in trials.items():
        parent = roots[key,ply]
        if (row["result"] not in ENDS or not 1<=row["plies"]<=cap or
                mode==2 and (row["first_from"],row["first_to"])!=(parent["human_from"],parent["human_to"])):
            raise ValueError("invalid cutoff or human first move")
    return roots,human,trials,diff


def by_milestone(rows,horizons):
    keys = ("contact","enemy_loss","enemy_exit","own_exit")
    totals = {k:[sum(0<row[k]<=n for row in rows) for n in horizons] for k in keys}
    wins = [row["plies"] for row in rows if row["result"]==2]
    totals.update(win=[sum(value<=n for value in wins) for n in horizons],
                  outcomes=dict(Counter(ENDS[row["result"]] for row in rows)), n=len(rows),
                  median_win_plies=median(wins) if wins else None)
    return totals


def compare(trials,table,old,next_mode):
    pairs = [(row,trials[key,ply,table,rep,next_mode])
             for (key,ply,t,rep,mode),row in trials.items() if t==table and mode==old]
    transitions = Counter(f"{ENDS[a['result']]}->{ENDS[b['result']]}" for a,b in pairs)
    a_lo = lambda row: max(row["result"],0)
    a_hi = lambda row: 2 if row["result"]<0 else row["result"]
    both_win = [(a["plies"],b["plies"]) for a,b in pairs if a["result"]==2 and b["result"]==2]
    return dict(pairs=len(pairs),transitions=dict(transitions),
                new_loss=sum(a["result"] in (1,2) and b["result"]==0 for a,b in pairs),
                both_win_faster=sum(b<a for a,b in both_win), both_win_slower=sum(b>a for a,b in both_win),
                both_win_equal=sum(b==a for a,b in both_win),
                team_bounds=[sum(a_lo(b)-a_hi(a) for a,b in pairs)/(2*len(pairs)),
                             sum(a_hi(b)-a_lo(a) for a,b in pairs)/(2*len(pairs))])


def profile(roots,human,trials,diff,horizons):
    tables = {}
    for table in range(2):
        tables[str(table)] = {name: by_milestone([r for k,r in trials.items() if (k[2],k[4])==(table,arm)],horizons)
                              for arm,name in MODES.items()}
        tables[str(table)].update(trained_vs_parent=compare(trials,table,0,1),
                                  human_first_vs_trained=compare(trials,table,1,2))
    margins = {name:dict(Counter({key:sum(row[name][key] for row in diff.values()) for key in DIFF if key!="first"}))
               for name in ("archive","selfplay")}
    for table in range(2):
        v = [r for k,r in trials.items() if (k[2],k[4])==(table,1)]
        tables[str(table)]["trained_first_contacts"] = sum(row["first_contact"] for row in v)
        tables[str(table)]["draw_by_clock"] = {MODES[arm]:sum(r["ending"]==2 for k,r in trials.items()
                                                             if (k[2],k[4])==(table,arm)) for arm in MODES}
    rates = [r["mass"] for r in roots.values() if r["contacts"]>0]
    return dict(roots=len(roots),families=len({key for key,_ in roots}), with_retirement=sum(r["retired"]>0 for r in roots.values()),
                by_ply=dict(Counter(ply for _,ply in roots)), horizons=horizons, tables=tables,
                human=by_milestone(list(human.values()),horizons), contact_roots=len(rates),
                human_first_contacts=sum(r["human_contact"] for r in roots.values()),
                known_flag_roots=sum(r["known_flag"]>0 for r in roots.values()),
                trained_contact_mass_median=median(rates)/1000000 if rates else None,
                rule_windows=sum(v["verified"] for v in diff.values()), differences=margins,
                failures=sum(v["failed"] for v in diff.values()), missing_end=sum(v["missing_end"] for v in diff.values()))


def sample(out,cap=240):
    feed, manifest = prepare()
    binary = out / "takeover.exe"
    result = subprocess.run([binary,CONFIG,RULES,str(cap)],input=feed,capture_output=True,text=True,encoding="utf-8",
                            errors="replace",timeout=170,check=False)
    if result.returncode:
        raise ValueError(f"native takeover exit {result.returncode}: {result.stderr.strip()}")
    if result.stderr:
        (out / "replay-rejections.log").write_text(result.stderr,encoding="utf-8")
    roots,human,trials,diff = parse(result.stdout,manifest["sources"],cap)
    if not roots:
        raise ValueError("no strictly verified takeover roots")
    summary = profile(roots,human,trials,diff,HORIZONS if cap==240 else (*HORIZONS,480,960))
    summary["by_ply"] = {str(k):v for k,v in summary["by_ply"].items()}
    summary["horizons"] = list(summary["horizons"])
    data = dict(summary=summary,manifest=manifest,engine_sha256=file_sha(binary), rows_sha256=sha(result.stdout.encode("utf-8")),
                rule_sources=diff)
    return data,result.stdout


def main(audit):
    key = os.environ.get("JUNQI_SERIAL_KEY","manual")
    if not key or not all(c.isalnum() or c=="-" for c in key):
        raise ValueError("invalid local serial key")
    out = ROOT / "build/replay-midgame-takeover" / key
    started = time.perf_counter()
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        compile_native(out)
    data,rows = sample(out)
    path = out / "summary.json"
    if audit:
        old = read(path)
        if old["evidence"]!=data or (out / "rows.txt").read_text(encoding="utf-8")!=rows:
            raise ValueError("midgame takeover is not exactly reproducible")
    else:
        (out / "rows.txt").write_text(rows,encoding="utf-8")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(evidence=data,wall_seconds=round(time.perf_counter()-started,3)),
                                        indent=2,sort_keys=True)+"\n",encoding="utf-8")
        temporary.replace(path)
    print(json.dumps(data["summary"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
