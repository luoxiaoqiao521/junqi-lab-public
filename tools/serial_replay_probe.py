"""Independently validate replay families through the existing C++ referee."""
from collections import Counter
import json
import os
import subprocess
import sys
from time import perf_counter

from audit_qq_archive import ROOT
from compare_qq_rules import VC
from serial_worker import write_receipt

RULES=ROOT/"config/ruleset-selfplay.json"
FIELDS=("direction","verified_plies","contacts","mid","post","notice","stopped","result","reason","learner")


def compile_probe(path):
    files=" ".join(f'"{ROOT.as_posix()}/{p}"' for p in
                   ("core/board.cpp","core/layout.cpp","core/rules.cpp","core/state.cpp","tests/replay_intake_probe.cpp"))
    command=f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I"{ROOT.as_posix()}/core" {files} /Fe:replay_intake_probe.exe'
    subprocess.run(f'cmd.exe /d /s /c "{command}"',cwd=path,check=True,timeout=80)


def summarize(catalog, raw):
    native={key:dict(zip(FIELDS,map(int,numbers),strict=True)) for key,*numbers in
            (line.split() for line in raw.splitlines())}
    if len(native)!=len(catalog["records"]) or set(native)!={r["key"] for r in catalog["records"]}:
        raise ValueError("native probe did not account for every decoded record")
    all_rows=[x|native[x["key"]] for x in catalog["records"]]
    records={}
    for row in all_rows:
        old=records.get(row["trajectory"])
        quality=lambda r:(r["format"]==".jgs" and r["stopped"]==0,r["verified_plies"],r["stopped"]==0,r["source"]=="1")
        if old is None or quality(row)>quality(old): records[row["trajectory"]]=row
    unique=list(records.values())
    groups={}
    for suffix in (".jqh",".jgs"):
        cases=[r for r in all_rows if r["format"]==suffix]
        novel=[r for r in unique if r["format"]==suffix and not r["prior_trajectory"]]
        independent=[r for r in novel if not r["prior_layout"]]
        def count(positions):
            return dict(positions=len(positions),mid=sum(bool(r["mid"]) for r in positions),post=sum(bool(r["post"]) for r in positions),
                        both=sum(bool(r["mid"] and r["post"]) for r in positions))
        outcome_verified=[r for r in independent if r["stopped"]==0 and r["result"] in (1,2,3)]
        groups[suffix]=dict(decoded=len(cases),prior_trajectory=sum(r["prior_trajectory"] for r in cases),
                           stops=dict(Counter(str(r["stopped"]) for r in cases)),novel_trajectories=len(novel),
                           novel_independent_games=len(independent),novel_independent_families=len({r["family"] for r in independent}),
                           coverage=count(independent),full_terminal=len(outcome_verified),
                           full_no_ambiguous_control=sum(not r["notice"] for r in outcome_verified),
                           four_dark_compatible_prefix_only=suffix==".jqh")
    selected={r["key"]:r for r in all_rows}
    paired=[dict(note=n,replay=selected[n["replay"]]) for n in catalog["notes"] if n["replay"] in selected]
    both=[r for r in unique if r["mid"] and r["post"] and not r["prior_layout"] and not r["prior_trajectory"]]
    return dict(format=groups, unique_traces=len(unique),semantic_repeats=len(all_rows)-len(unique),
                annotated=paired, both_phase_families=len({r["family"] for r in both}),rows=unique)


def main(audit):
    path=ROOT/"build/replay-intake"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit: compile_probe(path)
    catalog=json.loads((path/"manifest.json").read_text(encoding="utf8"))
    begun=perf_counter()
    with (path/"games.txt").open(encoding="utf8") as source:
        result=subprocess.run([path/"replay_intake_probe.exe",RULES],stdin=source,capture_output=True,
                              encoding="utf8",check=True,timeout=160)
    print(f"referee replay seconds={perf_counter()-begun:.2f}")
    report=summarize(catalog,result.stdout)
    if audit:
        if json.loads((path/"probe.json").read_text(encoding="utf8"))!=report or (path/"probe.txt").read_text(encoding="utf8")!=result.stdout:
            raise ValueError("referee result did not replay")
    else:
        write_receipt(path/"probe.json",report)
        (path/"probe.txt").write_text(result.stdout,encoding="utf8")
    print(json.dumps({k:report[k] for k in ("format","unique_traces","semantic_repeats","both_phase_families")}))
    if audit:
        print("JUNQI_RESEARCH "+("supported" if report["both_phase_families"]>=8 else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]): raise ValueError("expected audit or no argument")
    main(bool(sys.argv[1:]))
