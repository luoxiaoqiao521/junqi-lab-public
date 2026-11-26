"""Audit concrete alternatives in the natural opening and the response shortlist."""
from collections import Counter
import json
import os
import subprocess
import sys

from measure_deadline_opening import LABEL, ROOT, RULES, build, file_sha, model_roster, require
from serial_worker import write_receipt

START, COUNT = 1203100, 32
CATEGORIES = ("proved_direct_flag", "possible_direct_flag", "geometry_next_turn")
KEYS = ("legal", "listed", "selected")
FOLLOW = ("returned", "lost_piece", "route_remains", "followed", "captured")
BASE = ("seed", "seat", "partner", "table", "plies", "enemy_exit", "friendly_exit", "outcome", "ending", "decisions")
ALTERNATIVES = ("legal_alternatives", "listed_alternatives")
GEOMETRY_SCORES = ("geometry_filtered", "geometry_below", "geometry_tied")
REFERENCE = (ROOT/"build/first-exit-coverage/audit-first-enemy-retirement-progress-ebd67a55299f6b5ad6aa/natural.json")


def summarize(raw):
    fields = (*BASE,*(f"{a}_{b}" for a in CATEGORIES for b in KEYS),*FOLLOW,
              *(f"{a}_{b}" for a in CATEGORIES for b in ALTERNATIVES),*GEOMETRY_SCORES)
    rows = [dict(zip(fields,map(int,line.split()),strict=True)) for line in raw.splitlines()]
    expected = [(s,j,p,t) for s in range(START,START+COUNT) for j in range(4) for p in (1,3) for t in (0,1)]
    require([tuple(row[field] for field in BASE[:4]) for row in rows] == expected,"misplaced or missing natural start")
    for row in rows:
        require(0<row["plies"]<=140 and max(row["enemy_exit"],row["friendly_exit"])<=row["plies"] and
                row["outcome"] in LABEL,"invalid retirement or terminal time")
        for c in CATEGORIES:
            require(0<=row[f"{c}_selected"]<=row[f"{c}_listed"]<=row[f"{c}_legal"]<=row["decisions"],
                    "invalid nested opportunity counts")
            require(0<=row[f"{c}_listed_alternatives"]<=row[f"{c}_legal_alternatives"],
                    "invalid alternative nominations")
        require(all(row[k]>=0 for k in GEOMETRY_SCORES) and
                sum(row[k] for k in GEOMETRY_SCORES)==row["geometry_next_turn_legal_alternatives"],
                "incomplete natural scores for approach alternatives")
        require(0<=row["captured"]<=row["followed"]<=row["route_remains"]<=row["returned"]-row["lost_piece"]
                <=row["geometry_next_turn_selected"], "invalid follow-through records")
    panels = {}
    for p in (1,3):
        for t in (0,1):
            group = [r for r in rows if r["partner"]==p and r["table"]==t]
            panels[f"{p}/{t}"] = dict(games=len(group), sources=len({r["seed"] for r in group}),
                decisions=sum(r["decisions"] for r in group),
                opportunities={c:{k:sum(r[f"{c}_{k}"] for r in group) for k in KEYS} for c in CATEGORIES},
                alternatives={c:{k:sum(r[f"{c}_{k}"] for r in group) for k in ALTERNATIVES} for c in CATEGORIES},
                geometry_scores={k:sum(r[k] for r in group) for k in GEOMETRY_SCORES},
                follow_through={k:sum(r[k] for r in group) for k in FOLLOW},
                by_horizon={str(h):dict(first_enemy=sum(0<r["enemy_exit"]<=h for r in group),
                    first_friendly=sum(0<r["friendly_exit"]<=h for r in group),
                    results=dict(Counter((LABEL[r["outcome"]] if r["ending"] else "ongoing")
                                         if r["plies"]<=h else "ongoing" for r in group))) for h in (100,140)})
    return dict(panels=panels,rows=rows)


def main(audit):
    path=ROOT/"build/first-exit-coverage"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe=path/"deadline_opening.exe"
    roster=model_roster()
    models=[roster[name] for name in ("candidate","historical")]
    payload=" ".join(map(str,(START,COUNT,*(v for m in models for v in m["weights"]))))
    run=subprocess.run([exe,RULES,"first-exit-coverage"],cwd=ROOT,input=payload,encoding="utf8",
                       capture_output=True,timeout=150)
    require(run.returncode==0,f"natural coverage probe failed: {run.stderr.strip()}")
    result=dict(engine=file_sha(exe),rules=file_sha(RULES),models=models,**summarize(run.stdout))
    previous=json.loads(REFERENCE.read_text(encoding="utf8"))
    require(result["models"]==previous["models"] and result["rules"]==previous["rules"],
            "frozen weights or rules changed")
    require(all(all(current[key]==value for key,value in old.items() if not key.endswith("_listed"))
                for current,old in zip(result["rows"],previous["rows"],strict=True)),
            "natural outcome or coverage changed beyond shortlist correction")
    if audit:
        require(run.stdout==(path/"natural.txt").read_text(encoding="utf8") and
                result==json.loads((path/"natural.json").read_text(encoding="utf8")),"natural coverage replay changed")
    else:
        (path/"natural.txt").write_text(run.stdout,encoding="utf8")
        write_receipt(path/"natural.json",result)
    print(json.dumps(result["panels"],indent=2))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
