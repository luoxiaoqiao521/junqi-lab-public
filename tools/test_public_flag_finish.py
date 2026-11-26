"""Verify the opt-in final-flag decision through the normal research CLI."""
from pathlib import Path
import json
import sys

from evidence_model import sha
from evaluate_transfer import compare, totals
from partner_crossplay import evaluate, model_roster
from learn_short import EXE, RULES

OUT = EXE.parent / "public-flag-finish"
REPORT = OUT / "result.json"
CELLS = {
    "random": (920100, 64, "random", "random"),
    "historical": (920100, 64, "historical", "heuristic"),
    "draw-priority": (920200, 16, "random", "random"),
}


def rules_for(name):
    return OUT / ("draw-priority.json" if name == "draw-priority" else "archive.json")


def write_rules():
    base = json.loads(Path(RULES).read_text(encoding="utf-8-sig"))
    base["experimental"].update({"turn.order":"cw", "marshal_death.reveal_flag":True,
                                 "draw.no_progress.limit":71,
                                 "elim.no_legal_move":"eliminated_after_involved_move"})
    for name in ("random", "draw-priority"):
        base["experimental"]["terminal.priority"] = "draw_before_win" if name=="draw-priority" else "win_before_draw"
        rules_for(name).write_text(json.dumps(base,indent=2)+"\n",encoding="utf-8")


def summarize(cells):
    result={}
    for name,(seed,count,_,_) in CELLS.items():
        base, option = cells[name]
        identity = [(row["seed"],row["seat"]) for row in base]
        if identity != [(key,seat) for key in range(seed,seed+count) for seat in range(4)]:
            raise ValueError("frozen baseline coverage changed")
        summary = compare(base,option)
        changes, recovered = 0, 0
        for a,b in zip(base,option,strict=True):
            hit=b["public_flag_chances"]
            changed=b["public_flag_overrides"]
            if changed not in (0,1) or hit not in (0,1) or changed>hit:
                raise ValueError("a game had more than one confirmed ending")
            if name=="draw-priority" and hit:
                raise ValueError("draw priority was bypassed")
            if hit and b["score"]!=1:
                raise ValueError("confirmed ending did not win")
            if changed:
                if a["plies"] < b["plies"]:
                    raise ValueError("candidate preceded the frozen prefix")
                changes+=1
                recovered+=a["score"] in (0,.5)
                continue
            stable={k:v for k,v in b.items() if k!="learner_ns" and not k.startswith("public_flag_")}
            if stable!={k:v for k,v in a.items() if k!="learner_ns"}:
                raise ValueError("opt-in policy drifted outside its conditions")
        result[name]=dict(frozen=totals(base), candidate=totals(option), adopted=changes,
                          recovered=recovered, score=summary)
    if not result["random"]["adopted"]:
        raise ValueError("new normal layouts did not exercise the opt-in route")
    return result


def fingerprint():
    return {name:sha(path) for name,path in dict(engine=EXE, archive=rules_for("random"),
                                               draw_priority=rules_for("draw-priority")).items()}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    write_rules()
    roster=model_roster()
    roster["public-finish"]=dict(roster["candidate"],policy="short-final-flag")
    cells={}
    for name,(seed,count,partner,opponent) in CELLS.items():
        cells[name]=[evaluate(EXE,rules_for(name),roster,policy,partner,opponent,seed,count)
                     for policy in ("candidate","public-finish")]
    data=dict(hashes=fingerprint(), model=roster["candidate"]["artifact_sha256"],
              cells=cells, summary=summarize(cells))
    temp=REPORT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    temp.replace(REPORT)
    print(json.dumps(data["summary"],ensure_ascii=False))


def audit():
    data=json.loads(REPORT.read_text(encoding="utf-8"))
    if (data["hashes"]!=fingerprint() or data["model"]!=model_roster()["candidate"]["artifact_sha256"] or
            data["summary"]!=summarize(data["cells"])):
        raise ValueError("the public-finish receipt did not reproduce")
    print("PASS independent pairings, conservative fallback, and exact fixture profile")


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
