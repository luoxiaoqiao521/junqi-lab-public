"""Check whether vacating a center gate creates a real turn for a partner."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from camp_pairs import score_delta, scores_summary
from flag_interception import source_keys
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

TICKET = "measure-early-route-opportunities"
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build" / (TICKET + "-v1")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")
SOURCE_GRID = source_keys(303000, 64)


def validate_window(row):
    w = row["window"]
    played, opens, outside = w["moves"]
    if (w["decision"] not in (4,8,12,16) or w["decision"] > row["decisions"]
            or w["gate"] not in range(120,129) or opens[0] != w["gate"] or outside[0] == w["gate"]
            or played == outside or not 0 <= w["partner_ready"][0] <= w["partner_ready"][1] <= 25
            or not w["partner_ready"][1] or w["worlds"] not in (1,2)
            or len(w["cells"]) != 2*w["worlds"]):
        raise ValueError("invalid publicly nominated gate opportunity")
    for cell in w["cells"]:
        if len(cell)!=10 or any(len(forks)!=3 for forks in cell):
            raise ValueError("incomplete paired gate experiment")
        for forks in cell:
            if played == opens and forks[0] != forks[1]:
                raise ValueError("identical first moves diverged under identical RNG")
            for score,reached,available,used,enemy in forks:
                if not (score in (-1,0,1,2) and 0<=used<=available<=reached<=1 and enemy in (0,1)):
                    raise ValueError("invalid ally opportunity or terminal outcome")


def accumulate(forks, counts, deltas, seed, identical):
    counts["pairs"] += 1
    for name, (score,turn,can,used,enemy) in zip(("played","opens","outside"),forks,strict=True):
        counts.update({f"{name}_turn":turn, f"{name}_available":can, f"{name}_used":used,
                       f"{name}_enemy_first":enemy, f"{name}_truncated":int(score==-1)})
    for i,name in ((0,"played"),(2,"outside")):
        bounds = (0,0) if i==0 and identical else score_delta(forks[1][0],forks[i][0])
        deltas[name][seed].append(bounds)
        counts[f"gain_vs_{name}"] += int(bounds[0]>0)
        counts[f"loss_vs_{name}"] += int(bounds[1]<0)


def summarize(rows):
    if [(r["seed"],r["seat"],r["variant"]) for r in rows] != SOURCE_GRID:
        raise ValueError("early route source grid changed")
    coverage = {str(v):Counter() for v in (0,2)}
    panels, values = defaultdict(Counter), defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        counts = coverage[str(row["variant"])]
        counts["games"] += 1
        counts["reached_early_anchor"] += int(row["decisions"]>=4)
        if not row["window"]:
            continue
        validate_window(row)
        w = row["window"]
        counts["windows"] += 1
        counts[f'anchor_{w["decision"]}'] += 1
        counts["guaranteed_in_snapshot"] += int(w["partner_ready"][0]>0)
        counts["compatible_alternate"] += int(w["worlds"]==2)
        counts["already_played_opening"] += int(w["moves"][0]==w["moves"][1])
        for idx,cell in enumerate(w["cells"]):
            key = f'{row["variant"]}/{idx//2}/{idx%2}'
            for forks in cell:
                accumulate(forks,panels[key],values[key],row["seed"],w["moves"][0]==w["moves"][1])
    return dict(coverage={k:dict(v) for k,v in coverage.items()}, panels={key:dict(counts) |
                {"team":{name:scores_summary(deltas) for name,deltas in values[key].items()}}
                for key,counts in panels.items()})


def frozen(parent):
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"), parent=parent)


def replay(parent, key):
    return collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, *key, mode="early-route")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("early route run already exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    shutil.copyfile(EXE,RUN/"junqi.exe")
    shutil.copyfile(RULES,RUN/"ruleset.json")
    started = time.monotonic()
    rows = [replay(parent,key) for key in SOURCE_GRID]
    data = dict(frozen=frozen(parent), rows=rows, summary=summarize(rows), seconds=time.monotonic()-started)
    payload = (json.dumps(data,separators=(",",":"),allow_nan=False)+"\n").encode()
    if len(payload)>1024**2:
        raise ValueError("early route report exceeds byte cap")
    with REPORT.open("xb") as output:
        output.write(payload)
    print(f"Recorded {len(rows)} normal-opening sources and their early route forks")


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    rows = data["rows"]
    if data["frozen"] != frozen(parent) or data["summary"] != summarize(rows):
        raise ValueError("early route results did not reproduce")
    for variant in (0,2):
        selected = next((r for r in rows if r["variant"]==variant and r["window"]),rows[variant//2])
        if replay(parent,tuple(selected[k] for k in ("seed","seat","variant"))) != selected:
            raise ValueError("selected route replay differed")
    if replay(parent,SOURCE_GRID[-1]) != rows[-1]:
        raise ValueError("last source replay differed")
    print("PASS public gate window, paired score, no forced partner move and replay")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: early_route.py [audit]")
