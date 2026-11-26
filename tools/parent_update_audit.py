"""Compare one alternative with a wider shortlist on the same frozen decisions."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evidence_model import load, sha
from learn_short import EXE

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/"build/continuation-rate-trial-v1"
SOURCE = ROOT/"docs/REPORTS/continuation-rate-trial.json"
RUN = ROOT/"build/parent-update-audit-v1"
REPORT = ROOT/"docs/REPORTS/audit-frozen-parent-update-stability.json"
CONTEXTS = ("normal", "new_partner")
ARMS = ("prior", "smaller")
DEV = 301000
METHOD_SECONDS = 60
SCORES = (-1, 0, 1, 2)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def result_key(row):
    return row["seed"], row["seat"], row["first"]


def group(a, b):
    if a["score"] is None or b["score"] is None:
        return "truncated"
    if b["score"] > a["score"]:
        return "gain"
    if b["score"] < a["score"]:
        return "loss"
    return "tied_opportunity_change" if a["opportunities"] != b["opportunities"] else "tied"


def inputs():
    old = read(SOURCE)
    frozen = old["frozen"]
    engine, rules = ARCHIVE/"junqi.exe", ARCHIVE/"ruleset.json"
    if frozen["engine"] != sha(engine) or frozen["rules"] != sha(rules):
        raise ValueError("archived engine or rules changed")
    weights = {}
    for stream in old["streams"]:
        for arm in ARMS:
            child = f'{stream["seed"]}-{arm}'
            path = ARCHIVE/f"{child}.json"
            model = load(path, engine, rules, "short-v1")
            if (sha(path) != stream[arm]["model_sha256"] or model["parent_sha256"] !=
                    frozen["roster"]["candidate"]["artifact_sha256"]):
                raise ValueError("parent or child lineage changed")
            weights[child] = model["weights"]
    if len(weights) != 8:
        raise ValueError("expected eight frozen children")
    return old, frozen["roster"], weights


def sources(old):
    used_layouts, selected, counts = set(), [], {}
    for stream in old["streams"]:
        for arm in ARMS:
            child = f'{stream["seed"]}-{arm}'
            for context in CONTEXTS:
                parent, revised = old["parent"][context], stream[arm]["evaluation"][context]
                if [result_key(r) for r in parent] != [result_key(r) for r in revised]:
                    raise ValueError("archived opening conditions differ")
                pools = defaultdict(list)
                for a, b in zip(parent, revised, strict=True):
                    pools[group(a, b)].append((a,b))
                counts[f"{child}/{context}"] = {name: len(entries) for name, entries in pools.items()}
                for kind in ("gain", "loss", "tied_opportunity_change"):
                    pair = next((entry for entry in pools[kind] if entry[0]["seed"] not in used_layouts), None)
                    if pair:
                        a, b = pair
                        used_layouts.add(a["seed"])
                        selected.append(dict(child=child, context=context, kind=kind, seed=a["seed"],
                                             seat=a["seat"], anchor=0, expected=[a,b]))
    children = [f'{s["seed"]}-{arm}' for s in old["streams"] for arm in ARMS]
    for index, child in enumerate(children):
        for j, context in enumerate(CONTEXTS):
            selected.append(dict(child=child, context=context, kind="ordinary", seed=DEV+2*index+j,
                                 seat=(index+j)%4, anchor=12 if index%2==0 else 40))
    return selected, counts


def play(source, roster, weights):
    parent = roster["candidate"]["weights"]
    partner = roster["random" if source["context"]=="normal" else "historical"]
    args = [RUN/"junqi.exe", "parent-update-audit", RUN/"ruleset.json", source["seed"], source["seat"],
            source["anchor"], partner["policy"], *parent, *weights[source["child"]], *partner["weights"]]
    result = subprocess.run(list(map(str,args)), check=True, capture_output=True, text=True,
                            timeout=25, creationflags=0x40|0x08000000)
    return json.loads(result.stdout)


def validate_trace(source, observation):
    if "expected" not in source:
        return
    for archived, key in zip(source["expected"], ("parent","child"), strict=True):
        score = -1 if archived["score"] is None else int(2*archived["score"])
        if (observation["first"]!=archived["first"] or
                observation[key]!=[score,archived["plies"],archived["decisions"]]):
            raise ValueError(f"archived trajectory summary changed at {source['seed']} ({key})")


def score_screen(pairs):
    return sum(-2 if -1 in (a,b) else b-a for a,b in pairs)


def method_evidence(method, tested, worlds):
    screen, chosen, confirm = (method[name] for name in ("screen","picked","confirm"))
    if (len(screen)!=tested or any(len(cell)!=12//tested for cell in screen)
            or len(confirm)!=4*worlds or any(len(cell)!=3 for cell in confirm)):
        raise ValueError("unequal or incomplete rollout budget")
    if any(a not in SCORES or b not in SCORES for cell in (*screen,*confirm) for a,b in cell):
        raise ValueError("invalid team score")
    scores = list(map(score_screen,screen))
    best = 1+scores.index(max(scores))
    if chosen!=best or method["us"]<0:
        raise ValueError("confirmation did not keep the screening choice fixed")
    lows, highs = [], []
    for cell in confirm:
        lows.append(sum((0 if b<0 else b)-(2 if a<0 else a) for a,b in cell))
        highs.append(sum((2 if b<0 else b)-(0 if a<0 else a) for a,b in cell))
    if scores[best-1]<=0:
        return "no_screen_gain"
    if min(lows)>0 and worlds==2:
        return "positive_all_cells"
    if max(lows)>0 and min(highs)<0:
        return "reversed"
    if max(highs)<=0:
        return "no_confirmation_gain"
    return "uncertain"


def summarize(declared, rows):
    cells = defaultdict(Counter)
    timing = Counter()
    positive = []
    for source, row in zip(declared,rows,strict=True):
        validate_trace(source,row)
        category = source["kind"]
        cells[category]["sources"] += 1
        point = row["point"]
        if point is None:
            continue
        cells[category]["points"] += 1
        choices, worlds = point["choices"], point["worlds"]
        if (worlds not in (1,2) or not 2<=len(choices)<=min(point["legal"],5)
                or len({tuple(a[:2]) for a in choices})!=len(choices)
                or any(len(a)!=5 or not 0<a[2]<=1 or not 0<a[3]<=1 or len(a[4])!=5 for a in choices)):
            raise ValueError("invalid public shortlist or probabilities")
        cells[category]["compatible"] += int(worlds==2)
        cells[category]["exhaustive"] += int(point["legal"]==len(choices))
        for method,tested in (("A",1),("B",len(choices)-1)):
            result = method_evidence(point[method],tested,worlds)
            cells[category][f"{method}/{result}"] += 1
            timing[method] += point[method]["us"]
            chosen = point[method]["picked"]
            if result=="positive_all_cells":
                positive.append(dict(method=method,seed=source["seed"],seat=source["seat"],
                                     kind=category,ply=point["ply"],moves=[choices[0],choices[chosen]]))
    seconds = {key: timing[key]/1e6 for key in ("A","B")}
    if max(seconds.values())>METHOD_SECONDS:
        raise ValueError("a method exceeded its wall-clock budget")
    return dict(groups={k:dict(v) for k,v in cells.items()}, method_seconds=seconds,
                local_candidates=positive, promotion="no_public_rule_confirmed")


def frozen():
    return dict(source=sha(SOURCE), archive_engine=sha(ARCHIVE/"junqi.exe"),
                engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"))


def audit():
    data = read(REPORT)
    old, roster, weights = inputs()
    declared, counts = sources(old)
    rows = data["rows"]
    if (data["frozen"]!=frozen() or data["declared"]!=declared or data["full_archive"]!=counts
            or data["summary"]!=summarize(declared,rows)):
        raise ValueError("frozen audit does not match its inputs")
    for match in (0,len(rows)-1):
        a, b = play(declared[match],roster,weights), rows[match]
        for row in (a,b):
            if row["point"]:
                for method in ("A","B"):
                    row["point"][method].pop("us")
        if a!=b:
            raise ValueError("fixed replay differed")
    print("PASS archive, blind ordinary sources, equal rollout budgets, and fixed replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("parent audit already exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE,RUN/"junqi.exe")
    shutil.copyfile(ARCHIVE/"ruleset.json",RUN/"ruleset.json")
    old, roster, weights = inputs()
    declared, counts = sources(old)
    started = time.monotonic()
    rows = [play(source,roster,weights) for source in declared]
    data = dict(frozen=frozen(),declared=declared,full_archive=counts,rows=rows,
                summary=summarize(declared,rows),seconds=time.monotonic()-started)
    output = (json.dumps(data,separators=(",",":"),allow_nan=False)+"\n").encode()
    if len(output)>1024**2:
        raise ValueError("audit artifact exceeded byte cap")
    with REPORT.open("xb") as target:
        target.write(output)
    print(f"Audited {len(declared)} bounded source positions")


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
