"""Compare a single fixed defense intervention with its recorded action to team settlement."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter

from evidence_model import sha
from serial_public_defense_script import MANIFEST, ROOT, SOURCE, build, parse, validated
from serial_worker import read, write_receipt

PREVIOUS = ROOT / "build/public-defense-script/audit-public-flag-defense-witness-discrimination-44ceec8a17cc45172be2"
PAIR_FIELDS = ("state_hash", "desk", "trial", "recorded", "proposed", "recorded_plies", "proposed_plies",
               "recorded_end", "proposed_end")


def compare(folder, manifest, prior, repeat=False):
    recipe = prior["recipe"]
    path = SOURCE / f'practice-{recipe["seed"]}.jsonl'
    if sha(path) != manifest["traces"][str(recipe["seed"])]:
        raise ValueError("trace no longer matches the frozen input")
    args = [folder / "public_defense_audit.exe", path, "terminal"]
    command = dict(input=validated(recipe), encoding="utf8", capture_output=True, timeout=45)
    result = subprocess.run(args, **command)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    if repeat:
        other = subprocess.run(args, **command)
        if other.returncode or other.stdout != result.stdout:
            raise ValueError("paired requests did not reproduce")
    split = result.stdout.splitlines()
    short = parse("\n".join(line for line in split if not line.startswith("T ")))
    if json.loads(json.dumps(short)) != {k:v for k,v in prior.items() if k not in ("recipe", "output_sha256")}:
        raise ValueError("original, strict witnesses or local labels changed")
    worlds = {w["referee_hash"]: w for w in [short["original"], *short["witnesses"]]}
    grouped = defaultdict(list)
    for line in split:
        if not line.startswith("T "):
            continue
        pair = dict(zip(PAIR_FIELDS, map(int, line.split()[1:]), strict=True))
        worlds[pair["state_hash"]]
        grouped[pair["state_hash"]].append(pair)
    if set(grouped) != set(worlds) or any(
            [(p["desk"],p["trial"]) for p in rows] != [(d,t) for d in range(2) for t in range(4)]
            for rows in grouped.values()):
        raise ValueError("terminal rows do not cover the complete comparison matrix")
    return short | dict(recipe=recipe, pairs=[dict(world=worlds[key], runs=rows) for key,rows in grouped.items()])


def delta(pair, same):
    left,right = pair["recorded"],pair["proposed"]
    if same:
        if (left != right or pair["recorded_end"] != pair["proposed_end"] or
                pair["recorded_plies"] != pair["proposed_plies"]):
            raise ValueError("no-op intervention produced a difference")
        return 0,0
    return ((right if right >= 0 else 0)-(left if left >= 0 else 2),
            (right if right >= 0 else 2)-(left if left >= 0 else 0))


def summarize_arm(rows, desk, observed):
    tally, families, reversals = Counter(), defaultdict(list), []
    for case in rows:
        seed = case["recipe"]["seed"]
        if seed == 44:
            continue
        action = case["decision"]
        worlds = [cell for cell in case["pairs"]
                  if (cell["world"]["referee_hash"]==case["original"]["referee_hash"]) == observed]
        comparisons = []
        for cell in worlds:
            world = cell["world"]
            locally_better = world["actual_local"]==0 and world["proposal_local"]==1
            for p in cell["runs"]:
                if p["desk"] != desk:
                    continue
                lo,hi = delta(p,action["actual_key"]==action["proposal_key"])
                comparisons.append((lo,hi))
                tally["pairs"] += 1
                tally["win"] += lo > 0
                tally["harm"] += hi < 0
                tally["equal"] += lo == hi == 0
                tally["censored"] += min(p["recorded"],p["proposed"]) < 0
                tally["steps"] += p["recorded_plies"] + p["proposed_plies"]
                if locally_better and hi < 0:
                    tally["local_help_terminal_harm"] += 1
                    reversals.append(dict(seed=seed, ply=case["recipe"]["ply"], action=action["proposal_key"],
                                          desk=desk, stream=p["trial"], world=world["referee_hash"],
                                          from_score=p["recorded"], to_score=p["proposed"]))
        families[seed].append(tuple(map(mean,zip(*comparisons,strict=True))))
    block = [tuple(map(mean,zip(*x,strict=True))) for x in families.values()]
    bounds = [round(50*mean(x[i] for x in block),3) for i in range(2)]
    return dict(tally, families=len(block), equal_block_bounds=bounds, adverse_cases=reversals)


def summary(rows):
    return {f"{desk}/{source}": summarize_arm(rows,i,actual)
            for i,desk in enumerate(("flag-route","changed-enemies"))
            for source,actual in (("recorded-layout",True),("constructed-worlds",False))}


def main(audit, smoke):
    started = perf_counter()
    target = ROOT / "build/terminal-defense-script"
    target.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(dir=target,prefix="smoke-")) if smoke else target / os.environ["JUNQI_SERIAL_KEY"]
    folder.mkdir(parents=True,exist_ok=True)
    if not audit:
        build(folder)
    engine = sha(folder / "public_defense_audit.exe")
    if audit and engine != read(folder / "controls.json")["engine_sha256"]:
        raise ValueError("compiled engine changed after controls")
    phase = "confirmation" if audit else "controls"
    base = read(PREVIOUS / (phase + ".json"))
    manifest = read(MANIFEST)
    rows = [compare(folder,manifest,case,repeat=i==0) for i,case in enumerate(base["results"])]
    if not audit:
        noop = rows[-1]
        if noop["decision"]["actual_key"] != noop["decision"]["proposal_key"] or len(noop["pairs"])!=1:
            raise ValueError("expected known no-op control")
        for r in noop["pairs"][0]["runs"]:
            delta(r,True)
    results = summary(rows)
    report = dict(summary=results, cases=rows, engine_sha256=engine,
                  seconds=round(perf_counter()-started,3))
    write_receipt(folder / (phase + ".json"),report)
    print(json.dumps({k:{f:v for f,v in x.items() if f!="adverse_cases"} for k,x in results.items()}))
    if audit:
        adverse = any(r["adverse_cases"] for r in results.values())
        print("JUNQI_RESEARCH " + ("supported" if adverse else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    args = sys.argv[1:]
    if args not in ([],["audit"],["smoke"]):
        raise ValueError("expected no argument, audit or smoke")
    main(args==["audit"],args==["smoke"])
