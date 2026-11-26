"""Check same-width public defense nominations and two opposed teams on frozen layouts."""
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys
from time import perf_counter

from measure_deadline_opening import ROOT, build, require
from serial_archive_pressure import archives
from serial_dark_ordinary_learning import RULES, model_wire
from serial_dark_partner_support import SOURCE
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt

PARENT = SOURCE / "original.json"
SEED = 34129000
DEVELOPMENT = [(SEED-2, 0, 0), (SEED-1, 1, 1)]
PRESSURE = [(SEED+i, i%4, table) for i in range(15) for table in range(2)]
GAME_FIELDS = ("seed learner desk arm first layout trace result plies danger in_list nominated "
               "selected actual_risk validated_selected").split()
METERS = ("plies contacts flag_captures known_flag_offers known_flag_taken partner_replies "
          "risk clear unresolved").split()
NOMINATION_FIELDS = "seed desk arm ply actor base candidate selected actual_risk".split()
OUTCOMES = {-1: "truncated", 0: "loss", 1: "draw", 2: "win"}


def meter(values):
    item = dict(zip(METERS, map(int, values), strict=True))
    require(item["plies"] == item["risk"]+item["clear"]+item["unresolved"] and
            0 <= item["known_flag_taken"] <= item["known_flag_offers"] <= item["plies"] and
            0 <= item["partner_replies"] <= item["contacts"] <= item["plies"] and
            0 <= item["flag_captures"] <= item["contacts"], "invalid per-team telemetry")
    return item


def native(folder, mode, wire=""):
    run = subprocess.run([folder/"deadline_opening.exe", RULES, "public-flag-reply-"+mode],
                         input=wire, encoding="utf8", capture_output=True, timeout=110)
    require(run.returncode == 0, f"{mode} failed: {run.stderr.strip()}")
    return run.stdout.splitlines()


def game_rows(lines, grid, arms=(0, 1)):
    require(lines and lines[0].startswith("RULES "), "missing native stress rules")
    rows, nominations = [], {}
    for line in lines[1:]:
        tag, *values = line.split()
        if tag == "N":
            row = dict(zip(NOMINATION_FIELDS, map(int, values), strict=True))
            nominations.setdefault(tuple(row[n] for n in ("seed", "desk", "arm")), []).append(row)
            continue
        require(tag == "G", "unexpected native game receipt")
        row = dict(zip(GAME_FIELDS, map(int, values[:len(GAME_FIELDS)]), strict=True))
        ally, enemy = values[len(GAME_FIELDS):len(GAME_FIELDS)+len(METERS)], values[len(GAME_FIELDS)+len(METERS):]
        row.update(ours=meter(ally), opponent=meter(enemy))
        row["nominations"] = nominations.pop(tuple(row[n] for n in ("seed", "desk", "arm")), [])
        require(row["ours"]["plies"]+row["opponent"]["plies"] == row["plies"] and
                row["result"] in OUTCOMES and 0 < row["plies"] <= 720 and
                row["nominated"] == len(row["nominations"]) and
                row["actual_risk"] == sum(n["actual_risk"] for n in row["nominations"]) and
                row["selected"] == sum(n["selected"] != n["base"] for n in row["nominations"]) and
                0 <= row["validated_selected"] <= min(row["actual_risk"], row["selected"]),
                "game and nomination evidence disagree")
        rows.append(row)
    expected = [(seed, learner, table, arm) for seed, learner, table in grid for arm in arms]
    require(not nominations and [tuple(r[key] for key in GAME_FIELDS[:4]) for r in rows] == expected,
            "two pressure teams and both frozen arms were not exercised on every assigned layout")
    for old, new in zip(rows[::2],rows[1::2],strict=True) if len(arms)==2 else ():
        require((old["layout"], old["first"]) == (new["layout"], new["first"]), "unmatched opening")
        if not old["selected"]:
            require(old["trace"] == new["trace"] and old["result"] == new["result"] and
                    old["plies"] == new["plies"], "no-op nominations changed the frozen game")
    return int(lines[0].split()[1]), rows


def archive_panel(folder):
    choices, wires = archives()
    selected = {}
    for suffix in (".jqh", ".jgs"):
        group = [(key, row) for key, row in choices.items() if row["audit"] and row["format"] == suffix]
        selected.update(group[:15])
    require(len({r["family"] for r in selected.values()}) == 30, "archive panel crossed layout families")
    wire = "".join(wires[key] for key in selected)
    rows = {}
    for line in native(folder, "archive", wire):
        tag, key, cap, plies, diff, reason, *values = line.split()
        require(tag == "H" and key in selected and key not in rows, "unexpected archive receipt")
        cap, plies, diff = map(int, (cap, plies, diff))
        alternatives, matched, unique, both, risk = map(int, values[:5])
        require(cap == selected[key]["cap"] and 0 <= plies <= cap and
                (diff == plies+1 if reason != "matched" else diff == 0) and
                0 <= unique <= matched <= risk <= plies and 0 <= both <= alternatives <= plies,
                "archive source exceeded its strict replay prefix")
        a, b = meter(values[5:5+len(METERS)]), meter(values[5+len(METERS):])
        require(a["plies"]+b["plies"] == plies, "archive telemetry exceeds agreed prefix")
        rows[key] = dict(format=selected[key]["format"], plies=plies, diff=diff, reason=reason,
                         alternatives=alternatives, matched=matched, unique_clear=unique, both=both,
                         risk_roots=risk, first=a, second=b)
    require(rows.keys() == selected.keys(), "archive panel missing selected families")
    return rows


def sum_metrics(rows, key):
    return {name: sum(r[key][name] for r in rows) for name in METERS}


def summary(rows, sources):
    result = {}
    for desk in range(2):
        old = [r for r in rows if r["desk"] == desk and r["arm"] == 0]
        new = [r for r in rows if r["desk"] == desk and r["arm"] == 1]
        pairs = list(zip(old,new,strict=True))
        ob = {OUTCOMES[n]:sum(r["result"] == n for r in old) for n in OUTCOMES}
        nb = {OUTCOMES[n]:sum(r["result"] == n for r in new) for n in OUTCOMES}
        result[str(desk)] = dict(baseline=ob, probe=nb, paired=contrast([
            score_difference(a["result"],b["result"]) for a,b in pairs]),
            score_lower_upper=[[(x["win"]+.5*x["draw"])/len(old),
                                (x["win"]+.5*x["draw"]+x["truncated"])/len(old)] for x in (ob,nb)],
            trace_changes=sum(a["trace"] != b["trace"] for a,b in pairs),
            selection={k:[sum(r[k] for r in arm) for arm in (old,new)] for k in GAME_FIELDS[9:]},
            metrics=[{k:sum_metrics(arm,k) for k in ("ours","opponent")} for arm in (old,new)])
    archive = {}
    for suffix in (".jqh", ".jgs"):
        sample = [r for r in sources.values() if r["format"] == suffix]
        archive[suffix] = dict(families=len(sample), boundaries=dict(Counter(r["reason"] for r in sample)),
            totals={n:sum(r[n] for r in sample) for n in
                    ("plies","alternatives","matched","unique_clear","both","risk_roots")},
            metrics={n:sum(r["first"][n]+r["second"][n] for r in sample) for n in METERS})
    return dict(desks=result, archive=archive)


def research_outcome(report):
    desks = list(report["desks"].values())
    if all(r["selection"]["validated_selected"][1] >= 2 and r["paired"]["interval"][0] > 0 for r in desks):
        return "supported"
    if any(r["paired"]["interval"][1] < 0 for r in desks):
        return "refuted"
    if all(r["selection"]["actual_risk"][0] >= 8 and r["selection"]["selected"][0] == 0 for r in desks):
        return "refuted"
    return "inconclusive"


def main(audit):
    began = perf_counter()
    folder = ROOT / "build" / "public-flag-reply" / os.environ["JUNQI_SERIAL_KEY"]
    folder.mkdir(parents=True, exist_ok=True)
    if not audit:
        build(folder)
        require(native(folder, "controls") == ["CONTROL public-flag-reply"], "missing native controls")
    engine = hashlib.sha256((folder/"deadline_opening.exe").read_bytes()).hexdigest()
    grid = PRESSURE if audit else DEVELOPMENT
    model = model_wire(read(PARENT))
    raw = native(folder, "games", model+"\n"+"\n".join(" ".join(map(str,row)) for row in grid)+"\n")
    rules_hash, rows = game_rows(raw, grid)
    repeat = model+"\n"+" ".join(map(str,grid[0]))+"\n"
    require(native(folder, "games", repeat) == [raw[0]] + [
        line for line in raw[1:] if line.split()[1] == str(grid[0][0]) and
        int(line.split()[{"G":3,"N":2}[line[0]]]) == grid[0][2]], "native repeat disagrees")
    receipt = dict(engine=engine, rules_hash=rules_hash, grid=grid, games=rows)
    if audit:
        prior = read(folder/"controls.json")
        require(prior["engine"] == engine and prior["rules_hash"] == rules_hash and
                not {r["layout"] for r in prior["games"]} & {r["layout"] for r in rows},
                "development and confirmation share a layout or use different engines")
        sources = archive_panel(folder)
        receipt.update(archive=sources, summary=summary(rows,sources))
        useful = sum(r["validated_selected"] for r in rows if r["arm"] == 1)
        receipt["outcome"] = research_outcome(receipt["summary"])
        print(json.dumps(dict(games=len(rows), archives=len(sources), verified_selections=useful,
                              summary=receipt["summary"]), ensure_ascii=False))
    receipt["seconds"] = round(perf_counter()-began,3)
    write_receipt(folder/("confirmation.json" if audit else "controls.json"), receipt)
    if audit:
        print("JUNQI_RESEARCH " + receipt["outcome"])
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    main(sys.argv[1:] == ["audit"])
