"""Run a fixed, local defense comparison recipe against strict practice replays."""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter

from compare_qq_rules import ROOT, VC
from evidence_model import sha
from serial_worker import read, write_receipt

SOURCE = ROOT / "build/practice-flag-window/audit-four-dark-practice-flag-save-window-2571b0fecdf1ec72eb7f"
MANIFEST = ROOT / "build/public-defense-script-input/manifest.json"
FIELDS = ("ply", "actor", "from", "to", "flag", "depth")


def to_recipe(seed, trace, window, action):
    origin, target = divmod(action["key"], 129)
    return {"seed": int(seed), "op": "assess-defense", "ply": window["ply"], "actor": window["actor"],
            "from": origin, "to": target, "flag": trace["loss"]["flag"], "depth": window["depth"]}


def prepare():
    traces = read(SOURCE / "confirm.json")["traces"]
    controls, confirmation = [], []
    for seed, trace in traces.items():
        for window in trace["windows"]:
            if not (window["cuts_loss"] and window["actual_local"] == 0 and window["counts"].get("1")):
                continue
            cohort = controls if seed in ("34122501", "34122507") else confirmation
            cohort.extend(to_recipe(seed, trace, window, action)
                          for action in window["actions"] if action["local"] == 1)
    trace44 = read(SOURCE / "control.json")["traces"]["44"]
    cut = next(w for w in trace44["windows"] if w["ply"] == 129)
    controls.append(to_recipe(44, trace44, cut, {"key": cut["actual_key"]}))
    if len(controls) != 3 or len(confirmation) != 5:
        raise ValueError("prior diagnostics no longer supply the declared receipts")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    files = {str(row["seed"]): sha(SOURCE / f'practice-{row["seed"]}.jsonl')
             for row in controls + confirmation}
    write_receipt(MANIFEST, dict(schema=1, controls=controls, confirmation=confirmation, traces=files))


def build(folder):
    names = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
             "tests/public_defense_script_audit.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{name}"' for name in names)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'{files} /Fe:public_defense_audit.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=folder, check=True, timeout=105)


def validated(row):
    if set(row) != {"op", "seed", *FIELDS} or row["op"] != "assess-defense":
        raise ValueError("unsupported recipe shape")
    if any(type(row[field]) is not int for field in ("seed", *FIELDS)):
        raise ValueError("recipe fields must be integers")
    if row["seed"] < 0 or not (1 <= row["ply"] <= 800 and 0 <= row["actor"] < 4 and
                               1 <= row["depth"] <= 4 and
                               all(0 <= row[key] < 129 for key in ("from", "to", "flag"))):
        raise ValueError("recipe parameter out of range")
    return row["op"] + " " + " ".join(str(row[f]) for f in FIELDS) + "\n"


def invoke(folder, trace, wire):
    return subprocess.run([folder / "public_defense_audit.exe", trace], input=wire, text=True,
                          encoding="utf8", capture_output=True, timeout=26)


def parse(output):
    result = {"witnesses": [], "attempts": {}}
    for line in output.splitlines():
        tag, *parts = line.split()
        if line.startswith("REPLAY PASS ") and "replay" not in result:
            result["replay"] = line
            continue
        data = [int(v) for v in parts]
        if tag == "D" and "decision" not in result:
            result["decision"] = dict(zip(("ply", "actor", "actual_key", "proposal_key", "legal", "on_shortlist"),
                                          data, strict=True))
        elif tag in ("W", "O"):
            row = dict(zip(("collision", "actual_local", "proposal_local", "referee_hash"), data, strict=True))
            if tag == "W":
                result["witnesses"].append(row)
            elif "original" not in result:
                result["original"] = row
            else:
                raise ValueError("duplicate original")
        elif tag == "S":
            label, *numbers = data
            result["attempts"][label] = dict(zip(("rank_mask", "accepted", "unresolved", "replay_failed",
                                                  "contradicted", "solver_nodes"), numbers, strict=True))
        else:
            raise ValueError("unknown native output record")
    if not {"decision", "replay", "original"} <= result.keys():
        raise ValueError("incomplete native output")
    if result["attempts"] and set(result["attempts"]) != {1, 2, 3}:
        raise ValueError("missing collision queries")
    for label, attempt in result["attempts"].items():
        if attempt["accepted"] != sum(w["collision"] == label for w in result["witnesses"]):
            raise ValueError("actual witness count does not match the attempt receipt")
    return result


def run_case(folder, manifest, recipe, repeat=False):
    wire = validated(recipe)
    trace = SOURCE / f'practice-{recipe["seed"]}.jsonl'
    if sha(trace) != manifest["traces"][str(recipe["seed"])]:
        raise ValueError("replay changed after plan was fixed")
    process = invoke(folder, trace, wire)
    if process.returncode:
        raise RuntimeError(process.stderr.strip())
    if repeat:
        second = invoke(folder, trace, wire)
        if second.returncode or second.stdout != process.stdout:
            raise ValueError("fixed request did not replay byte-for-byte")
    result = parse(process.stdout)
    expected = (recipe["ply"], recipe["actor"], recipe["from"]*129+recipe["to"])
    if tuple(result["decision"][k] for k in ("ply", "actor", "proposal_key")) != expected:
        raise ValueError("native decision does not match the request")
    return result | {"recipe": recipe, "output_sha256": sha256(process.stdout.encode()).hexdigest()}


def reject_controls(folder, manifest):
    row = manifest["controls"][0]
    args = validated(row).split()
    trace = SOURCE / f'practice-{row["seed"]}.jsonl'
    tests = {
        "undeclared_command": (["execute", *args[1:]], "bad request"),
        "stale_seat": ([*args[:2], str((row["actor"]+1) % 4), *args[3:]], "stale decision"),
        "illegal_move": ([*args[:4], args[3], *args[5:]], "illegal submitted action"),
        "invalid_flag_scope": ([*args[:5], args[3], args[-1]], "flag is not visible and friendly"),
        "depth_overrun": ([*args[:-1], "5"], "request out of bounds"),
        "unexpected_payload": ([*args, "extra"], "request out of bounds"),
    }
    for name, (wire, reason) in tests.items():
        result = invoke(folder, trace, " ".join(wire) + "\n")
        if result.returncode != 1 or result.stderr.strip() != reason or any(
                line.startswith(("D ", "O ", "W ")) for line in result.stdout.splitlines()):
            raise ValueError(f"native request control did not fail closed: {name}")
    return list(tests)


def summarize(results):
    tally = Counter()
    seeds, flips = set(), set()
    for row in results:
        tally["recipes"] += 1
        original = row["original"]
        worlds = row["witnesses"]
        tally["verified_alternate_layouts"] += len(worlds)
        for attempt in row["attempts"].values():
            tally["solver_nodes"] += attempt["solver_nodes"]
            tally["solver_unresolved"] += attempt["unresolved"]
            tally["strict_replay_rejected"] += attempt["replay_failed"]
        if len({w["collision"] for w in [original, *worlds]}) >= 2:
            tally["collision_changes"] += 1
        if {w["proposal_local"] for w in [original, *worlds]} >= {0, 1}:
            tally["local_safety_changes"] += 1
            seeds.add(row["recipe"]["seed"])
        if any(w["actual_local"]==1 and w["proposal_local"]==0 for w in worlds):
            tally["proposal_loses_to_recorded"] += 1
            flips.add(row["recipe"]["seed"])
    return dict(tally, seeds_with_safety_changes=sorted(seeds), seeds_with_preference_reversal=sorted(flips))


def main(audit, smoke):
    begun = perf_counter()
    run_dir = ROOT / "build/public-defense-script"
    run_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(dir=run_dir, prefix="smoke-")) if smoke else run_dir / os.environ["JUNQI_SERIAL_KEY"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read(MANIFEST)
    if manifest["schema"] != 1:
        raise ValueError("unknown recipe schema")
    if not audit:
        build(folder)
    native_sha = sha(folder / "public_defense_audit.exe")
    if audit and native_sha != read(folder / "controls.json")["engine_sha256"]:
        raise ValueError("native executable changed between phases")
    phase = "confirmation" if audit else "controls"
    rows = [run_case(folder, manifest, recipe, repeat=i == 0) for i, recipe in enumerate(manifest[phase])]
    for row in rows:
        early = row["recipe"]["seed"] != 44
        if row["original"]["actual_local"] != 0 or (row["original"]["proposal_local"] == 1) != early:
            raise ValueError("known defense or late control no longer reproduces")
    rejections = [] if audit else reject_controls(folder, manifest)
    summary = summarize(rows)
    if not audit and (len(summary["seeds_with_safety_changes"]) != 2 or rows[-1]["original"]["proposal_local"] != 0):
        raise ValueError("positive and late control cases did not discriminate")
    write_receipt(folder / (phase + ".json"), dict(results=rows, summary=summary, negative_tests=rejections,
                  seconds=round(perf_counter()-begun, 3), engine_sha256=native_sha))
    print(json.dumps(dict(summary, rejected_requests=rejections), sort_keys=True))
    if audit:
        print("JUNQI_RESEARCH "+("supported" if summary["seeds_with_safety_changes"] else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    args = sys.argv[1:]
    if args == ["prepare"]:
        prepare()
    elif args in ([], ["audit"], ["smoke"]):
        main(args == ["audit"], args == ["smoke"])
    else:
        raise ValueError("expected prepare, smoke, audit or no argument")
