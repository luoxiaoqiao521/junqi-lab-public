"""Find the last native four-seat defensive window in frozen practice games."""
from collections import Counter, defaultdict
import json
import os
import shutil
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter

from audit_qq_archive import ROOT
from compare_qq_rules import VC
from evidence_model import sha
from serial_worker import read, write_receipt

INPUT = ROOT / "build/practice-flag-window-input-20261125"
PREVIOUS = ROOT / "build/handoff-verify-71b88947/20261124-235946-7f3dd3af"
CONTROL = (42, 44)
HOLDOUT = range(34122500, 34122532)
NAMES = ("junqi.exe", "ruleset.json", "observation.json")
FIELDS = ("ply", "loss", "actor", "owner", "flag", "depth", "key", "actual", "local", "steps", "score", "hash")


def prepare():
    INPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "dist/junqi-mvp/junqi.exe", INPUT / "junqi.exe")
    for name in NAMES[1:]:
        shutil.copyfile(PREVIOUS / name, INPUT / name)
    write_receipt(INPUT / "manifest.json", dict(paths={name: sha(INPUT / name) for name in NAMES},
                                               control=list(CONTROL), confirm=list(HOLDOUT)))


def compile_probe(folder):
    sources = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
               "tests/practice_flag_save_window.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{p}"' for p in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'{files} /Fe:practice_flag_save_window.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=folder, check=True, timeout=100)


def native(folder, seed):
    replay = folder / f"practice-{seed}.jsonl"
    result = subprocess.run([INPUT / "junqi.exe", "simulate", str(seed), replay, INPUT / "ruleset.json",
                             INPUT / "observation.json", "flag-route"],
                            capture_output=True, encoding="utf8", check=True, timeout=12)
    with replay.open(encoding="utf8") as source:
        header = json.loads(source.readline())
    if (int(header["seed"]) != seed or header["policy"] != "flag-route" or
            header["observation"]["id"] != "four-dark-flag-practice-1"):
        raise ValueError("frozen native practice entry drifted")
    try:
        output = subprocess.run([folder / "practice_flag_save_window.exe", replay],
                                capture_output=True, encoding="utf8", check=True, timeout=8)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"native trace rejected for seed {seed}: {error.stderr.strip()}") from error
    trace = parse(output.stdout)
    trace.update(replay_sha256=sha(replay), rules_hash=header["rules_hash"],
                 observation_hash=header["observation_hash"], layouts=header["layouts"],
                 simulation_receipt=result.stdout.splitlines()[-1])
    return trace


def parse(raw):
    match, windows = None, defaultdict(list)
    verified = False
    for line in raw.splitlines():
        tag, *fields = line.split()
        if line.startswith("REPLAY PASS ") and not verified:
            verified = True
        elif tag == "L" and match is None:
            match = dict(zip(("ply", "owner", "flag", "key"), map(int, fields), strict=True))
        elif tag == "D" and match is not None:
            values = [int(v) if i != 10 else float(v) for i, v in enumerate(fields)]
            row = dict(zip(FIELDS, values, strict=True))
            windows[row["ply"]].append(row)
        else:
            raise ValueError("unexpected native loss-window output")
    if not verified or not match or bool(match["ply"]) != bool(windows):
        raise ValueError("missing loss window or no-loss control")
    result = []
    for group in windows.values():
        actual, = [row for row in group if row["actual"]]
        if (len({row["key"] for row in group}) != len(group) or
                {row["hash"] for row in group} != {actual["hash"]} or
                actual["ply"] >= match["ply"] or actual["actor"] % 2 != match["owner"] % 2):
            raise ValueError("invalid decision matrix")
        alternatives = [row for row in group if not row["actual"] and row["local"] == 1]
        counts = Counter(row["local"] for row in group)
        result.append(dict(ply=actual["ply"], depth=actual["depth"], actor=actual["actor"],
                           actual_key=actual["key"], actual_local=actual["local"], counts=counts,
                           cuts_loss=actual["ply"] + actual["depth"] - 1 == match["ply"],
                           best_save_score=max((r["score"] for r in alternatives), default=None),
                           actual_score=actual["score"], actions=group))
    return dict(loss=match, windows=result)


def summarize(traces):
    count = Counter()
    for trace in traces.values():
        count["games"] += 1
        count["games_with_flag_loss"] += bool(trace["loss"]["ply"])
        rescued = [r for r in trace["windows"] if r["cuts_loss"] and r["actual_local"] == 0 and r["counts"][1]]
        count["games_with_bounded_fork"] += bool(rescued)
        count["bounded_forks_by_own_seat"] += sum(r["actor"] == trace["loss"]["owner"] for r in rescued)
        count["bounded_forks_by_partner"] += sum(r["actor"] != trace["loss"]["owner"] for r in rescued)
        count["games_all_actions_unsafe"] += any(r["cuts_loss"] and sum(r["counts"].values()) == r["counts"][0]
                                                 for r in trace["windows"])
        count["unresolved_actions"] += sum(r["counts"][2] for r in trace["windows"])
        count["evaluated_actions"] += sum(len(r["actions"]) for r in trace["windows"])
        count["referee_steps"] += sum(a["steps"] for r in trace["windows"] for a in r["actions"])
        trace["last_bounded_fork"] = max((r["ply"] for r in rescued), default=None)
    return count


def verify_controls(traces):
    if traces[42]["loss"]["ply"] or traces[44]["loss"] != dict(ply=132, owner=2, flag=86, key=11051):
        raise ValueError("known practice cutpoints drifted")
    checkpoints = {row["ply"]: row for row in traces[44]["windows"]}
    for ply, expected in ((129, 14), (131, 29)):
        row = checkpoints[ply]
        if not row["cuts_loss"] or row["counts"] != {0: expected}:
            raise ValueError(f"late no-rescue control changed at ply {ply}")
    early = checkpoints[123]
    if early["cuts_loss"] or not early["counts"][1] or early["actual_local"] != 2:
        raise ValueError("an earlier limited survival witness was mistaken for a rescue")


def main(audit, smoke=False):
    begun = perf_counter()
    run_key = mkdtemp(prefix="junqi-flag-smoke-") if smoke else os.environ["JUNQI_SERIAL_KEY"]
    folder = ROOT / "build/practice-flag-window" / run_key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read(INPUT / "manifest.json")
    if {name: sha(INPUT / name) for name in NAMES} != manifest["paths"]:
        raise ValueError("practice inputs changed")
    phase = "confirm" if audit else "control"
    if not audit:
        compile_probe(folder)
    engine = sha(folder / "practice_flag_save_window.exe")
    if audit and read(folder / "control.json")["engine_sha256"] != engine:
        raise ValueError("native probe changed after control verification")
    traces = {seed: native(folder, seed) for seed in manifest[phase]}
    if len({str(t["layouts"]) for t in traces.values()}) != len(traces):
        raise ValueError("duplicate native layout")
    if not audit:
        verify_controls(traces)
    else:
        control = read(folder / "control.json")
        if {str(t["layouts"]) for t in traces.values()} & {str(r["layouts"]) for r in control["traces"].values()}:
            raise ValueError("held-out practice layouts overlap controls")
    summary = summarize(traces)
    report = dict(engine_sha256=engine, seconds=round(perf_counter()-begun, 3), summary=summary, traces=traces)
    write_receipt(folder / (phase + ".json"), report)
    print(json.dumps(dict(summary, seconds=report["seconds"]), sort_keys=True))
    if audit:
        print("JUNQI_RESEARCH " + ("supported" if summary["games_with_bounded_fork"] >= 8 else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] in ([], ["audit"], ["smoke"]):
        main(sys.argv[1:] == ["audit"], sys.argv[1:] == ["smoke"])
    else:
        raise ValueError("expected prepare, smoke, audit or no argument")
