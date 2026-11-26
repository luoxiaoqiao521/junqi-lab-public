"""Verify the public hypothesis round trip against strictly loaded local JGS replays."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from compare_qq_rules import VC

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (
    "actual_moves", "roots", "revealed_roots", "foreign_flag_roots", "retired_flag_roots",
    "nonzero_clock_roots", "sampled_roots", "worlds", "rejected_attempts", "legacy_hidden_flag_worlds",
    "reply_actor_moved_evidence_roots", "reply_actor_proven_engineer_roots", "reply_actor_previous_move_roots",
    "active_foreign_non_marshal_exhaustion_roots",
)
MAX_FILES = 512


def build(directory):
    sources = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
               "tests/hypothesis_archive.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{source}"' for source in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'{files} /Fe:hypothesis_archive.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=directory, check=True, timeout=90)
    return Path(directory) / "hypothesis_archive.exe"


def read_inputs(sources):
    files = {}
    for source in sources:
        if not source.exists():
            raise SystemExit("A local replay input does not exist.")
        entries = sorted(source.rglob("*.jgs")) if source.is_dir() else [source]
        for entry in entries:
            files[entry.resolve()] = None
    if not 1 <= len(files) <= MAX_FILES:
        raise SystemExit(f"Use between 1 and {MAX_FILES} local JGS files per run.")
    tapes = []
    for path in files:
        if 412 <= path.stat().st_size <= 65536:
            tapes.append(path.read_bytes().hex())
    return len(files), tapes


def verify(executable, tapes):
    run = subprocess.run([executable], cwd=ROOT, input="\n".join(tapes), encoding="ascii",
                         capture_output=True, timeout=120)
    if run.returncode:
        raise SystemExit(f"Native hypothesis audit failed: {run.stderr.strip()}")
    return run.stdout


def verify_teacher(directory):
    from export_early_deadline_teacher_pairs import parse, public_record
    from measure_deadline_opening import RULES, build as build_teacher

    start, count = 42991, 1
    payload = " ".join(map(str, (start, count, *([0] * 10))))
    build_teacher(Path(directory))
    run = subprocess.run([Path(directory)/"deadline_opening.exe", RULES, "early-teacher-labels"],
                         input=payload, cwd=ROOT, capture_output=True, text=True, timeout=75)
    if run.returncode:
        raise SystemExit(f"Native teacher preflight failed: {run.stderr.strip()}")
    roots, teachers, branches, games = parse(run.stdout, start, count)
    labels = sum(len(public_record(key, root, teachers[key], branches[key], start)["branches"])
                 for key, root in roots.items())
    evaluated = sum(values[-2] > 0 for values in teachers.values())
    if not evaluated or not labels:
        raise SystemExit("Teacher preflight did not simulate or export any labeled branches.")
    print(f"PASS hypothesis teacher export games={games} roots={len(roots)} rollouts={evaluated} "
          f"unscored={len(roots)-evaluated} labels={labels}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="*")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixtures-only", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="also verify the actual teacher export")
    args = parser.parse_args()
    if args.fixtures_only or args.preflight:
        if args.sources:
            parser.error("fixture and preflight modes take no archive inputs")
        with TemporaryDirectory(prefix="junqi-hypothesis-") as directory:
            verify(build(directory), [])
            print("PASS hypothesis public reveal, rail turns, prior moves, control, isolation and rank inventory",
                  flush=True)
            if args.preflight:
                verify_teacher(directory)
        return
    if not args.sources:
        parser.error("provide local JGS inputs or --fixtures-only")
    total, tapes = read_inputs(args.sources)
    start = perf_counter()
    with TemporaryDirectory(prefix="junqi-hypothesis-") as directory:
        executable = build(directory)
        run = verify(executable, tapes)
        if verify(executable, tapes) != run:
            raise SystemExit("Archive audit did not reproduce on the same input.")
    cases = Counter()
    coverage = Counter()
    rows = run.splitlines()
    if len(rows) != len(tapes):
        raise SystemExit("The archive audit did not account for every input.")
    for row in rows:
        status, *counts = row.split()
        cases[status] += 1
        if status == "verified":
            coverage.update(dict(zip(COUNTS, map(int, counts), strict=True)))
    if not all(coverage[key] for key in COUNTS[2:6] + COUNTS[9:]):
        raise SystemExit("Archive audit did not exercise reveal, retirement, clock and reply history.")
    print(json.dumps({"files": total, "outside_size_range": total-len(tapes), "cases": cases,
                      "coverage": coverage, "repeat_identical": True,
                      "seconds_including_build_and_repeat": round(perf_counter()-start, 3)}, indent=2))


if __name__ == "__main__":
    main()
