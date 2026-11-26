"""Paired move guidance on held-out, strictly legal real replay windows."""
import shutil
import subprocess
import sys
import time

from archive_pressure_training import (ROOT, RULES, EXE, PARENT, ARCHIVE, ArchiveTrainingStopped,
                                       source_records, feed, compare_rows, read)
from evidence_model import load, sha

REFERENCE = ROOT / "build/archive-pressure-balanced"
OUT = ROOT / "build/archive-pressure-practice"
SEED = 850700


def inspect(rows, cases):
    expected = [(i, c["seat"], table, rep) for i, c in enumerate(cases)
                for table in (0, 4) for rep in range(12)]
    if [(r["root"], r["seat"], r["table"], r["rep"]) for r in rows] != expected:
        raise ValueError("practice grid does not contain both branches of every matched stream")
    if any(r["pressure"]["move"] == r["quiet"]["move"] for r in rows):
        raise ValueError("practice must compare distinct legal candidates")
    result = {}
    for label, table in (("random", 0), ("contact_enemy", 4)):
        selected = [r for r in rows if r["table"] == table]
        left = [dict(r, score=r["quiet"]["score"], contact=0, hq_pressure=0) for r in selected]
        right = [dict(r, score=r["pressure"]["score"], contact=1, hq_pressure=1) for r in selected]
        result[label] = compare_rows(left, right, cases)
    ranges = [panel["conservative_interval"] for panel in result.values()]
    verdict = "insufficient"
    if all(low > 0 for low, _ in ranges):
        verdict = "supported_in_both_tables"
    elif all(high < 0 for _, high in ranges):
        verdict = "rejected_in_both_tables"
    return dict(panels=result, verdict=verdict)


def audit(directory, reference):
    result = read(directory / "summary.json")
    assert result["profile"] == "archive-pressure-practice" and result["status"] == "complete"
    assert result["reference_sha256"] == sha(reference / "summary.json")
    assert result["engine_sha256"] == sha(directory / "junqi.exe")
    assert result["rules_sha256"] == sha(directory / "ruleset.json")
    assert result["parent_sha256"] == sha(PARENT)
    assert result["guidance"] == inspect(result["pairs"], result["sources"])
    print("PASS paired real-replay guidance:", result["guidance"]["verdict"], flush=True)
    for table, panel in result["guidance"]["panels"].items():
        low, high = panel["bounds"]
        ci_low, ci_high = panel["conservative_interval"]
        print(f"{table} | {panel['pairs']} pairs | team-score range [{low:+.5f}, {high:+.5f}] | "
              f"layout interval [{ci_low:+.5f}, {ci_high:+.5f}] | "
              f"rescued {panel['rescued']}, new losses {panel['new_losses']}", flush=True)


def gather(report):
    cases, _, census = source_records()
    if census != report["corpus_sha256"]:
        raise ValueError("reference corpus fingerprint changed")
    sources = [r for r in report["evaluation_sources"] if r["group"] == "pressure"]
    lookup = {case[0]["family"]: case for case in cases if case[0]["fold"] == "holdout"}
    roots = [lookup[s["family"]] for s in sources]
    for source, (meta, _, _) in zip(sources, roots, strict=True):
        if any(source[k] != v for k, v in meta.items()):
            raise ValueError("reference window changed")
    if not 1 <= len(roots) <= 32:
        raise ValueError("no held-out pressure fixtures or practice budget exceeded")
    return sources, roots


def run(directory, reference, hours, audit_only, command, write):
    if audit_only or (directory / "summary.json").exists():
        return audit(directory, reference)
    state = dict(profile="archive-pressure-practice", status="running", phase="practice",
                 started_utc=time.time(), hours_limit=hours)
    path = directory / "progress.json"
    write(path, state)
    try:
        measure(directory, reference, hours, command, write)
    except Exception as error:
        status = "stopped" if isinstance(error, ArchiveTrainingStopped) else "failed"
        write(path, dict(state, status=status, error=str(error)))
        raise
    write(path, dict(state, status="complete", phase="done"))


def measure(directory, reference, hours, command, write):
    started = time.monotonic()
    if (directory / "STOP").exists():
        raise ArchiveTrainingStopped("practice stopped before launch")
    manifest = read(reference / "summary.json")
    if manifest["status"] != "complete" or manifest["rules_sha256"] != sha(RULES):
        raise ValueError("practice requires a completed course under the current selfplay rules")
    parent = load(PARENT, ARCHIVE / "junqi.exe", ARCHIVE / "ruleset.json", "short-v1")
    if sha(PARENT) != manifest["parent_sha256"]:
        raise ValueError("practice reference parent changed")
    sources, roots = gather(manifest)
    print(f"paired practice on {len(roots)} held-out replay windows", flush=True)
    engine, rules = directory / "junqi.exe", directory / "ruleset.json"
    for original, snapshot in ((EXE, engine), (RULES, rules)):
        if not snapshot.exists():
            shutil.copyfile(original, snapshot)
        elif sha(snapshot) != sha(original):
            raise ValueError(f"practice snapshot diverged from {original.name}")
    if (directory / "STOP").exists():
        raise ArchiveTrainingStopped("practice stopped at phase boundary")
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    rows = command(directory, started+hours*3600, metrics, "archive-curriculum", rules,
                   "practice", input_text=feed(roots, parent["weights"], SEED))
    result = dict(status="complete", profile="archive-pressure-practice",
                  source_roots=len(roots), sources=sources, pairs=rows, seed=SEED,
                  rules_sha256=sha(rules), engine_sha256=sha(engine), parent_sha256=sha(PARENT),
                  reference_sha256=sha(reference / "summary.json"), guidance=inspect(rows, sources),
                  seconds=round(time.monotonic()-started,3), engine_metrics=metrics)
    write(directory / "summary.json", result)
    audit(directory, reference)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        sys.exit("usage: archive_pressure_practice.py [audit]")
    cli = [sys.executable, "-X", "utf8", str(ROOT / "tools/bounded_training.py"),
           "--profile", "archive-pressure-practice", "--directory", str(OUT), "--hours", "1"]
    if sys.argv[1:]:
        cli.append("--audit")
    subprocess.run(cli, check=True, timeout=150)
