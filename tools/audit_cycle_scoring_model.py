"""Reconcile the archived A/B before changing known-contact scoring or encoding."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import time

from audit_public_attack_windows import write_json
from audit_qq_archive import INPUT, ROOT, decode, public_records
from compare_archive_cycle_exit_values import PRIOR, RESULT, execute, summarized
from evidence_model import WIDTHS, load, save, sha
from partner_crossplay import ARCHIVE, FILES, model_roster
from verify_archive_cycle_exit_options import compile_probe

OUT = ROOT / "build/cycle-scoring-model-audit"
PROBE = OUT / "archive-collision-audit.exe"
RECEIPT = OUT / "result.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inputs():
    old = read(RESULT)
    roster = model_roster()
    corpus = Path(read(INPUT)["corpus"])
    paths = (INPUT, corpus, PRIOR, RESULT, ARCHIVE / "junqi.exe",
             ARCHIVE / "ruleset.json", *(ARCHIVE / f for f in FILES.values()))
    hashes = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name): sha(path)
              for path in paths}
    if old["prior_sha256"] != sha(PRIOR) or old["weights_sha256"] != roster["candidate"]["artifact_sha256"]:
        raise ValueError("historical input mismatch")
    return old, roster["candidate"]["weights"], corpus, hashes


def recover_cases(keys, corpus):
    required = set(keys)
    metadata = {row["key"]: row for row in read(PRIOR)["rows"] if row["key"] in required}
    if (len(keys) != 64 or len(required) != 64 or set(metadata) != required
            or len({row["layout"] for row in metadata.values()}) != 64
            or not all(row["source"] == "public" and row["native"]["return_quiet"]
                       and row["departure"]["kind"].startswith("quiet") for row in metadata.values())):
        raise ValueError("archived comparison is not 64 distinct eligible layouts")
    records = {}
    for raw in public_records(corpus):
        try:
            record = decode(raw)
        except ValueError:
            continue
        if record[0] in required:
            records[record[0]] = record
    if set(records) != required:
        raise ValueError("archived source record missing")
    return [(metadata[key], records[key]) for key in keys]


def check_loaders():
    engine, rules = ARCHIVE / "junqi.exe", ARCHIVE / "ruleset.json"
    with tempfile.TemporaryDirectory(dir=OUT) as directory:
        folder = Path(directory)
        rejected = []
        for encoding, width in WIDTHS.items():
            path = folder / f"{encoding}.json"
            save(path, [0.0] * width, executable=engine, rules=rules, encoding=encoding)
            load(path, engine, rules, encoding)
            for expected in WIDTHS.keys() - {encoding}:
                expect_reject(path, engine, rules, expected)
                rejected.append([encoding, expected])
        valid = read(folder / "short-v1.json")
        for field, replacement in (("schema", 2), ("weights", [0] * 6),
                                   ("weights", [0, 0, 0, 0, float("nan")]),
                                   ("rules_sha256", "0" * 64), ("engine_sha256", "0" * 64)):
            invalid = folder / "invalid.json"
            invalid.write_text(json.dumps(dict(valid, **{field: replacement})), encoding="utf8")
            expect_reject(invalid, engine, rules, "short-v1")
    return sorted(rejected)


def expect_reject(path, engine, rules, expected):
    try:
        load(path, engine, rules, expected)
    except ValueError:
        return
    raise ValueError(f"incompatible model accepted as {expected}")


def evaluate(old, cases, pairs):
    if set(pairs) != set(old["selected"]):
        raise ValueError("missing or extra continuation cases")
    changes = Counter()
    for key, samples in pairs.items():
        original = old["pairs"][key]
        if ([(v, n) for v, n, _, _ in samples] != [(v, n) for v, n, _, _ in original] or
                len(samples) != 24):
            raise ValueError("paired RNG coordinate mismatch")
        for (v, _, a, b), (_, _, before_a, before_b) in zip(samples, original, strict=True):
            if a not in (-1, 0, 1, 2) or b not in (-1, 0, 1, 2):
                raise ValueError("invalid team outcome")
            if (a, b) != (before_a, before_b):
                if v == 0:
                    raise ValueError("random continuation control changed")
                changes["pair_outcomes_changed"] += 1
                changes["individual_outcomes_changed"] += int(a != before_a) + int(b != before_b)
    return summarized(cases, pairs), dict(changes)


def audit():
    old, _, corpus, hashes = inputs()
    receipt = read(RECEIPT)
    if receipt["inputs_sha256"] != hashes or receipt["probe_sha256"] != sha(PROBE):
        raise ValueError("frozen inputs or measured binary changed")
    cases = recover_cases(old["selected"], corpus)
    if receipt["encoding_rejections"] != check_loaders():
        raise ValueError("encoding rejection matrix changed")
    summary, changes = evaluate(old, cases, receipt["corrected_pairs"])
    if (receipt["corrected_summary"] != summary or receipt["changes"] != changes
            or receipt["historical_summary"] != old["summary"]):
        raise ValueError("stored conclusions do not match the paired evidence")
    print("PASS inspector: frozen inputs, original panel, corrected pairs, random negative control, model compatibility")
    print(json.dumps(dict(before=old["summary"]["all"]["2"], after=summary["all"]["2"],
                          changes=changes, seconds=receipt["seconds"]), sort_keys=True))


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists():
        return audit()
    old, weights, corpus, hashes = inputs()
    cases = recover_cases(old["selected"], corpus)
    rejections = check_loaders()
    print("PASS reviewer: fixed historical 64-layout cohort and six incompatible model-load cases", flush=True)
    compile_probe(PROBE)
    if execute(cases, weights, PROBE) != (old["decisions"], old["pairs"]):
        raise ValueError("historical continuation does not reproduce cell for cell")
    decisions, pairs = execute(cases, weights, PROBE, "collision-pairs")
    if decisions != old["decisions"]:
        raise ValueError("initial observation or historical root diagnostics changed")
    summary, changes = evaluate(old, cases, pairs)
    write_json(RECEIPT, dict(inputs_sha256=hashes, probe_sha256=sha(PROBE),
                             encoding_rejections=rejections, corrected_pairs=pairs,
                             historical_summary=old["summary"], corrected_summary=summary,
                             changes=changes, seconds=round(time.monotonic()-started, 3)))
    print("PASS execution: historical reproduction and collision-aware sensitivity evidence")


if __name__ == "__main__":
    if sys.argv[1:] == []:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: audit_cycle_scoring_model.py [audit]")
