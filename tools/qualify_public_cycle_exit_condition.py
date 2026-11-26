"""Test coverage and compare flag-plan continuation on the fixed development block."""
from collections import Counter, defaultdict
import json
import sys
import time

from audit_cycle_scoring_model import ROOT, inputs, read, recover_cases
from audit_public_attack_windows import write_json
from compare_archive_cycle_exit_values import panel, probe_output
from evidence_model import sha
from verify_archive_cycle_exit_options import compile_probe

OUT = ROOT / "build/public-cycle-exit-condition"
PROBE, RESULT = OUT / "archive-public-flag-condition.exe", OUT / "development.json"
MIN_DEVELOPMENT_FAMILIES = 12


def retrieve():
    old, weights, corpus, fingerprints = inputs()
    cases = recover_cases(old["selected"], corpus)
    return cases, weights, fingerprints


def measure(cases, weights):
    offers, rows = {}, defaultdict(list)
    for line in probe_output(cases, weights, PROBE, "public-condition"):
        tag, key, *values = line.split()
        if tag == "Q" and key not in offers:
            offers[key] = list(map(int, values))
        elif tag == "F":
            rows[key].append(list(map(int, values)))
        else:
            raise ValueError("duplicate or invalid development prefix/probe record")
    return offers, dict(rows)


def summarize(cases, offers, trials):
    if set(offers) != {m["key"] for m, _ in cases}:
        raise ValueError("development coverage does not reconcile")
    expected = [(v, n, p) for v in (0, 2) for n in range(12) for p in (0, 1)]
    eligible = {key for key, values in offers.items() if values[0] in (1, 2)}
    if set(trials) != eligible:
        raise ValueError("missing or extra planned-response comparisons")
    for key, (code, source, target, back_source, back_target) in offers.items():
        if code not in range(4):
            raise ValueError("unknown coverage classification")
        if code in (1, 2) and (source != back_source or target in (source, back_target)):
            raise ValueError("branches must use distinct quiet moves from the same piece")
        if key in eligible:
            for row in trials[key]:
                _, _, plan, a, b, attack_a, attack_b = row
                if a not in (-1, 0, 1, 2) or b not in (-1, 0, 1, 2) or max(attack_a, attack_b) > plan:
                    raise ValueError("invalid rollout result or attack plan usage")
            if [tuple(row[:3]) for row in trials[key]] != expected:
                raise ValueError("paired design missing cells")
    counts = Counter(str(values[0]) for values in offers.values())
    tables = {}
    for code in (1, 2):
        keys = [key for key in offers if offers[key][0] == code]
        if not keys:
            continue
        tables[str(code)] = {}
        for plan in (0, 1):
            paired = {k: [[v, n, a, b] for v, n, p, a, b, _, _ in trials[k] if p == plan] for k in keys}
            for variant in (0, 2):
                sample = [row for k in keys for row in trials[k] if row[0] == variant and row[2] == plan]
                tables[str(code)][f"{variant}.{plan}"] = dict(
                    **panel(keys, paired, variant), plan_taken=[sum(row[i] for row in sample) for i in (5, 6)])
    enough = min(counts["1"], counts["2"]) >= MIN_DEVELOPMENT_FAMILIES
    readiness = "eligible_for_independent_confirmation" if enough else "insufficient_development_coverage"
    return dict(counts=dict(counts), panels=tables, readiness=readiness,
                minimum_families_per_group=MIN_DEVELOPMENT_FAMILIES)


def audit():
    cases, weights, hashes = retrieve()
    saved = read(RESULT)
    if saved["inputs_sha256"] != hashes or saved["engine_sha256"] != sha(PROBE):
        raise ValueError("public-condition inputs changed")
    offers, trials = measure(cases, weights)
    if ((offers, trials) != (saved["offers"], saved["trials"]) or
            summarize(cases, offers, trials) != saved["summary"]):
        raise ValueError("development screen or continuation cells do not reproduce")
    print("PASS inspector: Observation-only positive and masked-flag controls, matched plans, fixed cohorts")
    print(json.dumps(saved["summary"], sort_keys=True))


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        return audit()
    cases, weights, hashes = retrieve()
    print("PASS reviewer: input lock, fixed development cohort, 12-family positive and matched-control gate", flush=True)
    compile_probe(PROBE)
    offers, trials = measure(cases, weights)
    summary = summarize(cases, offers, trials)
    write_json(RESULT, dict(inputs_sha256=hashes, engine_sha256=sha(PROBE),
                            offers=offers, trials=trials, summary=summary,
                            seconds=round(time.monotonic()-started, 3)))
    print(json.dumps(dict(counts=summary["counts"], readiness=summary["readiness"]), sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] == []:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: qualify_public_cycle_exit_condition.py [audit]")
