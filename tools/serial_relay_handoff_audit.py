"""Audit fixed relay contacts and the effect of accepting them at the receiving seat."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from statistics import mean
from time import perf_counter

from audit_human_timing import sanitized_tape
from audit_qq_archive import LIMIT, ROOT, decode
from compare_qq_rules import VC, tape
from evidence_model import load, sha
from serial_worker import read, write_receipt

INDEX = ROOT.parent / "junqi-lab-qq-rule-evidence/build/human-timing/audit.json"
MANIFEST = ROOT / "build/relay-handoff-input-20261125/manifest.json"
RULES = ROOT / "config/ruleset-selfplay.json"
ARCHIVE = ROOT / "build/tuned-policy-confirmation-20261118"
PEER = ARCHIVE / "default-162000.json"
ARCHIVE_ROOT = Path(os.environ.get("JUNQI_LOCAL_ARCHIVE_ROOT", ROOT / "data/local-archive"))
SOURCES = {"art": ARCHIVE_ROOT / "annotated", "replays": ARCHIVE_ROOT / "replays"}
CONTROLS = {("1b125aef942a284f", 58): (1, 1, 1, 0, 1),
            ("ad81de8b7f44b668", 104): (1, 0, 0, 0, -1),
            ("ad81de8b7f44b668", 142): (1, 0, 0, 1, -1)}
STRATA = {"attacker_won": 8, "mutual": 6, "available_unused": 12, "closed": 8}
H_FIELDS = ("opened", "reachable", "human_took", "target_changed", "contact_result",
            "reply_count", "reply_stays", "reply_blocks", "reply_changes", "reply_skips", "root_hash")
P_INT = ("table", "stream", "ordinary", "contact", "on_shortlist", "ordinary_score", "contact_score",
         "ordinary_steps", "contact_steps", "ordinary_hash", "contact_hash")
P_LAST = ("ordinary_teacher", "contact_teacher", "ordinary_work", "contact_work")
TABLES = ("flag-route", "changed_receiver", "changed_enemies")


def select_cases(windows):
    controls = [r for r in windows if r["kind"] == "relay" and (r["key"], r["ply"]) in CONTROLS]
    if len(controls) != len(CONTROLS):
        raise ValueError("missing pinned archive controls")
    old = ROOT / "build/relay-chain/audit-relay-opening-and-adaptive-continuation-a4af7fea33ed9c445687"
    prior = {row["family"] for row in read(old / "development.json")["cases"].values()}
    used = prior | {r["family"] for r in controls}
    eligible = [r for r in windows if r["kind"] == "relay" and r["split"] == "holdout" and
                r["visible_to_opener"] and r["decision"] == r["ply"] + 2 and r["ply"] <= 278]
    order = lambda r: hashlib.sha256(f'handoff-v1/{r["family"]}/{r["key"]}/{r["ply"]}'.encode()).digest()
    confirmation = []
    for label, size in STRATA.items():
        selected = []
        for row in sorted((r for r in eligible if r["status"] == label), key=order):
            if row["family"] not in used and len(selected) < size:
                selected.append(row)
                used.add(row["family"])
        confirmation.extend(selected)
    return controls, confirmation, len(prior)


def read_tapes(cases):
    wanted = {r["key"] for r in cases}
    result = {}
    for root in SOURCES.values():
        if not root.is_dir():
            raise ValueError("missing declared local archive root")
        for file in sorted(root.rglob("*.jgs")):
            if file.stat().st_size > LIMIT:
                continue
            raw = file.read_bytes()
            try:
                key, layouts, seats, events = decode(raw)
            except ValueError:
                continue
            short_key = key[:16]
            if short_key in wanted:
                _, ccw = tape(layouts, seats, events)
                result[short_key] = [ccw, sanitized_tape(raw, events)]
                wanted.remove(short_key)
    if wanted:
        raise ValueError("pinned source traces are no longer available")
    return result


def prepare():
    controls, confirmation, excluded = select_cases(read(INDEX)["windows"])
    if len(confirmation) < 20:
        raise ValueError("too few independent source families")
    traces = read_tapes(controls + confirmation)
    model = load(PEER, ARCHIVE / "junqi.exe", ARCHIVE / "ruleset.json", "short-v1")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(index_sha256=sha(INDEX), rule_sha256=sha(RULES), peer_sha256=sha(PEER),
                    weights=model["weights"] + [0], prior_families_excluded=excluded,
                    controls=controls, confirmation=confirmation, traces=traces)
    write_receipt(MANIFEST, manifest)
    print(json.dumps(dict(controls=len(controls), families=len(confirmation),
                          strata=Counter(r["status"] for r in confirmation), excluded=excluded)))


def compile_probe(folder):
    files = " ".join(f'"{ROOT.as_posix()}/{p}"' for p in
                     ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
                      "tests/relay_handoff_audit.cpp"))
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'{files} /Fe:relay_handoff.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=folder, check=True, timeout=110)


def run_cases(folder, manifest, cohort):
    lines = [" ".join(map(str, manifest["weights"]))]
    cases = {}
    for row in manifest[cohort]:
        trace, ply = row["key"], row["ply"]
        case = f"{trace}-{ply}"
        ccw, hex_tape = manifest["traces"][trace]
        fields = ("ply", "decision", "receiver", "open_from", "open_to", "origin", "target")
        lines.append(" ".join(map(str, [case, ccw, *(row[f] for f in fields), hex_tape])))
        cases[case] = row
    run = subprocess.run([folder / "relay_handoff.exe", RULES], input="\n".join(lines)+"\n",
                         capture_output=True, encoding="utf8", timeout=125)
    if run.returncode:
        raise RuntimeError(run.stderr.strip() or "native probe exited")
    (folder / (cohort + ".txt")).write_text(run.stdout, encoding="utf8")
    return parse(cases, run.stdout)


def parse(cases, raw):
    output = {case: dict(root=root, pairs=[]) for case, root in cases.items()}
    for line in raw.splitlines():
        if not line.startswith(("H ", "A ", "P ")):
            continue
        tag, case, *values = line.split()
        data = output[case]
        if tag == "H":
            data["handoff"] = dict(zip(H_FIELDS, map(int, values), strict=True))
        elif tag == "A":
            matched, reported, reason = values
            data.update(verified=int(matched), reported=bool(int(reported)), reason=reason)
        else:
            integers = dict(zip(P_INT, map(int, values[:len(P_INT)]), strict=True))
            remaining = [convert(value) for convert, value in
                         zip((float, float, int, int), values[len(P_INT):], strict=True)]
            data["pairs"].append(integers | dict(zip(P_LAST, remaining, strict=True)))
    for data in output.values():
        if "verified" not in data:
            raise ValueError("archive run omitted a declared root")
        metrics, pairs = data.get("handoff", {}), data["pairs"]
        expected = [(table, stream) for table in range(len(TABLES)) for stream in range(4)]
        if metrics and sum(metrics[field] for field in H_FIELDS[6:10]) != metrics["reply_count"]:
            raise ValueError("intervening action census does not balance")
        if [(p["table"], p["stream"]) for p in pairs] != (expected if metrics.get("reachable") else []):
            raise ValueError("conditional continuation matrix is incomplete")
    return output


def bounds(pair):
    left, right = pair["ordinary_score"], pair["contact_score"]
    if pair["ordinary"] == pair["contact"]:
        return 0, 0
    return ((right if right >= 0 else 0) - (left if left >= 0 else 2),
            (right if right >= 0 else 2) - (left if left >= 0 else 0))


def uncertainty(families):
    if not families:
        return None
    rng = random.Random(271159)
    values = list(families.values())
    low, high = [], []
    for _ in range(4096):
        block = rng.choices(values, k=len(values))
        low.append(mean(pair[0] for pair in block) * 50)
        high.append(mean(pair[1] for pair in block) * 50)
    return [round(sorted(low)[102], 4), round(sorted(high)[3993], 4)]


def table_summary(rows, table):
    families, changed = {}, set()
    counts = Counter()
    for data in rows:
        group = [p for p in data["pairs"] if p["table"] == table]
        if not group:
            continue
        family = data["root"]["family"]
        if family in families:
            raise ValueError("duplicate layout family in confirmation")
        families[family] = tuple(map(mean, zip(*(bounds(p) for p in group), strict=True)))
        for pair in group:
            counts["pairs"] += 1
            counts["nominated"] += pair["on_shortlist"]
            if pair["ordinary"] == pair["contact"]:
                counts["already_selected"] += 1
                continue
            changed.add(family)
            delta = bounds(pair)
            if min(pair["ordinary_score"], pair["contact_score"]) < 0:
                counts["censored_changed"] += 1
            teacher = pair["contact_teacher"] - pair["ordinary_teacher"]
            counts["teacher_prefers_contact"] += teacher > 1e-12
            counts["confirmed_improvement"] += delta[0] > 0
            counts["confirmed_harm"] += delta[1] < 0
            counts["teacher_misses_improvement"] += delta[0] > 0 and teacher <= 1e-12
            counts["teacher_prefers_harm"] += delta[1] < 0 and teacher > 1e-12
    return dict(counts, distinct_changed_families=len(changed), reachable_families=len(families),
                bounds=[round(mean(v[i] for v in families.values())*50, 4) for i in range(2)] if families else None,
                family_interval=uncertainty(families))


def summarize(results):
    verified = [r for r in results.values() if r["reason"] == "matched" and r["reported"] and
                r["verified"] == r["root"]["decision"]]
    if any(r["pairs"] for r in results.values() if r not in verified):
        raise ValueError("continuations were generated from a nonmatching replay")
    tables = {name: table_summary(verified, i) for i, name in enumerate(TABLES)}
    strata = {status: dict(declared=sum(r["root"]["status"] == status for r in results.values()),
                           verified=sum(r["root"]["status"] == status for r in verified),
                           opened=sum(r["root"]["status"] == status and r["handoff"]["opened"] for r in verified),
                           reachable=sum(r["root"]["status"] == status and r["handoff"]["reachable"] for r in verified))
              for status in STRATA}
    supported = all(s["distinct_changed_families"] >= 8 and s["family_interval"][0] > 0
                    for s in tables.values())
    refuted = all(s["distinct_changed_families"] >= 8 and s["family_interval"][1] < 0
                  for s in tables.values())
    verdict = "supported" if supported else "refuted" if refuted else "inconclusive"
    return dict(strata=strata, divergences=Counter(r["reason"] for r in results.values() if r not in verified),
                tables=tables, verdict=verdict)


def main(audit):
    begun = perf_counter()
    folder = ROOT / "build/relay-handoff" / os.environ["JUNQI_SERIAL_KEY"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read(MANIFEST)
    if manifest["rule_sha256"] != sha(RULES):
        raise ValueError("frozen rules changed")
    if not audit:
        compile_probe(folder)
    if audit:
        control_receipt = read(folder / "controls.json")
        if control_receipt["engine_sha256"] != sha(folder / "relay_handoff.exe"):
            raise ValueError("native probe changed after positive and negative controls")
    cohort = "confirmation" if audit else "controls"
    results = run_cases(folder, manifest, cohort)
    if audit:
        summary = summarize(results)
    else:
        for data in results.values():
            key = (data["root"]["key"], data["root"]["ply"])
            if (data["verified"] != data["root"]["decision"] or data["reason"] != "matched" or
                    not data["reported"] or tuple(data["handoff"][field] for field in H_FIELDS[:5]) != CONTROLS[key]):
                raise ValueError(f"pinned positive/negative control diverged: {key} {data.get('handoff')} "
                                 f'{data["reason"]} at {data["verified"]}')
        summary = {case: data["handoff"] for case, data in results.items()}
    write_receipt(folder / (cohort + ".json"),
                  dict(engine_sha256=sha(folder / "relay_handoff.exe"), seconds=round(perf_counter()-begun, 3),
                       summary=summary, cases=results))
    print(json.dumps(summary, sort_keys=True))
    if audit:
        print("JUNQI_RESEARCH " + summary["verdict"])
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["smoke"]:
        folder = ROOT / "build/relay-handoff/smoke"
        folder.mkdir(parents=True, exist_ok=True)
        compile_probe(folder)
        results = run_cases(folder, read(MANIFEST), "controls")
        print(json.dumps({case: {k: v for k, v in data.items() if k not in ("pairs", "root")}
                          for case, data in results.items()}, sort_keys=True))
    elif sys.argv[1:] in ([], ["audit"]):
        main(bool(sys.argv[1:]))
    else:
        raise ValueError("expected prepare, smoke, audit or no argument")
