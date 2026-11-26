"""Check legal return choices and finite-horizon recurrence after human exits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time

from audit_archive_cycle_exit_decisions import RESULT as PRIOR, timeline
from audit_public_attack_windows import write_json
from audit_qq_archive import INPUT, ROOT, decode, initial_board, local_records, public_records
from compare_qq_rules import VC, tape
from partner_crossplay import model_roster
from test_defense_opportunity_cost import new_sources

RESULT = PRIOR.with_name("archive-cycle-exit-options.json")
EXE = RESULT.with_suffix(".exe")
FIELDS = ("ply", "seat", "actions", "return_legal", "return_quiet", "alias",
          "actual_probability", "return_probability", "heuristic_actual", "heuristic_return")


def recurrence(line, root, departure, horizon=3):
    turns, resumed, paired = 0, None, None
    previous = None
    for e in line:
        if e["index"] == departure["index"] and root["piece"] in e.get("removed", []):
            status = "loop_piece_removed_at_exit"
            break
        if e["index"] <= departure["index"]:
            continue
        if e["kind"] in ("terminal", "interruption"):
            status = e["kind"]
            break
        if root["piece"] in e.get("removed", []):
            status = "loop_piece_removed"
            break
        if e["kind"] != "move" or e["actor"] != root["actor"]:
            continue
        turns += 1
        on_edge = (not e["contact"] and e["piece"] == root["piece"] and
                   sorted([e["origin"], e["target"]]) == sorted([root["origin"], root["target"]]))
        if on_edge and resumed is None:
            resumed = dict(ply=e["ply"], index=e["index"], own_turn=turns)
        if on_edge and previous and previous["origin"] == e["target"] and paired is None:
            paired = dict(ply=e["ply"], index=e["index"], own_turn=turns)
        previous = e if on_edge else None
        if turns == horizon:
            status = "three_own_decisions"
            break
    else:
        status = "record_end"
    return dict(decisions=turns, stopped_by=status, edge_reused=resumed, opposite_pair=paired)


def selected():
    prior = json.loads(PRIOR.read_text(encoding="utf8"))
    paths = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    streams = (public_records(Path(paths["corpus"])), local_records(Path(paths["local"])), new_sources())
    cases, seen = [], set()
    for source, stream in zip(("public", "local", "curated"), streams, strict=True):
        eligible = {r["key"]: r for r in prior["rows"][source] if r["root"] and
                    r["trace"]["actor_exit"]["kind"] in ("quiet_same_piece", "quiet_other_piece", "contact_self")}
        for raw in stream:
            try:
                record = decode(raw)
            except ValueError:
                continue
            key, layouts, seats, events = record
            if key not in eligible or key in seen:
                continue
            seen.add(key)
            r = eligible[key]
            line, _ = timeline(layouts, seats, events)
            end = r["trace"]["actor_exit"]
            before = [e for e in line if e["index"] < end["index"] and
                      e["kind"] == "move" and e["actor"] == r["root"]["actor"]][-1]
            assert before["piece"] == r["root"]["piece"]
            meta = dict(key=key, source=source, layout=r["layout"], score=r["score"],
                        root=r["root"], departure=end, back=[*before["target"], *before["origin"]],
                        recurrence=recurrence(line, r["root"], end))
            cases.append((meta, record))
        assert set(eligible) <= seen
    return sorted(cases, key=lambda item: (item[0]["source"], item[0]["key"]))


def compile_probe(executable=EXE):
    sources = ("core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
               "tests/archive_cycle_exit_options.cpp")
    units = " ".join(f'"{ROOT.as_posix()}/{p}"' for p in sources)
    cmd = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT '
           f'/utf-8 /I"{ROOT.as_posix()}/core" {units} /Fe:{executable.name}')
    subprocess.run(f'cmd.exe /d /s /c "{cmd}"', cwd=executable.parent, check=True, timeout=75)


def native(cases, weights):
    # Frozen short-v1 stores five values; the current C++ array has one inactive slot.
    assert len(weights) == 5, "review policy schema before adapting another feature set"
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=EXE.parent) as feed:
        print(len(cases), file=feed)
        print(*weights, 0.0, file=feed)
        for meta, (_, layouts, seats, events) in cases:
            entries, ccw = tape(layouts, seats, events[:meta["departure"]["index"]+1])
            print(meta["key"], len(entries), ccw, *meta["back"], file=feed)
            for (x, y), (_, rank) in initial_board(layouts).items():
                print(x, y, rank, file=feed)
            for entry in entries:
                print(*entry, file=feed)
        feed.seek(0)
        output = subprocess.run([EXE], stdin=feed, capture_output=True, text=True, check=True, timeout=100).stdout
    result = {}
    for line in output.splitlines():
        tag, key, *values = line.split()
        assert key not in result
        if tag == "D":
            assert len(values) == len(FIELDS)
            result[key] = dict(zip(FIELDS, [int(v) for v in values[:6]] + [float(v) for v in values[6:]], strict=True))
        else:
            assert tag == "X" and not values
            result[key] = None
    assert set(result) == {r["key"] for r, _ in cases}
    return result


def summarize(rows):
    summary = {}
    for source in ("public", "local", "curated"):
        group = [r for r in rows if r["source"] == source]
        valid = [r for r in group if r["native"]]
        kinds = {}
        for kind in ("quiet_same_piece", "quiet_other_piece", "contact_self"):
            subset = [r for r in group if r["departure"]["kind"] == kind]
            kinds[kind] = dict(n=len(subset), edge_reused=sum(bool(r["recurrence"]["edge_reused"]) for r in subset),
                              opposite_pair=sum(bool(r["recurrence"]["opposite_pair"]) for r in subset),
                              outcomes=dict(Counter(str(r["score"]) for r in subset)))
        summary[source] = dict(selected=len(group), aligned=len(valid),
            return_legal=sum(r["native"]["return_legal"] for r in valid),
            return_quiet=sum(r["native"]["return_quiet"] for r in valid),
            same_features=sum(r["native"]["alias"] for r in valid),
            followup_stops=dict(Counter(r["recurrence"]["stopped_by"] for r in group)), by_departure=kinds)
    return summary


def audit(cases=None):
    runpy.run_path(str(ROOT / "tests/archive_cycle_resume_checks.py"))["check"]()
    data = json.loads(RESULT.read_text(encoding="utf8"))
    assert data["parent_sha256"] == hashlib.sha256(PRIOR.read_bytes()).hexdigest()
    frozen = model_roster()["candidate"]
    assert data["weights_sha256"] == frozen["artifact_sha256"]
    cases = selected() if cases is None else cases
    fresh = native(cases, frozen["weights"])
    expected = [dict(meta, native=fresh[meta["key"]]) for meta, _ in cases]
    assert data["rows"] == expected and data["summary"] == summarize(expected)
    for r in expected:
        n = r["native"]
        if n:
            assert n["ply"] == r["departure"]["ply"]-1 and n["seat"] == (r["root"]["actor"]+1) % 4
            assert n["return_quiet"] <= n["return_legal"] and 0 <= n["actual_probability"] <= 1
            assert not n["alias"] or abs(n["actual_probability"]-n["return_probability"]) < 1e-12
        follow = r["recurrence"]
        assert 0 <= follow["decisions"] <= 3
        assert (follow["stopped_by"] == "three_own_decisions") == (follow["decisions"] == 3)
    print("PASS frozen native replay, return options, same-input aliases and followup censoring")
    print(json.dumps(data["summary"], sort_keys=True))


def run():
    start = time.perf_counter()
    cases = selected()
    compile_probe()
    frozen = model_roster()["candidate"]
    answers = native(cases, frozen["weights"])
    rows = [dict(meta, native=answers[meta["key"]]) for meta, _ in cases]
    write_json(RESULT, dict(parent_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
        weights_sha256=frozen["artifact_sha256"], rows=rows, summary=summarize(rows),
        elapsed_seconds=time.perf_counter()-start))
    audit(cases)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
