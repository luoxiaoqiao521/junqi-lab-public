"""Test quiet route defense before a flag-front crisis, on fixed replay turns."""
from collections import Counter
import json
from pathlib import Path
from random import Random
import subprocess
import sys
import tempfile

from archive_action_screen import layout_group, select
from audit_public_attack_windows import OUT, write_json
from audit_qq_archive import INPUT, decode, initial_board, public_records
from compare_qq_rules import tape
from partner_crossplay import model_roster
from test_defense_opportunity_cost import EXE, RESULT as LATE, compile_probe, new_sources, parse, profile

RESULT = OUT / "pre-flag-route-defense.json"
PLIES = (72, 120)
ORDINARY = 48


def sources():
    archived = json.loads(LATE.read_text(encoding="utf-8"))
    excluded = {key.split("-")[1] for key in archived["decisions"]}
    for raw in new_sources():
        _, layouts, seats, events = decode(raw)
        yield "new-" + layout_group(layouts).hex()[:16], layouts, seats, events
    manifest = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    groups = select(public_records(Path(manifest["corpus"])))
    available = sorted(shape for shape in groups if shape.hex()[:16] not in excluded)
    for shape in Random(688194).sample(available, ORDINARY):
        _, layouts, seats, events = groups[shape]
        yield "qq-" + shape.hex()[:16], layouts, seats, events


def fixed_prefixes(events):
    step = 0
    for i, row in enumerate(events):
        step += row[0]>=0
        if row[0]>=0 and step in PLIES:
            yield step, events[:i+1]


def write_inputs(sink, cases, roster):
    print(len(cases), file=sink)
    print(*roster["weights"], file=sink)
    for key, board, prefix, direction in cases:
        print(key, len(prefix), direction, file=sink)
        for item in board+prefix:
            print(*item, file=sink)


def summarize(decisions, pairs):
    total = {}
    for label in ("new", "qq"):
        rows = {key: row for key, row in decisions.items() if key.startswith(label+"-")}
        aligned = {key: row for key, row in rows.items() if isinstance(row, list)}
        threats = [key for key, (rep, _) in aligned.items() if rep[3]]
        route = profile(3, aligned, pairs)
        total[label] = dict(sampled=len(rows), aligned=len(aligned),
                            too_short=list(rows.values()).count("too-short"),
                            mismatched=list(rows.values()).count(None),
                            possible_approach=len(threats),
                            approach_source_groups=len({key.rsplit("-",1)[0] for key in threats}),
                            available_route_groups=len({key.rsplit("-",1)[0] for key, row in aligned.items()
                                                        if row[1][3][0]>=0}), route=route)
    return total


def run():
    decisions, cases = {}, []
    for group, layouts, seats, events in sources():
        rows, direction = tape(layouts, seats, events)
        pieces = [(x,y,rank) for (x,y), (_,rank) in initial_board(layouts).items()]
        complete = dict(fixed_prefixes(rows))
        for ply in PLIES:
            key = f"{group}-{ply}"
            if ply in complete:
                cases.append((key,pieces,complete[ply],direction))
            else:
                decisions[key] = "too-short"
    roster = model_roster()["candidate"]
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as sink:
        write_inputs(sink, cases, roster)
        sink.seek(0)
        result = subprocess.run([EXE,"approach"],stdin=sink,capture_output=True,text=True,
                                check=True,timeout=110)
    checked, pairs = parse(result.stdout)
    assert set(checked) == {key for key, *_ in cases}
    decisions.update(checked)
    write_json(RESULT,dict(model=roster["artifact_sha256"],decisions=decisions,pairs=pairs,
                           summary=summarize(decisions, pairs)))
    audit()


def audit():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    decisions, pairs = data["decisions"], data["pairs"]
    assert Counter(key.split("-")[0] for key in decisions) == {"new":42, "qq":96}
    assert len({key.rsplit("-",1)[0] for key in decisions}) == 21+ORDINARY
    assert data["model"] == model_roster()["candidate"]["artifact_sha256"]
    for key, row in decisions.items():
        if not isinstance(row,list):
            assert key not in pairs
            continue
        prefix, actions = row
        assert prefix[0] == int(key.rsplit("-",1)[1]) and actions[2] == [-1]*4
        recorded, block = actions[0], actions[3]
        changed = block[0]>=0 and block[:2]!=recorded[:2]
        if block[0]>=0:
            assert prefix[3]>0
        assert len(pairs.get(key,[])) == (24 if changed else 0)
        if changed:
            modes = {(v,s) for v,s,*_ in pairs[key]}
            assert modes == {(v,s) for v in (0,2) for s in range(12)}
            assert all(len(row)==7 and row[5] in (-1,0,1,2) for row in pairs[key])
    assert data["summary"] == summarize(decisions,pairs)
    print(json.dumps(data["summary"], sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] == ["audit"]:
        audit()
    else:
        compile_probe()
        run()
