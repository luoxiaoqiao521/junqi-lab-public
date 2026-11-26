"""Trace what follows the first confirmed human quiet backtracking prefix.

Archive events describe what happened, not whether the alternative was better.
The first external change and the looping actor's departure are separate clocks.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time

from audit_public_attack_windows import write_json
from audit_qq_archive import INPUT, archived_move, decode, initial_board, local_records, public_records
from check_archive_local_cycle_timing import RESULT as TIMING, COHORTS, replay
from test_defense_opportunity_cost import new_sources

RESULT = TIMING.with_name("archive-cycle-exits.json")


def move_event(board, tokens, raw, index, ply):
    origin, target = (raw[2], 16-raw[3]), (raw[4], 16-raw[5])
    piece, victim = tokens.get(origin), tokens.get(target)
    actor, contact = archived_move(board, raw)
    outcome = raw[1] & 3
    del tokens[origin]
    if outcome <= 1:
        tokens[target] = piece
    elif outcome == 3:
        del tokens[target]
    removed = ([piece] if outcome in (2, 3) else [])
    removed += [victim] if victim is not None and outcome in (1, 3) else []
    return dict(index=index, ply=ply, kind="move", actor=actor, piece=piece,
                origin=list(origin), target=list(target), contact=contact, removed=removed)


def timeline(layouts, seats, events):
    board = initial_board(layouts)
    tokens = {point: i for i, point in enumerate(sorted(board))}
    active, rows, ends = set(range(4)), [], []
    ply = 0
    for index, raw in enumerate(events):
        if raw[0] == 95:
            if ends:
                raise ValueError("moves_after_end")
            ply += 1
            rows.append(move_event(board, tokens, raw, index, ply))
        elif raw[0] == 245:
            kind, code, cause = raw[1:4]
            item = dict(index=index, ply=ply, actor=-1, cause=cause)
            if kind == 5:
                ends.append(cause)
                rows.append(dict(item, kind="terminal"))
            elif kind == 2:
                rows.append(dict(item, kind="interruption"))
            elif kind in (3, 4) and code in seats and seats.index(code) in active:
                actor = seats.index(code)
                active.remove(actor)
                gone = [p for p, v in board.items() if v[0] == actor]
                rows.append(dict(item, kind="retirement", removed=[tokens[p] for p in gone]))
                rows[-1]["actor"] = actor
                for point in gone:
                    del board[point]
                    del tokens[point]
        else:
            raise ValueError("event_type")
    if len(ends) != 1 or ends[0] not in (2, 4):
        raise ValueError("unresolved_terminal")
    teams = {s % 2 for s in active}
    winner = next(iter(teams)) if ends[0] == 4 and len(teams) == 1 else None
    return rows, winner


def relation(actor, focal):
    return "self" if actor == focal else "partner" if actor % 2 == focal % 2 else "enemy"


def landmark(event, label):
    return dict(index=event["index"], ply=event["ply"], actor=event["actor"], kind=label)


def prefix_root(events, at):
    root = next(e for e in events if e["kind"] == "move" and e["ply"] == at)
    own = [e for e in events if e["index"] <= root["index"] and
           e["kind"] == "move" and e["actor"] == root["actor"]][-4:]
    assert len(own) == 4 and len({e["piece"] for e in own}) == 1
    assert all(a["origin"] == b["target"] and a["target"] == b["origin"]
               for a, b in zip(own, own[1:]))
    assert all(e["kind"] == "move" and not e["contact"] for e in events
               if own[0]["index"] <= e["index"] <= root["index"])
    return dict(actor=root["actor"], piece=root["piece"], start=own[0]["ply"],
                confirmed=at, index=root["index"], origin=root["origin"], target=root["target"])


def changed(event, root, origin, target):
    if event["kind"] != "move":
        return event["kind"] + ("_" + relation(event["actor"], root["actor"])
                                  if event["kind"] == "retirement" else "")
    if event["contact"]:
        return "contact_" + relation(event["actor"], root["actor"])
    if event["actor"] == root["actor"] and not (
            event["piece"] == root["piece"] and event["origin"] == target and event["target"] == origin):
        return "quiet_same_piece" if event["piece"] == root["piece"] else "quiet_other_piece"
    return None


def follow(events, root):
    origin, target = root["origin"], root["target"]
    first = departure = None
    repeats = repeats_after_change = 0
    for event in events:
        if event["index"] <= root["index"]:
            continue
        label = changed(event, root, origin, target)
        if label and first is None:
            first = landmark(event, label)
        if root["piece"] in event.get("removed", []) and event["actor"] != root["actor"]:
            departure = landmark(event, "loop_piece_removed")
        elif event["kind"] in ("terminal", "interruption") or (
                event["kind"] == "retirement" and event["actor"] == root["actor"]):
            departure = landmark(event, label)
        elif event["kind"] == "move" and event["actor"] == root["actor"]:
            if label:
                departure = landmark(event, label)
            else:
                repeats += 1
                repeats_after_change += first is not None
                origin, target = event["origin"], event["target"]
        if departure:
            break
    return dict(first_change=first, actor_exit=departure, continued_returns=repeats,
                returns_after_change=repeats_after_change)


def collect():
    old = json.loads(TIMING.read_text(encoding="utf8"))
    paths = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    streams = (public_records(Path(paths["corpus"])), local_records(Path(paths["local"])), new_sources())
    records, seen = {}, set()
    for source, stream in zip(COHORTS, streams, strict=True):
        wanted = {r["key"]: r for r in old["rows"][source]}
        results = []
        for raw in stream:
            try:
                key, layouts, seats, events = decode(raw)
            except ValueError:
                continue
            if key in seen or key not in wanted:
                continue
            seen.add(key)
            prior = wanted[key]
            measured = replay(layouts, seats, events)
            assert all(prior[k] == v for k, v in measured.items())
            line, winner = timeline(layouts, seats, events)
            root = prefix_root(line, prior["quiet_at"]) if prior["quiet_at"] else None
            outcome = None if root is None or winner is None else int(winner == root["actor"] % 2)
            if prior["end"] == 2 and root:
                outcome = 0.5
            results.append(dict(key=key, layout=prior["layout"], plies=prior["plies"],
                                end=prior["end"], score=outcome, root=root,
                                trace=follow(line, root) if root else None))
        assert {r["key"] for r in results} == set(wanted), "frozen source coverage changed"
        records[source] = sorted(results, key=lambda r: r["key"])
    return records


def counts(rows):
    return dict(Counter("unknown" if r["score"] is None else str(r["score"]) for r in rows))


def summarize(rows):
    result = {}
    for source, games in rows.items():
        cycles = [r for r in games if r["root"]]
        first = Counter(r["trace"]["first_change"]["kind"] for r in cycles)
        exits = sorted({r["trace"]["actor_exit"]["kind"] for r in cycles})
        by_exit = {}
        for kind in exits:
            group = [r for r in cycles if r["trace"]["actor_exit"]["kind"] == kind]
            by_exit[kind] = dict(n=len(group), outcome=counts(group), delay_median=median(
                r["trace"]["actor_exit"]["ply"] - r["root"]["confirmed"] for r in group))
        after = [r for r in cycles if r["trace"]["returns_after_change"]]
        selected = sorted(cycles, key=lambda r: r["key"])
        balanced = list({r["layout"]: r for r in reversed(selected)}.values())
        result[source] = dict(games=len(games), cycles=len(cycles), first_change=dict(first),
                              actor_exit=by_exit, continued_after_change=len(after),
                              continued_after_change_outcomes=counts(after),
                              layout_groups=len(balanced), cycle_outcomes=counts(cycles))
    return result


def audit(rescan=True):
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests/archive_cycle_exit_checks.py"))["check"]()
    data = json.loads(RESULT.read_text(encoding="utf8"))
    assert data["timing_sha256"] == hashlib.sha256(TIMING.read_bytes()).hexdigest()
    if rescan:
        assert data["rows"] == collect()
    assert data["summary"] == summarize(data["rows"])
    for games in data["rows"].values():
        for r in games:
            if r["root"]:
                a, b = r["trace"]["first_change"], r["trace"]["actor_exit"]
                assert r["root"]["index"] < a["index"] <= b["index"]
                assert r["root"]["confirmed"] <= a["ply"] <= b["ply"] <= r["plies"]
    print("PASS archived prefixes, timeline order, outcomes and counterexample fixtures")
    print(json.dumps(data["summary"], sort_keys=True))


def run():
    start = time.perf_counter()
    rows = collect()
    write_json(RESULT, dict(timing_sha256=hashlib.sha256(TIMING.read_bytes()).hexdigest(),
                           rows=rows, summary=summarize(rows), elapsed_seconds=time.perf_counter()-start))
    audit(False)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
