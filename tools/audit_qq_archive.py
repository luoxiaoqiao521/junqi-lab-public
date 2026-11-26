"""Summarize local QQ replay evidence without exporting player fields."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "build/jgs-evidence-input.json"
OUTPUT = ROOT / "build/qq-evidence/archive.json"
SEAT_BLOCKS = (60, 148, 236, 324)
INVENTORY = Counter({0: 5, 2: 1, 3: 3, 4: 2, 5: 1, 6: 1,
                     7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3})
CAMPS = {(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)}
LIMIT = 65536


def grid(seat, row, col):
    return ((11+row, 6+col), (6+col, 5-row), (5-row, 10-col), (10-col, 11+row))[seat]


HQ = {grid(seat, 5, col) for seat in range(4) for col in (1, 3)}


def valid_grid(layout):
    return (len(layout) == 30 and Counter(layout) == INVENTORY and
            all((layout[5*r+c] == 0) == ((r, c) in CAMPS) for r in range(6) for c in range(5)))


def decode(raw):
    if len(raw) < 412 or not raw.startswith(b"QQGame JQS"):
        raise ValueError("signature")
    layouts = [raw[i:i+30] for i in SEAT_BLOCKS]
    if not all(valid_grid(layout) for layout in layouts):
        raise ValueError("four_complete_layouts")
    count = int.from_bytes(raw[26:28], "little")
    if not count or 412 + 10*count > len(raw):
        raise ValueError("event_length")
    seats = [raw[i-28] for i in SEAT_BLOCKS]
    if set(seats) != set(range(4)):
        raise ValueError("seat_codes")
    events = [raw[i:i+10] for i in range(412, 412+10*count, 10)]
    key = hashlib.sha256(b"".join(layouts)+b"".join(events)).hexdigest()
    return key, layouts, seats, events


def initial_board(layouts):
    return {grid(seat, r, c): (seat, layout[r*5+c])
            for seat, layout in enumerate(layouts)
            for r in range(6) for c in range(5) if layout[r*5+c]}


def collision(attacker, defender):
    if 4 in (attacker, defender) or attacker == defender:
        return 3
    if defender == 3:
        return 1 if attacker == 13 else 2
    return 1 if defender == 2 or attacker < defender else 2


def archived_move(board, event):
    origin, target = (event[2], 16-event[3]), (event[4], 16-event[5])
    actor, defender = board.get(origin), board.get(target)
    outcome = event[1] & 3
    if actor is None or (outcome == 0) != (defender is None):
        raise ValueError("occupancy")
    if defender:
        if actor[0] % 2 == defender[0] % 2:
            raise ValueError("ally_collision")
        if collision(actor[1], defender[1]) != outcome:
            raise ValueError("collision")
    del board[origin]
    if outcome <= 1:
        board[target] = actor
    elif outcome == 3:
        del board[target]
    return actor[0], bool(outcome)


def replay_facts(layouts, seats, events):
    board = initial_board(layouts)
    stats = Counter()
    active = set(range(4))
    zero_since = {}
    ply = quiet = 0
    last = None
    ends = []
    for event in events:
        if event[0] == 95:
            actor, contact = archived_move(board, event)
            ply += 1
            if last is not None:
                stats[f"turn_gap_{(actor-last) % 4}"] += 1
            last = actor
            stats["contacts"] += contact
            if ply <= 40:
                stats["opening_plies"] += 1
                stats["opening_contacts"] += contact
            if quiet == 70:
                stats["next_after_70_contact" if contact else "next_after_70_quiet"] += 1
            quiet = 0 if contact else quiet+1
            owned = {owner for owner, _ in board.values()}
            movable = {owner for point, (owner, rank) in board.items()
                       if rank not in (2, 3) and point not in HQ}
            for seat in (active & owned) - movable:
                zero_since.setdefault(seat, ply)
        elif event[0] == 245:
            kind, seat_code, cause = event[1:4]
            stats[f"control_{kind}_{cause}"] += 1
            if kind == 5:
                ends.append((cause, quiet, ply))
            if kind not in (3, 4) or seat_code not in seats:
                continue
            seat = seats.index(seat_code)
            if (kind, cause) == (3, 2):
                stats["no_move_with_flag"] += any((seat, 2) == item for item in board.values())
                delay = ply-zero_since[seat] if seat in zero_since else None
                label = "no_move_blocked_or_empty" if delay is None else "no_move_permanent_now" if not delay else "no_move_permanent_delayed"
                stats[label] += 1
                if last is not None:
                    next_seat = next(((last-s) % 4 for s in range(1, 5)
                                      if (last-s) % 4 in active), None)
                    stats["no_move_not_next_seat"] += next_seat != seat
            elif seat in active and seat in zero_since:
                stats["permanent_other_exit_reason"] += 1
            active.discard(seat)
            board = {pos: item for pos, item in board.items() if item[0] != seat}
        else:
            raise ValueError("event_type")
    stats["moves"] = ply
    stats["missing_end"] = not ends
    stats["permanent_still_active_at_end"] = len(active & zero_since.keys())
    for cause, streak, _ in ends:
        if cause == 2:
            stats["end_draw_71" if streak == 71 else "end_draw_other"] += 1
    stats["layouts"] = len(layouts)
    stats["non_flag_hq_bomb"] = sum(layout[26] == 4 or layout[28] == 4 for layout in layouts)
    return stats, ends


def public_records(path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not 412 <= member.size <= LIMIT:
                continue
            if member.name.rsplit("/", 1)[-1].startswith("._"):
                continue
            with archive.extractfile(member) as data:
                yield data.read(LIMIT+1)


def local_records(path):
    for file in sorted(path.rglob("*.jgs")):
        if file.is_file() and file.stat().st_size <= LIMIT:
            yield file.read_bytes()


def summarize(records):
    totals = Counter()
    bytes_seen, rounds = set(), set()
    evidence = {}
    for raw in records:
        totals["files"] += 1
        raw_id = hashlib.sha256(raw).digest()
        if raw_id in bytes_seen:
            continue
        bytes_seen.add(raw_id)
        try:
            key, layouts, seats, events = decode(raw)
            if key in rounds:
                continue
            rounds.add(key)
            stats, ends = replay_facts(layouts, seats, events)
        except ValueError as error:
            totals["rejected_" + str(error)] += 1
            continue
        totals["aligned_games"] += 1
        totals.update(stats)
        if (stats["next_after_70_contact"] or stats["end_draw_71"] or
                stats["control_3_2"]):
            evidence[key[:16]] = {k: v for k, v in stats.items() if k.startswith("no_move")}
            evidence[key[:16]].update(moves=stats["moves"], ends=ends,
                                     next_collision=stats["next_after_70_contact"],
                                     after_70_quiet=stats["next_after_70_quiet"])
    return {"totals": dict(totals), "unique_files": len(bytes_seen), "evidence": evidence}, rounds


def decode_rank_grid(raw):
    layout = raw[20:50]
    if len(raw) != 50 or not raw.startswith(b"QQGame JQL") or not valid_grid(layout):
        return None
    if layout.index(2) not in (26, 28) or 3 in layout[:20] or 4 in layout[:5]:
        return None
    return layout


def rank_grid_check(path):
    files = 0
    valid = set()
    rejected = 0
    for file in path.rglob("*.jql"):
        files += 1
        layout = decode_rank_grid(file.read_bytes())
        if layout is None:
            rejected += 1
        else:
            valid.add(layout)
    return {"files": files, "unique_layouts": len(valid), "rejected": rejected,
            "bomb_hq_layouts": sum(4 in (layout[26], layout[28]) for layout in valid)}


def run():
    inputs = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    sources = {name: Path(path) for name, path in inputs.items()}
    corpus, seen = summarize(public_records(sources["corpus"]))
    local, local_seen = summarize(local_records(sources["local"]))
    paired, paired_seen = summarize([sources["supplemental"].read_bytes()])
    summary = {
        "archive": corpus, "local": local, "supplemental": paired,
        "public_local_overlap": len(seen & local_seen),
        "supplemental_overlap": len(paired_seen & (local_seen | seen)),
        "jql": rank_grid_check(sources["layouts"]),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
    temp.replace(OUTPUT)
    for source in ("archive", "local", "supplemental"):
        print(source, json.dumps(summary[source]["totals"], sort_keys=True))
    print("jql", summary["jql"], "public_local_overlap", summary["public_local_overlap"])


def audit():
    evidence = json.loads(OUTPUT.read_text(encoding="utf8"))
    archive, local, personal = (evidence[name]["totals"] for name in ("archive", "local", "supplemental"))
    if (archive["aligned_games"], local["aligned_games"], personal["aligned_games"]) != (3235, 41, 1):
        raise ValueError("source coverage changed; review provenance")
    if any(t.get("missing_end") or t.get("rejected_occupancy") or t.get("rejected_collision")
           for t in (archive, local, personal)):
        raise ValueError("missing end, collision or occupancy failure")
    if (archive.get("end_draw_71"), archive.get("next_after_70_contact"),
            local.get("end_draw_71"), personal.get("no_move_permanent_now")) != (42, 2, 1, 2):
        raise ValueError("frozen rule witnesses no longer reproduced")
    witnesses = evidence["archive"]["evidence"]
    for key in ("42db2f58cd7087e1", "a757ad6bae52e860", "7781084cad2c6cb6"):
        print("draw witness", key, witnesses[key])
    print("PASS local and archival evidence; aggregate only")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
