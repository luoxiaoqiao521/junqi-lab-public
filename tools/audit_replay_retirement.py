"""Read-only JGS audit. Print rule evidence without player fields or layouts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_qq_archive import HQ, archived_move, decode, grid, initial_board, local_records, public_records


BOTTOM = {grid(seat, 5, col) for seat in range(4) for col in (0, 2, 4)}
SCREENSHOT = "2985df11fd596d4f"
SAMPLES = {"ab93590e0a397354": 172, "2d33397f6f4a24e3": 92}
TALLIES = ("games", "plies", "flag_captures", "flag_capture_with_movers", "next_notice_flag",
           "later_move_after_flag", "next_quiet_to_cleared_square", "adjacent_quiet_to_cleared_square",
           "bottom_target", "bottom_quiet_in", "bottom_out",
           "hq_departure", "last_mover_hq_no_move", "no_move_with_flag", "no_move_with_mover")


def mover(point, piece):
    return piece[1] not in (2, 3) and point not in HQ


def points(event):
    return (event[2], 16-event[3]), (event[4], 16-event[5])


def tally_move(stats, board, event):
    origin, target = points(event)
    actor = board.get(origin)
    defender = board.get(target)
    archived_move(board, event)
    stats["bottom_target"] += target in BOTTOM
    stats["bottom_quiet_in"] += target in BOTTOM and (event[1] & 3) == 0
    stats["bottom_out"] += origin in BOTTOM
    stats["hq_departure"] += origin in HQ
    flag = defender is not None and defender[1] == 2 and (event[1] & 3) in (1, 3)
    entered_hq = target in HQ and board.get(target) == actor
    return actor[0], defender[0] if flag else None, entered_hq


def verify_flag_notice(events, index, seats, flag, board, ply, stats):
    stats["flag_captures"] += 1
    remaining = {point: piece for point, piece in board.items() if piece[0] == flag}
    movers = sum(mover(point, piece) for point, piece in remaining.items())
    stats["flag_capture_with_movers"] += movers > 0
    if index + 1 >= len(events) or events[index + 1][:4] != bytes((245, 3, seats[flag], 1)):
        return None
    stats["next_notice_flag"] += 1
    return index, ply, remaining, movers


def tally_next(stats, previous_flag, event, index, ply, key, witnesses):
    if previous_flag is None:
        return
    stats["later_move_after_flag"] += 1
    previous_index, previous_ply, remaining, movers = previous_flag
    origin, target = points(event)
    if (event[1] & 3) or target not in remaining:
        return
    stats["next_quiet_to_cleared_square"] += 1
    stats["adjacent_quiet_to_cleared_square"] += index == previous_index + 2
    if SAMPLES.get(key) == ply:
        witnesses[key] = dict(capture_ply=previous_ply, next_ply=ply, reserve=len(remaining),
                              movers=movers, move=[origin, target],
                              capture_seq=previous_index+1, next_seq=index+1)


def tally_control(event, board, seats, entered, ply, stats):
    kind, raw_seat, cause = event[1:4]
    if kind not in (3, 4) or raw_seat not in seats:
        return
    seat = seats.index(raw_seat)
    own = {point: piece for point, piece in board.items() if piece[0] == seat}
    if (kind, cause) == (3, 2):
        movers = sum(mover(point, piece) for point, piece in own.items())
        stats["no_move_with_flag"] += (seat, 2) in own.values()
        stats["no_move_with_mover"] += movers > 0
        stats["last_mover_hq_no_move"] += entered == (ply, seat) and not movers
    for point in own:
        del board[point]


def screenshot_event(seq, ply, event, board, seats):
    if seq in (28, 95, 198):
        origin, target = points(event)
        losses = [(board[point][0], board[point][1]) for point in (origin, target)
                  if point in board]
        revealed = [board.get((event[i], 16-event[i+1])) for i in (6, 8) if any(event[i:i+2])]
        dead_marshals = sorted(owner for owner, rank in losses if rank == 5)
        if any(piece is None or piece[1] != 2 for piece in revealed):
            raise ValueError("screenshot reveal points to a non-flag square")
        if sorted(owner for owner, _ in revealed) != dead_marshals:
            raise ValueError("screenshot reveal did not match marshal losses")
        return dict(seq=seq, ply=ply+1, contact=event[1]&3, ranks=[rank for _, rank in losses],
                    revealed=len(revealed))
    if seq in (216, 247, 248):
        kind, raw_seat, cause = event[1:4]
        seat = seats.index(raw_seat) if kind == 3 and raw_seat in seats else None
        own = [(point, piece) for point, piece in board.items() if piece[0] == seat]
        return dict(seq=seq, ply=ply, kind=kind, cause=cause, seat=seat,
                    remaining=len(own), movers=sum(mover(p, piece) for p, piece in own),
                    hq_lock=sum(p in HQ and piece[1] >= 4 for p, piece in own))
    return None


def game(layouts, seats, events, key):
    board = initial_board(layouts)
    stats, witnesses, screenshot = Counter(games=1), {}, []
    ply, entered, previous_flag = 0, None, None
    for index, event in enumerate(events):
        if key == SCREENSHOT:
            marker = screenshot_event(index+1, ply, event, board, seats)
            if marker:
                screenshot.append(marker)
        if event[0] == 245:
            tally_control(event, board, seats, entered, ply, stats)
            continue
        if event[0] != 95:
            raise ValueError("event_type")
        ply += 1
        tally_next(stats, previous_flag, event, index, ply, key, witnesses)
        actor, flag, in_hq = tally_move(stats, board, event)
        entered = (ply, actor) if in_hq else None
        previous_flag = (verify_flag_notice(events, index, seats, flag, board, ply, stats)
                         if flag is not None else None)
    stats["plies"] = ply
    return stats, witnesses, screenshot


def source(records):
    totals, rejected, found, screenshots = Counter(), Counter(), {}, []
    seen = set()
    for raw in records:
        try:
            key, layouts, seats, events = decode(raw)
            if key in seen:
                continue
            seen.add(key)
            stats, witnesses, screenshot = game(layouts, seats, events, key[:16])
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        totals.update(stats)
        found.update(witnesses)
        screenshots.extend(screenshot)
    return {"totals": {name: totals[name] for name in TALLIES}, "rejected": dict(rejected),
            "witnesses": found, "screenshot": screenshots}


def verify(result):
    expected = ((3235, 654829, 2795, 1896, 45, 3092, 9462, 29, 995, 239),
                (41, 5265, 31, 23, 1, 40, 90, 0, 3, 1), (1, 245, 0, 0, 0, 0, 5, 0, 2, 0))
    found = {}
    for detail, want in zip(result.values(), expected, strict=True):
        tally = detail["totals"]
        keys = ("games", "plies", "flag_captures", "later_move_after_flag",
                "next_quiet_to_cleared_square", "bottom_target", "bottom_out",
                "last_mover_hq_no_move", "no_move_with_flag", "no_move_with_mover")
        if tuple(tally[name] for name in keys) != want:
            raise ValueError("source totals changed; review evidence provenance")
        if (tally["hq_departure"] or tally["flag_captures"] != tally["next_notice_flag"] or
                tally["flag_captures"] != tally["flag_capture_with_movers"]):
            raise ValueError("historical retirement witness changed")
        found.update(detail["witnesses"])
    if set(found) != SAMPLES.keys():
        raise ValueError("named clearing witness missing")
    markers = result["supplemental"]["screenshot"]
    if [(m["seq"], m["ply"]) for m in markers] != [(28,28),(95,95),(198,198),(216,215),(247,245),(248,245)]:
        raise ValueError("screenshot event and ply mapping changed")
    if [(m["contact"], m["ranks"], m["revealed"]) for m in markers[:3]] != [
            (3, [5, 5], 2), (3, [4, 5], 1), (3, [4, 5], 1)]:
        raise ValueError("screenshot marshal contacts changed")
    if [(m["kind"], m["cause"], m["movers"], m["hq_lock"]) for m in markers[3:]] != [
            (3,2,0,1),(3,2,0,1),(5,4,0,0)]:
        raise ValueError("screenshot exit or terminal markers changed")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs", type=Path, help="local jgs-evidence-input.json manifest")
    cli.add_argument("--verify", action="store_true", help="check the frozen source witnesses")
    args = cli.parse_args()
    inputs = {k: Path(v) for k, v in json.loads(args.inputs.read_text(encoding="utf-8-sig")).items()}
    result = {"archive": source(public_records(inputs["corpus"])),
              "local": source(local_records(inputs["local"])),
              "supplemental": source([inputs["supplemental"].read_bytes()])}
    print(json.dumps(result, ensure_ascii=False))
    if args.verify:
        verify(result)
        print("PASS frozen replay retirement evidence")


if __name__ == "__main__":
    main()
