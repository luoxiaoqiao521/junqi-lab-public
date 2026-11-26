"""Count sustained disadvantages in local, de-identified archived games."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_qq_archive import INPUT, ROOT, archived_move, decode, initial_board, public_records
from audit_replay_retirement import mover, points


LANDMARKS = (40, 80, 100, 140)
THRESHOLDS = (50, 80, 90)


def natural_result(events, seats):
    if events[-1][0:2] != bytes((245, 5)) or events[-1][3] != 4:
        return None
    retired = {}
    for event in events[:-1]:
        if event[0] == 95:
            continue
        if event[0:2] != bytes((245, 3)) or event[2] not in seats or event[3] not in (1, 2):
            return None
        seat = seats.index(event[2])
        if seat in retired:
            return None
        retired[seat] = event[3]
    lost = [team for team in range(2) if {team, team + 2} <= retired.keys()]
    if len(lost) != 1:
        return None
    team = lost[0]
    ending = "two_flags" if all(retired[s] == 1 for s in (team, team + 2)) else "includes_no_move"
    return 1 - team, ending


def snapshot(board, live, previous):
    counts, movers, sides = [0, 0], [0, 0], [0, 0]
    for point, piece in board.items():
        side = piece[0] % 2
        counts[side] += 1
        movers[side] += mover(point, piece)
    for seat in live:
        sides[seat % 2] += 1
    return counts, movers, sides, previous


def replay(layouts, seats, events):
    board, live = initial_board(layouts), set(range(4))
    frames = []
    previous = "opening"
    for event in events[:-1]:
        if event[0] == 95:
            frames.append(snapshot(board, live, previous))
            _, target = points(event)
            flag_hit = board.get(target, (None, None))[1] == 2
            _, contact = archived_move(board, event)
            previous = "flag_collision" if flag_hit else "collision" if contact else "quiet"
            continue
        seat = seats.index(event[2])
        if event[3] == 1 and (seat, 2) in board.values():
            raise ValueError("flag_retirement_with_flag")
        live.remove(seat)
        board = {point: piece for point, piece in board.items() if piece[0] != seat}
        previous = "flag_retirement" if event[3] == 1 else "no_move_retirement"
    if not frames:
        raise ValueError("no_moves")
    return frames


def trajectory(frames, side):
    return [(counts[side] - counts[1-side], movers[side] - movers[1-side],
             sides[side] - sides[1-side], previous) for counts, movers, sides, previous in frames]


def describe(curve):
    negative = [i for i, p in enumerate(curve) if p[0] < 0]
    last = negative[-1] if negative else None
    low = min(range(len(curve)), key=lambda i: curve[i][0])
    return {"plies": len(curve), "pieces_below": len(negative),
            "movers_below": sum(p[1] < 0 for p in curve),
            "both_below": sum(p[0] < 0 and p[1] < 0 for p in curve),
            "seats_below": sum(p[2] < 0 for p in curve), "lowest_gap": curve[low][0], "low_ply": low,
            "pre_final_gap": curve[-1][0], "last_negative_ply": last,
            "parity_recovery": curve[last + 1][3] if last is not None and last + 1 < len(curve) else None}


def add_side(panels, phase, result, curve, desc):
    score = panels[phase][result]
    score["sides"] += 1
    score["below_immediately_before_final_move"] += desc["pre_final_gap"] < 0
    for t in THRESHOLDS:
        for name in ("pieces_below", "movers_below", "both_below", "seats_below"):
            score[f"{name}_pct_{t}"] += (100 * desc[name] >= t * len(curve))
    for at in LANDMARKS:
        if len(curve) > at:
            score[f"reached_{at}"] += 1
            score[f"below_at_{at}"] += curve[at][0] < 0
    if desc["parity_recovery"]:
        score[f"recovered_via_{desc['parity_recovery']}"] += 1


def exhibit(book):
    wins = [r for r in book if r["outcome"] == "win" and r["ending"] == "two_flags"
            and 5 * r["pieces_below"] >= 4 * r["plies"]]
    keep = [r for r in wins if r["pre_final_gap"] < 0]
    rise = [r for r in wins if r["parity_recovery"] is not None]
    examples = {}
    if keep:
        examples["wins_with_gap_until_final"] = max(keep, key=lambda r: r["pieces_below"] / r["plies"])
    if rise:
        examples["wins_after_regaining_parity"] = max(rise, key=lambda r: r["pieces_below"] / r["plies"])
    eligible = [r for r in book if r["ending"] == "two_flags" and r["plies"] > 100 and r["curve"][100][0] < 0]
    keyed = defaultdict(list)
    for r in eligible:
        if r["outcome"] == "loss":
            keyed[tuple(r["curve"][100][:3])].append(r)
    for win in sorted(eligible, key=lambda r: (-r["pieces_below"] / r["plies"], r["key"])):
        if win["outcome"] != "win":
            continue
        options = [loser for loser in keyed[tuple(win["curve"][100][:3])] if loser["key"] != win["key"]]
        if options:
            examples["matched_win_at_100"] = win
            examples["matched_loss_at_100"] = min(options, key=lambda r: abs(r["plies"] - win["plies"]))
            break
    return examples


def study(corpus):
    seen = set()
    rejected = Counter()
    panels = {phase: {"win": Counter(), "loss": Counter()} for phase in
              ("all_natural", "two_flags", "includes_no_move")}
    book = []
    for source in public_records(corpus):
        try:
            key, layouts, seats, events = decode(source)
            if key in seen:
                continue
            seen.add(key)
            outcome = natural_result(events, seats)
            if outcome is None:
                continue
            winning, phase = outcome
            frames = replay(layouts, seats, events)
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        for side in range(2):
            curve = trajectory(frames, side)
            desc = describe(curve)
            result = "win" if side == winning else "loss"
            for panel in ("all_natural", phase):
                add_side(panels, panel, result, curve, desc)
            book.append(dict(desc, key=key[:16], side=side, outcome=result, ending=phase, curve=curve))
    examples = exhibit(book)
    if len(examples) < 3 or panels["all_natural"]["win"]["sides"] != panels["all_natural"]["loss"]["sides"]:
        raise ValueError("incomplete matched cohort")
    with corpus.open("rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    return {"corpus_sha256": digest, "unique_complete_layout_games": len(seen), "rejected": dict(rejected),
            "panels": panels, "examples": examples}


def certify(corpus, directory, examples):
    keys = {r["key"] for r in examples.values()}
    confirmed = {}
    for raw in public_records(corpus):
        try:
            key = decode(raw)[0][:16]
        except ValueError:
            continue
        if key not in keys or key in confirmed:
            continue
        source = directory / f"{key}.jgs"
        log = directory / f"{key}.jsonl"
        if not log.is_file():
            source.write_bytes(raw)
            try:
                args = [ROOT / "build/junqi.exe", "import-jgs", source, log,
                        ROOT / "config/ruleset.json", ROOT / "config/observation.json"]
                subprocess.run(args, capture_output=True, check=True, timeout=10)
            finally:
                source.unlink(missing_ok=True)
        run = subprocess.run([ROOT / "build/junqi.exe", "replay", log],
                             capture_output=True, check=True, text=True, encoding="utf8", timeout=10)
        if not run.stdout.startswith("JGS REPLAY PASS"):
            raise ValueError("example did not pass strict native replay")
        confirmed[key] = run.stdout.strip()
    if confirmed.keys() != keys:
        raise ValueError("example source missing")
    return confirmed


def main(audit):
    corpus = Path(json.loads(INPUT.read_text(encoding="utf-8-sig"))["corpus"])
    directory = ROOT / "build/comeback" / os.environ["JUNQI_SERIAL_KEY"]
    directory.mkdir(parents=True, exist_ok=True)
    result = study(corpus)
    wire = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    receipt = directory / "summary.json"
    if audit:
        if receipt.read_bytes() != wire.encode():
            raise ValueError("trajectory analysis did not reproduce")
        print(json.dumps(certify(corpus, directory, result["examples"]), sort_keys=True))
    else:
        pending = receipt.with_suffix(".tmp")
        pending.write_bytes(wire.encode())
        pending.replace(receipt)
    print(json.dumps(result["panels"], sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
