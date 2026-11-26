"""An alternate move may be a one-turn detour, or followup may be missing."""
from verify_archive_cycle_exit_options import recurrence
from pathlib import Path
import runpy

move = runpy.run_path(str(Path(__file__).with_name("archive_cycle_exit_checks.py")))["move"]


def check():
    root = dict(actor=0, piece=0, origin=[1, 0], target=[0, 0])
    departure = dict(index=4)
    moves = [move(5, 0, 0, 1), move(6, 0, 1, 0), move(7, 0, 0, 1)]
    r = recurrence(moves, root, departure)
    assert r["edge_reused"]["own_turn"] == 1 and r["opposite_pair"]["own_turn"] == 2
    assert r["stopped_by"] == "three_own_decisions"
    stop = dict(index=5, kind="interruption", ply=5, actor=-1)
    r = recurrence([stop]+moves, root, departure)
    assert r["decisions"] == 0 and not r["edge_reused"]
    removed = move(5, 1, 9, 0, True, removed=[0], piece=7)
    assert recurrence([removed], root, departure)["stopped_by"] == "loop_piece_removed"
    lost_now = move(4, 0, 0, 2, True, removed=[0])
    gone = recurrence([lost_now]+moves, root, departure)
    assert gone["stopped_by"] == "loop_piece_removed_at_exit" and gone["decisions"] == 0
    other = [move(i, 0, 0, 1, piece=4) for i in range(5, 8)]
    assert not recurrence(other, root, departure)["edge_reused"]
    assert recurrence([], root, departure)["stopped_by"] == "record_end"
    print("PASS recurrence, absent followup, piece removal and replacement identity fixtures")
