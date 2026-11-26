"""Counterexamples for the event-only cycle exit audit; no new chess rules."""
from audit_archive_cycle_exit_decisions import follow, prefix_root


def move(index, actor, a, b, contact=False, removed=(), piece=0):
    return dict(index=index, ply=index+1, kind="move", actor=actor, piece=piece,
                origin=[a, 0], target=[b, 0], contact=contact, removed=list(removed))


def check():
    prefix = [move(i, 0, i % 2, 1-i % 2) for i in range(4)]
    root = prefix_root(prefix, 4)
    external = move(4, 1, 9, 8, True, piece=10)
    more = [external, move(5, 0, 0, 1), move(6, 0, 1, 2)]
    answer = follow(prefix+more, root)
    assert answer["first_change"]["kind"] == "contact_enemy"
    assert answer["actor_exit"]["kind"] == "quiet_same_piece"
    assert answer["returns_after_change"] == 1
    # Seeing a different future must not retroactively change the selected root.
    terminal = dict(index=7, ply=7, kind="terminal", actor=-1)
    assert prefix_root(prefix+more+[terminal], 4) == root
    assert prefix_root(prefix+[terminal], 4) == root
    assert follow(prefix+[terminal], root)["actor_exit"]["kind"] == "terminal"
    removal = move(4, 1, 9, 0, True, removed=[0], piece=10)
    assert follow(prefix+[removal], root)["actor_exit"]["kind"] == "loop_piece_removed"
    retired = dict(index=4, ply=4, kind="retirement", actor=2, removed=[4])
    answer = follow(prefix+[retired, move(5, 0, 0, 1), terminal], root)
    assert answer["actor_exit"]["kind"] == "terminal" and answer["returns_after_change"] == 1
    interruption = dict(index=4, ply=4, kind="interruption", actor=-1)
    assert follow(prefix+[interruption, removal], root)["actor_exit"]["kind"] == "interruption"
    other = move(4, 0, 9, 8, piece=2)
    assert follow(prefix+[other], root)["actor_exit"]["kind"] == "quiet_other_piece"
    # Continuing locally can coexist with other-seat captures; no inferred intent.
    assert follow(prefix+[external, move(5, 0, 0, 1), terminal], root)["continued_returns"] == 1
    print("PASS contact-not-exit, later-future invariance, retirement, censoring and piece identity")
