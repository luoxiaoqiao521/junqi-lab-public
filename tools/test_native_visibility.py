"""Check that native visibility changes player input and survives replay."""
import json
from pathlib import Path
import tempfile

from test_single_match import CONFIGS, command, read


def trajectory(log):
    return [(frame["event"], frame["hash"]) for frame in log[1:]]


def check_visibility(log, mode):
    header, *frames = log
    ally_visible = mode != "simulate"
    enemy_visible = mode == "simulate-full"
    assert header["observation"]["ally_visibility"] == ("all" if ally_visible else "none")
    assert header["observation"]["enemy_identity_visible"] == enemy_visible
    assert "match" not in header
    for frame in frames:
        for viewer, view in enumerate(frame["views"]):
            for node, owner in enumerate(frame["seats"]):
                visible = enemy_visible or owner == viewer or (ally_visible and owner == (viewer+2)%4)
                assert view[node] == (frame["truth"][node] if visible else -1)


def changed_header_rejected(path, new_observation, original):
    header, *frames = original
    header = dict(header, observation=new_observation["observation"])
    path.write_text("".join(json.dumps(row)+"\n" for row in (header,*frames)), encoding="utf-8")
    command("replay",path,expected_error="observation hash mismatch")
    header["observation_hash"] = new_observation["observation_hash"]
    path.write_text("".join(json.dumps(row)+"\n" for row in (header,*frames)), encoding="utf-8")
    command("replay",path,expected_error="initial frame mismatch")


def main():
    modes = ("simulate", "simulate-dual", "simulate-full")
    with tempfile.TemporaryDirectory(prefix="junqi-native-visibility-") as directory:
        root = Path(directory)
        random, heuristics = {}, {}
        for mode in modes:
            for policy, output in (("random",random),("heuristic-all",heuristics)):
                path = root / f"{mode}-{policy}.jsonl"
                command(mode,42,path,*CONFIGS,policy)
                command("replay",path)
                output[mode] = read(path)
                check_visibility(output[mode],mode)
                error = None if mode == "simulate" else "diagnostic requires a four-dark match"
                command("replay-public-evidence",path,0,expected_error=error)
        initial = [log[0] for log in random.values()]
        assert len({h["rules_hash"] for h in initial}) == 1
        assert len({h["observation_hash"] for h in initial}) == 3
        assert all(h["layouts"] == initial[0]["layouts"] for h in initial)
        assert all(trajectory(log) == trajectory(random["simulate"]) for log in random.values())
        assert trajectory(heuristics["simulate"]) != trajectory(heuristics["simulate-full"])
        changed_header_rejected(root / "bad.jsonl",initial[2],random["simulate-dual"])
    print("PASS all four acting seats per mode; random trajectories match; full-information heuristic changed moves")
    print("PASS distinct visibility hashes, cross-mode frame tampering rejected, and six replayed games")


if __name__ == "__main__":
    main()
