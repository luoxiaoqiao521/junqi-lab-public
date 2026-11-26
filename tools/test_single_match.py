"""Exercise the offline single-dark CLI with generated games."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/junqi.exe"
CONFIGS = [ROOT / "config/ruleset.json", ROOT / "config/observation.json"]


def command(*args, expected_error=None):
    result = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True, timeout=25)
    if expected_error is None:
        result.check_returncode()
    elif result.returncode == 0 or expected_error not in result.stderr:
        raise AssertionError(f"expected {expected_error!r}, got {result.stderr!r}")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_game(header, frames):
    assert header["match"] == "single-dark" and header["first"] in (0, 2)
    assert header["rules"]["players"] == 2 and header["rules"]["teams"] == [[0], [2]]
    assert list(map(len, header["layouts"])) == [25, 0, 25, 0]
    assert len(header["nodes"]) == 129
    assert Counter(frames[0]["seats"]) == {0: 25, 2: 25, -1: 79}
    actor = header["first"]
    wings = set()
    for index, frame in enumerate(frames):
        assert set(frame["seats"]) <= {-1, 0, 2}
        assert frame["out"][1] == frame["out"][3] == 1
        assert frame["legal_move_counts"][1] == frame["legal_move_counts"][3] == 0
        if index:
            assert frame["event"][0] == actor
            actor = 2 - actor
            to = frame["event"][2]
            if 30 <= to < 60 or 90 <= to < 120:
                wings.add(to // 30)
        if index < len(frames)-1:
            assert frame["turn"] == actor
        for node, owner in enumerate(frame["seats"]):
            for viewer in (0, 2):
                expected = frame["truth"][node] if owner == viewer else -1
                assert frame["views"][viewer][node] == expected
    return wings


def check_rejection(path, log, four):
    single, *frames = log
    for header, message in (
        ({k: v for k, v in single.items() if k != "match"}, "rules hash mismatch"),
        (dict(four, match="single-dark"), "single match metadata mismatch"),
        (dict(single, first=1), "invalid first seat"),
        (dict(single, layouts=[single["layouts"][0]]*4), "layout slots for match"),
    ):
        path.write_text("".join(json.dumps(x)+"\n" for x in (header, *frames)), encoding="utf-8")
        command("replay", path, expected_error=message)


def check_matched_opening(single, four):
    dark_header, dark_start = four[:2]
    solo_header, solo_start = single[:2]
    assert all(solo_header[key] == dark_header[key] for key in ("nodes", "roads", "rails"))
    assert all(solo_header["layouts"][s] == dark_header["layouts"][s] for s in (0,2))
    for n, owner in enumerate(dark_start["seats"]):
        if owner in (0,2):
            assert solo_start["seats"][n] == owner and solo_start["truth"][n] == dark_start["truth"][n]
        else:
            assert solo_start["seats"][n] == solo_start["truth"][n] == -1
    first = dark_header["first"]
    expected = first if first in (0,2) else (first+1)%4
    assert solo_header["first"] == expected
    return first in (0,2)


def main():
    wings = set()
    results = Counter()
    matched_starts = set()
    with tempfile.TemporaryDirectory(prefix="junqi-single-contract-") as directory:
        scratch = Path(directory)
        for seed in range(43, 55):
            path = scratch / "sample.jsonl"
            command("simulate-single", seed, path, *CONFIGS)
            log = read(path)
            command("replay", path)
            wings |= check_game(log[0], log[1:])
            results[log[-1]["result"]] += 1
            if seed < 48:
                four = scratch / "aligned.jsonl"
                command("simulate",seed,four,*CONFIGS)
                matched_starts.add(check_matched_opening(log,read(four)))
                four.unlink()
            path.unlink()
        assert wings == {1, 3}
        assert matched_starts == {False, True}
        a, b = (scratch / f"{name}.jsonl" for name in ("a", "b"))
        for path in (a, b):
            command("simulate-single", 42, path, *CONFIGS, "heuristic-all")
            command("replay", path)
        assert a.read_bytes() == b.read_bytes()
        command("simulate-single", 42, a, *CONFIGS, expected_error="output exists")
        four = scratch / "four.jsonl"
        command("simulate", 42, four, *CONFIGS)
        check_rejection(scratch / "bad.jsonl", read(a), read(four)[0])
    print(f"PASS 12 generated single-dark matches, both wings and strict replay; results={dict(sorted(results.items()))}")
    print("PASS five paired openings: identical retained ranks and board, first seat retained or skipped")
    print("PASS deterministic heuristics, no overwrite, and cross-match metadata rejection")


if __name__ == "__main__":
    main()
