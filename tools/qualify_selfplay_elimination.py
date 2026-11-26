"""Exercise the opt-in retirement policy through the built CLI."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/junqi.exe"
RULES = ROOT / "config/ruleset-selfplay.json"
ARCHIVE = ROOT / "config/ruleset.json"
OBS = ROOT / "config/observation.json"
OUT = ROOT / "build/selfplay-settlement"
RESULT = OUT / "receipt.json"


def run_cli(*args, timeout=60):
    return subprocess.run([str(EXE), *map(str,args)], capture_output=True, text=True,
                          timeout=timeout, check=True).stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_profile():
    current = json.loads(RULES.read_text(encoding="utf8"))
    previous = json.loads(ARCHIVE.read_text(encoding="utf8"))
    changes = {k for k,v in current["experimental"].items() if previous["experimental"].get(k)!=v}
    assert changes == {"elim.no_legal_move"}
    assert current["experimental"]["elim.no_legal_move"] == "eliminated_no_movable_piece"
    assert current["id"] != previous["id"]


def check_replay_details(log, folder):
    rows = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines()]
    assert rows[0]["settlement_detail"] is True
    assert rows[-1]["end_reason"] != "none"
    for row in rows[2:]:
        assert row["event"][4] == sum(1 << exit["seat"] for exit in row["eliminations"])
        assert all(exit["ply"] == row["seq"] for exit in row["eliminations"])
        assert row["event"][4] & row["auto_skipped_mask"] == 0
    # Old selfplay logs lack the extension. Removing the headers and optional
    # fields gives the same format and state hashes as the earlier writer.
    prior = json.loads(json.dumps(rows))
    del prior[0]["settlement_detail"]
    for row in prior[2:]:
        for field in ("eliminations", "auto_skipped_mask", "end_reason"):
            del row[field]
    name = folder / "previous-selfplay-format.jsonl"
    name.write_text("".join(json.dumps(row) + "\n" for row in prior), encoding="utf8")
    assert run_cli("replay",name).startswith("REPLAY PASS")
    for key, value in (("end_reason", "tampered"), ("auto_skipped_mask", -1),
                       ("eliminations", [{"seat": 0, "ply": -1, "reason": "flag_removed"}])):
        bad = json.loads(json.dumps(rows))
        bad[-1][key] = value
        name = folder / "tampered.jsonl"
        name.write_text("".join(json.dumps(row) + "\n" for row in bad), encoding="utf8")
        rejected = subprocess.run([str(EXE), "replay", str(name)], text=True,
                                  capture_output=True, timeout=20)
        assert rejected.returncode != 0 and "frame/event mismatch" in rejected.stderr
    return dict(end_reason=rows[-1]["end_reason"], exits=sum(len(row["eliminations"]) for row in rows[2:]),
                old_selfplay_format=True, tamper_rejections=3)


def run():
    check_profile()
    OUT.mkdir(exist_ok=True)
    limits = ("740130", ".015", "1", "32", "short", "0")
    trained = json.loads(run_cli("train-general", RULES, *limits, timeout=110))
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        folder = Path(folder)
        new, legacy = (folder / name for name in ("selfplay.jsonl","legacy.jsonl"))
        run_cli("simulate",998123,new,RULES,OBS)
        run_cli("simulate",998123,legacy,ARCHIVE,OBS)
        headers = []
        for log in (new,legacy):
            with log.open(encoding="utf8") as stream:
                headers.append(json.loads(stream.readline()))
        replays = [run_cli("replay",log).strip() for log in (new,legacy)]
        details = check_replay_details(new,folder)
        limited = json.loads(RULES.read_text(encoding="utf8"))
        limited["experimental"]["sim.truncation_cap"] = 1
        cap_rules = folder / "cap.json"
        cap_rules.write_text(json.dumps(limited), encoding="utf8")
        cutoff = json.loads(run_cli("train-general",cap_rules,*limits))
    result = dict(engine=digest(EXE), profile=digest(RULES), trained=trained, replays=replays,
                  replay_settlement=details, forced_cutoff=cutoff,
                  rules_hashes=[header["rules_hash"] for header in headers],
                  rule_ids=[header["rules"]["id"] for header in headers])
    RESULT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    audit()


def audit():
    check_profile()
    r = json.loads(RESULT.read_text(encoding="utf8"))
    t = r["trained"]
    assert (r["engine"],r["profile"])==(digest(EXE),digest(RULES))
    assert len(set(r["rules_hashes"]))==len(set(r["rule_ids"]))==2
    assert all(value.startswith("REPLAY PASS ") for value in r["replays"])
    assert t["natural"]+t["truncated"]==512 and t["natural"]==t["wins"]+t["draws"]+t["losses"]
    assert t["decisions"]>0 and len(t["weights"])==5 and all(math.isfinite(w) for w in t["weights"])
    assert any(t["weights"])
    counts = t["settlement"]["terminal_reasons"]
    assert counts["team_eliminated"] == t["wins"] + t["losses"] and counts["no_progress_draw"] == t["draws"]
    assert counts["simultaneous_exhaustion"] + counts["no_legal_actions"] + counts["ply_cap"] == t["truncated"]
    assert t["settlement"]["scored_with_decisions"] == t["natural"] - t["no_decisions"]
    assert t["settlement"]["exit_reasons"]["no_legal_move"]["count"] == 0
    cut = r["forced_cutoff"]
    assert cut["settlement"]["terminal_reasons"]["ply_cap"] == cut["truncated"] == 512
    assert not cut["natural"] and not cut["settlement"]["scored_with_decisions"] and not any(cut["weights"])
    assert r["replay_settlement"]["old_selfplay_format"] and r["replay_settlement"]["tamper_rejections"] == 3
    print("PASS separate rules and main training CLI",t["natural"],t["truncated"],
          "wins/draws/losses",t["wins"],t["draws"],t["losses"],r["replays"])


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
