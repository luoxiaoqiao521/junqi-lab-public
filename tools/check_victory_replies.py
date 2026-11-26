"""Compile and verify the matched attack and defence teaching branches."""
import json
import os
import subprocess
import sys
from collections import Counter

from compare_qq_rules import VC
from measure_deadline_opening import ROOT, RULES, require


SOURCE = ROOT / "tests/victory_reply_probe.cpp"
PAIRED = (ROOT / "build/deadline-opening/expand-book-hard-outcomes-27d694d58dced9de82e9"
          / "paired-public.jsonl")
NAMES = ("finish_now", "same_attack_live_counter", "counter_wing_first",
         "exchange_last_mover", "same_exchange_with_reserve")
FIELDS = ("from", "to", "own_pieces", "enemy_pieces", "first_retired_mask", "first_result",
          "legal_replies", "own_finish_after_reply", "enemy_finishes", "draws", "cutoffs", "unresolved_replies")


def build(directory):
    sources = [SOURCE, *(ROOT / "core" / f"{name}.cpp" for name in ("board", "layout", "rules", "state"))]
    files = " ".join(f'"{p.as_posix()}"' for p in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'/I"{ROOT.as_posix()}/core" {files} /Fe:victory_reply_probe.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=directory, check=True, timeout=100)


def calculate(directory):
    run = subprocess.run([directory / "victory_reply_probe.exe", RULES], capture_output=True,
                         check=False, text=True, encoding="utf8", errors="replace", timeout=60)
    require(run.returncode == 0, f"lesson probe failed: {run.stderr.strip()}")
    rows = {}
    for line in run.stdout.splitlines():
        name, *values = line.split()
        require(name not in rows, f"repeated lesson row: {name}")
        row = dict(zip(FIELDS, map(int, values), strict=True))
        require(row["legal_replies"] == sum(row[k] for k in FIELDS[7:]), f"unclassified reply: {name}")
        require(row["own_pieces"] > row["enemy_pieces"], f"material advantage lost in fixture: {name}")
        rows[name] = row
    require(tuple(rows) == NAMES, "missing or unexpected lesson cases")
    trap = rows["same_attack_live_counter"]
    alternate = rows["counter_wing_first"]
    require(trap["enemy_finishes"] > 0 and trap["own_finish_after_reply"] > 0
            and alternate["enemy_finishes"] == 0, "positive and negative replies were not distinguished")
    return rows


def pressure_pairs():
    counts = Counter()
    examples = []
    total = 0
    with PAIRED.open(encoding="utf8") as source:
        for line in source:
            row = json.loads(line)
            view = row["observation"]
            side = view["actor"] % 2
            owners = [node[0] for node in view["board"] if node[0] >= 0]
            advantage = sum(1 if owner % 2 == side else -1 for owner in owners)
            total += 1
            if advantage < 5:
                continue
            outcomes = tuple(label["final"] or "unresolved" for _, label in row["labels"])
            counts[outcomes] += 1
            if (row["family"], *row["example"]) in ((1091320, 3, 0), (1091322, 0, 0)):
                labels = [{**label, "move": divmod(move, len(view["board"]))}
                          for move, label in row["labels"]]
                examples.append({"family": row["family"], "seat": view["actor"], "ply": view["ply"],
                                 "lead": advantage, "moves": labels, "diagnostics": row["diagnostics"]})
    require(len(examples) == 2, "previous paired examples did not reproduce")
    return {"total_roots": total, "five_more_on_board": sum(counts.values()),
            "at_least_one_loss": sum(n for outcomes, n in counts.items() if "loss" in outcomes),
            "outcome_pairs": {"/".join(outcomes): n for outcomes, n in counts.items()}, "examples": examples}


def main(audit):
    directory = ROOT / "build/deadline-opening" / os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True, exist_ok=True)
        build(directory)
    result = {"rule_fixtures": calculate(directory), "pressure_pairs": pressure_pairs()}
    wire = json.dumps(result, sort_keys=True, indent=2) + "\n"
    receipt = directory / "summary.json"
    if audit:
        require(receipt.read_text(encoding="utf8") == wire, "teaching replies did not reproduce")
    else:
        pending = receipt.with_suffix(".tmp")
        pending.write_text(wire, encoding="utf8")
        pending.replace(receipt)
    print(wire)
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
