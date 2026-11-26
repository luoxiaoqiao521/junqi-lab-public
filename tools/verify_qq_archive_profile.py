"""Verify the offline archive profile against a real game and the launcher."""
import json
import os
from pathlib import Path
import subprocess
from audit_qq_archive import ROOT, INPUT, decode, initial_board
from compare_qq_rules import EXE, compile_probe, tape


def strict_replay():
    manifest = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    _, layouts, seats, events = decode(Path(manifest["supplemental"]).read_bytes())
    rows, ccw = tape(layouts,seats,events)
    data = [f"{len(rows)} {ccw}"]
    data.extend(f"{x} {y} {rank}" for (x,y),(_,rank) in initial_board(layouts).items())
    data.extend(" ".join(map(str,row)) for row in rows)
    compile_probe()
    subprocess.run([EXE,"strict"],input=("\n".join(data)+"\n").encode("ascii"),
                   check=True,timeout=20)


def verify_launcher():
    shell = os.environ.get("JUNQI_PWSH","pwsh")
    dist = ROOT / "dist/junqi-mvp"
    launch = subprocess.run([shell,"-NoProfile","-File",str(dist/"start-mvp.ps1"),
                             "-ArchiveQQ","-Seed","178","-NoOpen"],
                            capture_output=True,check=True,timeout=35)
    prefix = "Private log: "
    paths = [Path(line.removeprefix(prefix)) for line in
             launch.stdout.decode("utf8",errors="replace").splitlines() if line.startswith(prefix)]
    if len(paths)!=1:
        raise ValueError("offline launch did not return one replay")
    rules = json.loads((paths[0].parent/"ruleset.json").read_text(encoding="utf8"))["experimental"]
    expect = {"turn.order":"cw","elim.no_legal_move":"eliminated_after_involved_move",
              "draw.no_progress.limit":71,"marshal_death.reveal_flag":True}
    if any(rules.get(k)!=v for k,v in expect.items()):
        raise ValueError("launcher and research rules differ")
    subprocess.run([dist/"junqi.exe","replay",paths[0]],check=True,timeout=25)
    print("PASS packaged ArchiveQQ option and strict JSONL replay")


if __name__ == "__main__":
    strict_replay()
    verify_launcher()
