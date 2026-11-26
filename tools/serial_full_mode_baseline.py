"""Verify shared mode contracts, full-information bounds and portable execution."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from measure_deadline_opening import ROOT, build, require
from serial_capture_history import SOURCE
from serial_worker import write_receipt
from train_early_deadline_student import native

MODES = ("simulate-full", "simulate", "simulate-dual", "simulate-single")
CLASSIFICATIONS = {"matched", "collision", "flag_reveal", "retirement", "terminal", "next_seat", "quiet_clock", "illegal"}


def archive_inputs(audit):
    census = json.loads((SOURCE/"probe.json").read_bytes())["rows"]
    pool = (r for r in census if r["format"]==".jgs" and r["verified_plies"]>=120 and r["result"] in (1,2,3)
            and not any(r[k] for k in ("prior_layout","prior_trajectory","stopped","notice")))
    unique = {}
    for row in sorted(pool,key=lambda r:hashlib.sha256(("full-mode-baseline:"+r["family"]).encode()).digest()):
        unique.setdefault(row["family"],row)
        if len(unique)==12:
            break
    keys = {r["key"]:r for r in list(unique.values())[6 if audit else 0:12 if audit else 6]}
    wires = {}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape = line.split(maxsplit=2)
        if key in keys:
            require(tag=="G","full-mode replay uses a JGS prefix")
            row = keys[key]
            wires[key] = f'{key} {row["direction"]} {min(200,row["verified_plies"])} {tape}'
    require(len(wires)==6 and wires.keys()==keys.keys(),"missing fixed archive cases")
    return wires


def compare(directory, wires):
    raw = native(directory,"full-mode-baseline","\n".join(wires.values())+"\n")
    rows, proofs = {},{}
    for line in raw.splitlines():
        tag,*fields = line.split()
        if tag=="F":
            name,case,*values = fields
            proofs[f"{name}/{case}"] = list(map(int,values))
        elif tag=="X":
            require(fields[0]=="route_controls","unknown route control")
            proofs[fields[0]] = int(fields[1])
        elif tag=="J":
            key,arm,agrees,first,cause = fields
            require(key in wires and arm in ("0","1","2") and cause in CLASSIFICATIONS,"invalid archive difference row")
            require(0<=int(agrees)<=200 and (int(first)==0 if cause=="matched" else int(first)==int(agrees)+1),
                    "archive boundary did not stop at the first divergence")
            group = rows.setdefault(key,{})
            require(arm not in group,"repeated archive comparison")
            group[arm] = dict(agrees=int(agrees),first_difference=int(first),cause=cause)
        else:
            raise ValueError(f"unexpected full-mode baseline output {tag}")
    require(rows.keys()==wires.keys() and all(set(r)=={"0","1","2"} for r in rows.values()) and
            set(proofs)=={"routes/0","routes/1","bait/0","route_controls"},"incomplete full-mode comparison")
    return dict(proofs=proofs,rows=rows)


def check_modes(directory, engine, config):
    temporary = directory/"headless replay output"
    temporary.mkdir()
    inspections, shared = {}, []
    paths = [config/"ruleset-selfplay.json",config/"observation.json"]
    for command, visible, seats in zip(MODES,(100,25,50,25),(4,4,4,2),strict=True):
        target = temporary/(command+".jsonl")
        subprocess.run([engine,command,"42",target,*paths,"random"],cwd=temporary,check=True,capture_output=True,timeout=15)
        subprocess.run([engine,"export-review",target,temporary/(command+".js")],cwd=temporary,
                       check=True,capture_output=True,timeout=15)
        header, initial, *frames = [json.loads(line) for line in target.read_bytes().splitlines()]
        active = tuple(s for s,layout in enumerate(header["layouts"]) if len(layout)==25)
        require(len(active)==seats and all(sum(rank>=0 for rank in initial["views"][s])==visible for s in active),
                "wrong participant or initial visibility")
        require({f["event"][0] for f in frames}<=set(active),"inactive seat played a move")
        require(frames[-1]["result"] in (1,2,3,4),"headless mode did not finish")
        if seats==4:
            shared.append((header["layouts"], header["rules_hash"], [(f["hash"],f["event"]) for f in frames]))
        else:
            require(active==(0,2) and header["rules_hash"]!=shared[0][1] and
                    all(header["layouts"][s]==shared[0][0][s] for s in active),"single match did not preserve paired board seats")
        inspections[command] = dict(visible=visible,active=active,rules=header["rules_hash"],
                                    observation=header["observation_hash"],plies=frames[-1]["seq"],result=frames[-1]["result"])
    require(shared[0]==shared[1]==shared[2] and len({r["observation"] for r in inspections.values()})>=3,
            "visibility-only modes changed referee truth under fixed actions")
    return inspections


def portable(directory):
    source = ROOT/"dist/junqi-mvp"
    archive = directory/"junqi-portable.zip"
    allow = {"junqi.exe","config/ruleset.json","config/ruleset-selfplay.json","config/observation.json","web/index.html",
             "web/style.css","web/review.js","web/app.js","web/demo.js","Start-MVP.cmd","start-mvp.ps1","README.md"}
    files = {path.relative_to(source).as_posix():path for path in source.rglob("*") if path.is_file()}
    require(set(files)==allow and sum(p.stat().st_size for p in files.values())<40000000,"portable whitelist or size changed")
    require(files["junqi.exe"].read_bytes()==(ROOT/"build/junqi.exe").read_bytes(),"portable executable is stale")
    with ZipFile(archive,"x",ZIP_DEFLATED) as package:
        for name,path in sorted(files.items()):
            package.write(path,name)
    relocated = directory/"relocated copy with spaces"
    with ZipFile(archive) as package:
        require(set(package.namelist())==allow and package.testzip() is None,"portable archive is invalid")
        package.extractall(relocated)
    return dict(modes=check_modes(directory,relocated/"junqi.exe",relocated/"config"),
                packaged_files=len(files),expanded_bytes=sum(p.stat().st_size for p in files.values()),
                compressed_bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


def main(audit):
    out = ROOT/"build/full-mode-baseline"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
        result = compare(out,archive_inputs(False))
        write_receipt(out/"controls.json",result)
    else:
        controls = json.loads((out/"controls.json").read_bytes())
        require(compare(out,archive_inputs(False))==controls,"registered full-mode conditions did not reproduce")
        result = compare(out,archive_inputs(True))
        result["portable"] = portable(out)
        write_receipt(out/"confirmation.json",result)
        print("JUNQI_RESEARCH "+("supported" if result["proofs"]["route_controls"]==2 else "inconclusive"))
    print(json.dumps({"proofs":result["proofs"],"causes":{arm:dict(Counter(r[arm]["cause"] for r in result["rows"].values()))
        for arm in ("0","1","2")},"package":result.get("portable")}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
