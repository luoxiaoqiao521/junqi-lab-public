"""Verify leading JGS skips in the shipping importer and the takeover cohort."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from probe_replay_midgame_takeover import ROOT, CONFIG, compile_native, parse, prepare, read, sample
from pilot_relay_candidate_budget import file_sha

SOURCE = "faa88c8dde3c31e6"
PREVIOUS = ROOT / "build/replay-midgame-horizon/extend-replay-midgame-horizon-95c15a7872280096d149"
OBSERVATION = ROOT / "config/observation.json"


def require(condition,message):
    if not condition:
        raise ValueError(message)


def cli_result(key,sources,out):
    feed,_ = prepare()
    tapes = dict(line.split() for line in feed.splitlines()[1:])
    engine = ROOT / "build/junqi.exe"
    with tempfile.TemporaryDirectory(dir=out) as directory:
        raw = Path(directory) / "source.jgs"
        replay = Path(directory) / "archived.jsonl"
        source = bytes.fromhex(tapes[key])
        raw.write_bytes(source)
        args = [engine,"import-jgs",raw,replay,CONFIG,OBSERVATION]
        result = subprocess.run(args,capture_output=True,text=True,encoding="utf8",errors="replace",timeout=20)
        require(result.returncode==0,f"the witnessed archive failed to import: {result.stderr.strip()}")
        result = subprocess.run([engine,"replay",replay],capture_output=True,text=True,encoding="utf8",
                                errors="replace",timeout=20)
        require(result.returncode==0,f"the imported archive failed to verify: {result.stderr.strip()}")
        lines = replay.read_text(encoding="utf8").splitlines()
        header = json.loads(lines[0])
        first_control = header["source_controls"][0]
        require(first_control[:3]==[0,1,2] and header["source_steps"][:2]==[0,2],
                "the skip consumed a move or lost the source sequence")
        invalid = bytearray(source)
        invalid[416] = (invalid[416]+2)%4
        replay.unlink()
        raw.write_bytes(invalid)
        bad = subprocess.run(args,capture_output=True,text=True,encoding="utf8",errors="replace",timeout=20)
        require(bad.returncode!=0 and not replay.exists(),"a mismatched next seat was silently accepted")
    return dict(anonymous_replay=sources[key]["replay"], source_control=first_control,
                native_engine_sha256=file_sha(engine), read_back_frames=len(lines)-1)


def verify_cohort(data,rows,key):
    old = read(PREVIOUS / "summary.json")["evidence"]
    require(old["manifest"]==data["manifest"],"the frozen cohort or models differ")
    sources = old["manifest"]["sources"]
    before = parse((PREVIOUS/"rows.txt").read_text(encoding="utf8"),sources,960)
    now = parse(rows,sources,960)
    require(all({k:v for k,v in new.items() if (k[0] if isinstance(k,tuple) else k)!=key}==prior_filtered
                for new,prior_filtered in zip(now,(
                    before[0],before[1],before[2],{k:v for k,v in before[3].items() if k!=key}),strict=True)),
            "a previously accepted source or trajectory changed")
    roots = [ply for family,ply in now[0] if family==key]
    require(roots==[96,144] and not now[3][key]["failed"],"witness did not recover both takeover roots")
    outcomes = {str(ply):now[1][key,ply] for ply in roots}
    return dict(recovered_roots=roots,previous_roots=len(before[0]),now_roots=len(now[0]),
                checked_unmodified_trials=len(before[2]),human_suffix=outcomes)


def main(audit):
    key = os.environ.get("JUNQI_SERIAL_KEY","manual")
    require(key and all(c.isalnum() or c=="-" for c in key),"invalid local serial key")
    out=ROOT / "build/leading-archive-skip" / key
    out.mkdir(parents=True,exist_ok=True)
    if not audit:
        compile_native(out)
    data,rows=sample(out,960)
    sources=data["manifest"]["sources"]
    witness,= [key for key,source in sources.items() if source["replay"]==SOURCE]
    receipt=dict(importer=cli_result(witness,sources,out),cohort=verify_cohort(data,rows,witness),
                 evidence=data)
    path=out / "summary.json"
    if audit:
        require(read(path)==receipt and (out/"rows.txt").read_text(encoding="utf8")==rows,
                "the opening skip result or cohort did not reproduce")
    else:
        (out/"rows.txt").write_text(rows,encoding="utf8")
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf8")
        temporary.replace(path)
    print(json.dumps(dict(importer=receipt["importer"],cohort=receipt["cohort"]),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
