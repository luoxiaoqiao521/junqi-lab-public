"""Opt-in isolated execution of the common full-information research stages."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


current_scope=ContextVar("full_mode_scope",default=None)
ACTIVE_MODELS={
    "junqi-public-early-action-v1",
    "junqi-public-early-action-collision-v1",
    "junqi-full-reply-window-action-v1",
}
ACTIVE_PEER_MODES={"full-leaf-teacher-data","visited-reply-value","visited-lesson-review","full-scoring-audit"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv(data,value=0xCBF29CE484222325):
    for byte in data:
        value=((value^byte)*0x100000001B3)&0xffffffffffffffff
    return value


def model_hashes(payload):
    tokens=payload.split()
    hashes=[]
    for i,name in enumerate(tokens):
        if name not in ACTIVE_MODELS:
            continue
        width,heads,hidden=map(int,tokens[i+1:i+4])
        count=2*width+width*hidden+hidden+(hidden or width)*heads+heads
        values=map(float,tokens[i+4:i+4+count])
        hashes.append((name,fnv(struct.pack(f"<{count}d",*values),fnv(name.encode("ascii")))))
    return hashes


class FullModeRun:
    def __init__(self,folder):
        self.folder=folder.resolve()
        self.engine=(self.folder/"engine").resolve()
        self.binary=self.engine/"deadline_opening.exe"
        self.rules=self.engine/"rules.json"
        self.log=self.folder/"runtime.jsonl"

    def run(self,mode,payload="",*,expected=None,strict=True):
        env=dict(os.environ)
        if strict:
            env["JUNQI_FULL_RESEARCH_ACTIVE"]="1"
        else:
            env.pop("JUNQI_FULL_RESEARCH_ACTIVE",None)
        run=subprocess.run([self.binary,self.rules,mode],input=payload,encoding="utf8",errors="replace",
                           capture_output=True,timeout=95,cwd=self.engine,env=env)
        if expected is not None:
            code,marker=expected
            if code!=run.returncode or marker not in run.stderr:
                raise ValueError(f"negative control {mode} did not return {code} with {marker}")
            self.record(dict(control=mode,returncode=code,guarded=strict))
            return run.stdout
        if run.returncode:
            raise ValueError(f"active command {mode} failed ({run.returncode}): {run.stderr.strip()}")
        if not strict:
            self.record(dict(control=mode,returncode=0,guarded=False))
            return run.stdout
        trace=[line.split() for line in run.stderr.splitlines() if line.startswith("JUNQI_ACTIVE_")]
        paths=[(t[1],int(t[2])) for t in trace if t[0]=="JUNQI_ACTIVE_PATH"]
        loaded=[(t[1],int(t[2])) for t in trace if t[0]=="JUNQI_ACTIVE_MODEL"]
        peers=[int(t[1]) for t in trace if t[0]=="JUNQI_ACTIVE_PEER"]
        expected_peers=[fnv(struct.pack("<6d",*map(float,payload.split()[:6])))] if mode in ACTIVE_PEER_MODES else []
        if len(paths)!=1 or paths[0][0]!=mode or loaded!=model_hashes(payload) or peers!=expected_peers:
            raise ValueError(f"runtime did not verify the executed path and loaded artifacts for {mode}")
        self.record(dict(mode=paths[0][0],engine_sha=sha(self.binary),config_sha=sha(self.rules),rules_hash=paths[0][1],
                         loaded=loaded,peer=peers,input_sha=hashlib.sha256(payload.encode()).hexdigest(),
                         result_sha=hashlib.sha256(run.stdout.encode()).hexdigest(),returncode=run.returncode))
        return run.stdout

    def record(self,row):
        with self.log.open("a",encoding="utf8") as file:
            file.write(json.dumps(row,ensure_ascii=False)+"\n")


@contextmanager
def isolated(scope,source_build):
    def guard(event,args):
        current=current_scope.get()
        if event!="open" or current is not scope or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(source_build) and not path.is_relative_to(scope.folder):
            raise PermissionError(f"active research could not read undeclared old build artifact {path.name}")

    sys.addaudithook(guard)
    token=current_scope.set(scope)
    try:
        yield
    finally:
        current_scope.reset(token)
