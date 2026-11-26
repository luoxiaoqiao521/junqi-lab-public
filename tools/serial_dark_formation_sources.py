"""Search strictly replayed formation samples, holding identity weights constant."""
import json
import os
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_formation_library import prepare, select_roots
from serial_dark_search_budget import decode, finish_roots, tally
from serial_dynamic_reply import ROOT, require
from serial_worker import read, write_receipt
from train_early_deadline_student import model_arguments


PREVIOUS=ROOT/"build/dark-formation-library/audit-dark-formation-library-and-search-cross-e146136be7520bea6fae"
ARMS=("fixed","exact","swaps")
NATIVE_ARMS={0:"fixed",5:"exact",6:"swaps"}


def prepare_sources(scope):
    previous=read(PREVIOUS/"manifest.json")["cohorts"]
    unseen=select_roots((r["family"] for group in previous for r in group),count=12)
    prepare(scope,[previous[0],unseen[0]+unseen[1]])


def parse_source_arms(raw,cohort):
    search=[]
    counts={}
    for line in raw.splitlines():
        tag,key,*fields=line.split()
        if tag=="J":
            arm,side,exact,swap=map(int,fields)
            require(0<=exact and 0<=swap and exact+swap<=2 and (arm or not swap),"invalid formation rollout counts")
            counts.setdefault(key,{})[ARMS[1+arm],side]=(exact,swap)
        elif tag not in ("L","X","F"):
            search.append(line)
    roots=decode("\n".join(search),cohort,NATIVE_ARMS)
    active=finish_roots(roots,expect_breadth=False)
    require(counts.keys()=={r["key"] for r in active},"formation execution did not cover all verified roots")
    for r in active:
        groups=counts[r["key"]]
        require(len(groups)==4 and set(r["arms"])==set(ARMS),"lost paired search arm")
        r["source_counts"]={name:[list(groups[name,s]) for s in range(2)] for name in ARMS[1:]}
        for name in ARMS[1:]:
            if not any(sum(x) for x in r["source_counts"][name]):
                require(r["arms"][name]==r["arms"]["fixed"],"fallback changed the original continuation")
    return roots,active


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-formation-sources"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare_sources(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"formation source inputs changed")
    fold=int(audit)
    model=read(folder/"student.json")
    tape="\n".join((model_arguments(model,"collision",model["encoding"]),
                    (folder/"library.txt").read_text(encoding="ascii"),
                    (folder/f"replay-{fold}.txt").read_text(encoding="ascii")))
    begin=perf_counter()
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-formation-sources",tape)
    roots,active=parse_source_arms(raw,read(folder/"manifest.json")["cohorts"][fold])
    summary=dict(roots=len(roots),completed=len(active),seconds=perf_counter()-begin,profiles=tally(active,ARMS),
                 outcomes=sum(len(r["outcomes"]) for r in active),
                 truncated=sum(y[-2]<0 for r in active for y in r["outcomes"]),
                 sources={name:[sum(r["source_counts"][name][side][kind] for r in active)
                                for side in range(2) for kind in range(2)] for name in ARMS[1:]})
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(roots=roots,summary=summary))
    if audit:
        supported=any(all(t["interval"][0]>0 for t in arm["tables"].values()) for arm in summary["profiles"].values())
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    main(bool(sys.argv[1:]))
