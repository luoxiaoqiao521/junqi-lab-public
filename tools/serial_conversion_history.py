"""Save and restore the four-arm handoff prefixes through a fresh process."""
import json
import os
import subprocess
import sys
from time import perf_counter

from measure_deadline_opening import ROOT, RULES, build, file_sha, model_roster, require
from serial_first_exit_approach import COUNT, START
from serial_worker import write_receipt

CROSS = ROOT/"build/conversion-cross/audit-first-exit-conversion-interaction-b569e0fa2a8e91c137a7/cross.txt"
BLIND = ROOT/"build/blind-extension/audit-blind-conditional-extension-ed8ac99813e7a2a21983/blind.json"


def main(audit):
    path=ROOT/"build/conversion-history"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe=path/"deadline_opening.exe"
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    payload=" ".join(map(str,(START,COUNT,*weights)))+"\n"
    def run(mode,journal=""):
        return subprocess.run([exe,RULES,mode],cwd=ROOT,input=payload+journal,encoding="utf8",capture_output=True,timeout=130)
    begun=perf_counter()
    exported=run("export-conversion-history")
    require(exported.returncode==0,f"journal export failed: {exported.stderr.strip()}")
    journal=path/"prefixes.txt"
    if audit:
        require(journal.read_text(encoding="utf8")==exported.stdout,"saved prefix journal changed")
    else:
        journal.write_text(exported.stdout,encoding="utf8")
    saved=journal.read_text(encoding="utf8")
    restored=run("restore-conversion-history",saved)
    require(restored.returncode==0,f"journal restore failed: {restored.stderr.strip()}")
    require(restored.stdout==CROSS.read_text(encoding="utf8"),"restored four arms differ from original checkpoints")
    lines=saved.splitlines()
    first=lines[0].split()
    require(len(lines)==COUNT*32 and all(int(v.split()[12])==len(v.split())-13 for v in lines),"missing prefix records")
    first[8]=str(int(first[8])^1)
    rejected=run("restore-conversion-history"," ".join(first)+"\n")
    require(rejected.returncode!=0 and "conversion journal checkpoint mismatched" in rejected.stderr,
            "tampered seat-history fingerprint was not rejected")
    blind=json.loads(BLIND.read_text(encoding="utf8"))["summary"]
    receipt=dict(engine=file_sha(exe),rules=file_sha(RULES),journal=file_sha(journal),cross=file_sha(CROSS),
                 prefixes=len(lines),per_seat_observed_histories=4*len(lines),plies=sum(int(line.split()[12]) for line in lines),
                 restored_continuations=2*len(lines),tampered_history_rejected=True,
                 student_evidence=dict(confirmed_ranking_changes=blind["different_order"]-blind["different_order_unresolved"],
                                       confirmation_scope="development-only; requires independent positive and counterexample before training"))
    if audit:
        require(receipt==json.loads((path/"history.json").read_text(encoding="utf8")),"restored history receipt changed")
    else:
        write_receipt(path/"history.json",receipt)
    print(json.dumps(receipt,indent=2))
    print(f"history checks seconds={perf_counter()-begun:.2f}")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
