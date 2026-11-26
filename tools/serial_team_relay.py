"""Prove the conditions for an engineer relay with four acting seats."""
import json
import os
import sys

from serial_route_choice import ROOT, build, native, require, write_receipt


def main(audit):
    out = ROOT/"build/team-relay"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
    raw = native(out,"team-relay","")
    require([row[:4] for row in raw[:3]]==[["T","0","2","2"],["T","1","0","0"],["T","2","0","0"]] and
            raw[3:]==[["V","execute","decline","intercept","exhaustion","shared-legals"]],"relay check did not settle cleanly")
    evidence = dict(proofs=[list(map(int,row[1:])) for row in raw[:3]],witnesses=raw[3][1:])
    if audit:
        require(evidence==json.loads((out/"proofs.json").read_bytes()),"same roots did not reproduce")
        write_receipt(out/"confirmation.json",evidence)
        print("JUNQI_RESEARCH supported")
    else:
        write_receipt(out/"proofs.json",evidence)
    print(json.dumps(evidence))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
