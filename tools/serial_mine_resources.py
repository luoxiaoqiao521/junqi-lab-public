"""Isolate attacker rank on all parsed mine diagrams and three legal root moves."""
from collections import Counter
import json
import os
import sys

from serial_route_choice import ROOT, build, native, panels, require, write_receipt


def measure(directory):
    catalog = panels()
    wire = "".join(" ".join(map(str,[r["id"],*r["mines"]]))+"\n" for r in catalog)
    raw = native(directory,"route-resources",wire)
    require(all(len(r)==10 and r[0]=="R" for r in raw),"invalid resource proof record")
    records = [list(map(int,r[1:])) for r in raw]
    require([r[:2] for r in records]==[[r["id"],rank] for r in catalog for rank in (3,11)],"missing resource arm")
    require(all(0<=lo<=hi<=2 for r in records for lo,hi in zip(r[3::2],r[4::2],strict=True)),"invalid team interval")
    return dict(catalog=catalog,records=records)


def summary(data):
    report={}
    for rank,name in ((3,"engineer"),(11,"marshal")):
        rows=[row for row in data["records"] if row[1]==rank]
        outcomes={}
        for index,route in ((3,"top"),(5,"outside"),(7,"inside")):
            counts=Counter("win" if r[index]==2 else "loss" if r[index+1]==0 else "unresolved" for r in rows)
            outcomes[route]=dict(wins=counts["win"],losses=counts["loss"],unresolved=counts["unresolved"])
        report[name]=outcomes
    marshals = data["records"][1::2]
    blocked = [r for r in marshals if r[4]==0]
    report["marshal_top_loses"] = dict(
        roots=len(blocked), viable_detour=sum(r[5]==2 or r[7]==2 for r in blocked),
        no_viable_tested_route=sum(r[6]==0 and r[8]==0 for r in blocked),
        unresolved=sum(max(r[5],r[7])<2 and max(r[6],r[8])>0 for r in blocked))
    return report


def main(audit):
    out=ROOT/"build/mine-resources"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
    data=measure(out)
    report=summary(data)
    if audit:
        prior=json.loads((out/"proofs.json").read_bytes())
        require(data==prior["data"] and report==prior["summary"],"resource proof did not reproduce")
        changed=sum(a[3:]!=b[3:] for a,b in zip(data["records"][::2],data["records"][1::2],strict=True))
        write_receipt(out/"confirmation.json",dict(changed_forms=changed,summary=report))
        print("JUNQI_RESEARCH "+("supported" if changed else "inconclusive"))
    else:
        write_receipt(out/"proofs.json",dict(data=data,summary=report))
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
