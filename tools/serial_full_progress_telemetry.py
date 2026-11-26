"""Reproduce frozen games and measure movement, repetition and win conversion."""
from collections import Counter
import json
import os
from statistics import mean, median
import sys

from serial_dynamic_reply import ROOT, PRIOR, build, native, require
from serial_engineer_formation import formation_payload
from serial_full_student_evolution import GAME_KEYS, STARTS
from serial_quiet_options import FROZEN
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

FORM = ROOT/"build/engineer-formation/compare-engineer-deployment-student-interaction-0f84f1dfb1ba9fcdf0ee"
STATE = ("repeated","first_contact","first_flag","first_enemy_flag","ending","decision_micros","reversals_final70")
BY_SEAT = ("turns","quiet","reversals","with_contact")
OUTCOMES = {-1:"unresolved",0:"loss",1:"draw",2:"win"}


def analyze(raw, originals, reference):
    verified={}
    accepted={}
    for tag,key,*fields in map(str.split,raw.splitlines()):
        if tag=="A":
            accepted[key]=tuple(map(int,fields[-2:]))
        if tag not in ("G","T"):
            continue
        table,arm,*values=map(int,fields)
        idx=key,table,arm
        if tag=="G":
            row=dict(zip(GAME_KEYS[3:],values,strict=True))
            old=originals[idx]
            require(idx not in verified and all(old[k]==v for k,v in row.items() if k!="micros"),
                    f"original game diverged at {idx}")
            verified[idx]=dict(row,table=table,arm=arm)
        else:
            trace=dict(zip(STATE,values[:len(STATE)],strict=True))
            rest=values[len(STATE):]
            seats=[dict(zip(BY_SEAT,rest[i:i+4],strict=True)) for i in range(0,len(rest),4)]
            row=verified[idx]
            require("trace" not in row and len(seats)==4 and sum(s["turns"] for s in seats)==row["plies"] and
                    all(0<=s["with_contact"]<=s["reversals"]<=s["quiet"]<=s["turns"] for s in seats) and
                    0<=trace["first_contact"]<=row["plies"] and 0<=trace["first_flag"]<=row["plies"] and
                    0<=trace["first_enemy_flag"]<=row["plies"] and 0<trace["ending"]<=5 and
                    0<=trace["repeated"]<row["plies"] and trace["reversals_final70"]<=sum(s["reversals"] for s in seats),
                    f"telemetry accounting failed at {idx}")
            row["trace"]=dict(trace,seats=seats)
    require(verified.keys()==originals.keys() and all("trace" in r for r in verified.values()) and accepted==reference,
            "missing episode or archived opening evidence")
    return [{"key":key,**row} for (key,_,_),row in verified.items()]


def distribution(values):
    return dict(n=len(values),mean=mean(values),median=median(values),p90=sorted(values)[(9*len(values)+9)//10-1]) if values else dict(n=0)


def metrics(rows):
    counts=dict(Counter(OUTCOMES[r["result"]] for r in rows))
    elapsed={}
    for limit in (100,140):
        deadline=Counter()
        for r in rows:
            deadline[OUTCOMES[r["result"]] if r["plies"]<=limit else "in_progress_at_horizon"]+=1
        elapsed[str(limit)]=dict(deadline)
    seats={}
    for label,indexes in (("learner",(0,)),("partner",(2,)),("opponents",(1,3)),("all",range(4))):
        seats[label]={k:sum(r["trace"]["seats"][i][k] for r in rows for i in indexes) for k in BY_SEAT}
    ending=Counter(str(r["trace"]["ending"]) for r in rows)
    return dict(games=len(rows),outcomes=counts,horizons=elapsed,seats=seats,endings=ending,
                plies=distribution([r["plies"] for r in rows]),
                by_outcome={label:distribution([r["plies"] for r in rows if r["result"]==j]) for j,label in OUTCOMES.items()},
                first_contact=distribution([r["trace"]["first_contact"] for r in rows if r["trace"]["first_contact"]]),
                first_retirement=distribution([r["first_exit"] for r in rows if r["first_exit"]]),
                first_flag=distribution([r["trace"]["first_flag"] for r in rows if r["trace"]["first_flag"]]),
                flag_to_win=distribution([r["plies"]-r["trace"]["first_enemy_flag"] for r in rows if
                                          r["result"]==2 and r["trace"]["first_enemy_flag"]]),
                wins_without_enemy_flag=sum(r["result"]==2 and not r["trace"]["first_enemy_flag"] for r in rows),
                repeated_positions=sum(r["trace"]["repeated"] for r in rows),
                games_with_repetition=sum(r["trace"]["repeated"]>0 for r in rows),
                last70_draw_reversals=sum(r["trace"]["reversals_final70"] for r in rows if r["result"]==1),
                decision_micros=distribution([r["trace"]["decision_micros"] for r in rows]))


def panel(folder, audit):
    label="confirmation" if audit else "development"
    parent=read(PRIOR/"students.json")["models"]["window"]
    child=read(FROZEN/"student.json")["models"]["window"]
    frozen=read(FROZEN/(label+".json"))["rows"]
    by_key={(str(r["seed"]),r["table"],r["arm"]):r for r in frozen}
    models=model_wire(parent,"window")+" "+model_wire(child,"window")
    evolution=analyze(native(folder,"full-student-evolution-trace",f"{STARTS[audit]} 16 "+models),by_key,{})
    blocks=read(FORM/(label+".json"))["accepted"]
    expected={(k,g["table"],g["arm"]):g for k,r in blocks.items() for g in r["games"]}
    _,payload=formation_payload(parent,audit)
    archived_hashes={k:(r["original_hash"],r["shifted_hash"]) for k,r in blocks.items()}
    formation=analyze(native(folder,"engineer-student-formation-trace",payload),expected,archived_hashes)
    require(len(evolution)==240 and len(formation)==64,"frozen game count changed")
    return dict(evolution=evolution,formation=formation)


def summary(data):
    result={name:metrics(rows) for name,rows in data.items()}
    result["combined"]=metrics(data["evolution"]+data["formation"])
    result["evolution_by_table"]={f"{t}/{a}":metrics([r for r in data["evolution"] if (r["table"],r["arm"])==(t,a)])
                                  for t in range(3) for a in range(5)}
    result["formation_by_cell"]={str(a):metrics([r for r in data["formation"] if r["arm"]==a]) for a in range(4)}
    return result


def main(audit):
    folder=ROOT/"build/full-progress-telemetry"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    result=panel(folder,audit)
    label="confirmation" if audit else "development"
    write_receipt(folder/(label+".json"),result)
    if audit:
        old=read(folder/"development.json")
        combined={name:old[name]+result[name] for name in result}
        for name,rows in combined.items():
            require(len(rows)==len({(r["key"],r["table"],r["arm"]) for r in rows}),f"replay blocks overlap in {name}")
        derived=summary(combined)
        write_receipt(folder/"summary.json",derived)
        print(json.dumps(derived["combined"]))
        print("JUNQI_RESEARCH supported")
    else:
        print(json.dumps(metrics(result["evolution"]+result["formation"])))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
