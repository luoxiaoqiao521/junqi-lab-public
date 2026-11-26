"""Check the local backtracking signature on existing human replay events."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import sys
import time

from archive_action_screen import layout_group
from audit_local_selfplay_cycle_entry import RESULT as SIM
from audit_public_attack_windows import FORMAL_ENDS, write_json
from audit_qq_archive import HQ, INPUT, archived_move, decode, initial_board, local_records, public_records
from calibrate_selfplay_evaluation_signal import NAMES, OUT
from test_defense_opportunity_cost import new_sources

RESULT = OUT / "archive-local-cycle.json"
COHORTS = ("public","local","curated")


@dataclass
class Edge:
    origin: tuple = ()
    target: tuple = ()
    start: int = 0
    span: int = 0

    def record(self, origin, target, ply, permitted):
        if not permitted:
            self.span=0
        else:
            if self.span and self.target==origin and self.origin==target:
                self.span+=1
            else:
                self.start,self.span=ply,1
            self.origin,self.target=origin,target


@dataclass
class ReplayTiming:
    first: int = 0
    confirmed: int = 0
    quiet_start: int = 0
    quiet_at: int = 0
    quiet_clock: int = 0
    warning: int = 0
    warning_seat: int = -1
    checked: int = 0
    prediction: int = -1

    def record(self, edge, actor, ply, clock):
        quiet=edge.span>=3 and clock>=ply-edge.start
        if self.warning and not self.checked and actor==self.warning_seat:
            self.checked=ply
            self.prediction=int(quiet and edge.span==4)
        if not self.warning and quiet and edge.span==3:
            self.warning,self.warning_seat=ply,actor
        if not self.confirmed and edge.span==4:
            self.first,self.confirmed=edge.start,ply
        if not self.quiet_at and quiet and edge.span==4:
            self.quiet_start,self.quiet_at,self.quiet_clock=edge.start,ply,clock


def replay(layouts, seats, events):
    board=initial_board(layouts)
    active=set(range(4))
    edges=[Edge() for _ in range(4)]
    timing=ReplayTiming()
    ply=quiet=0
    ends=[]
    interruptions=0
    for e in events:
        if e[0]==95:
            if ends:
                raise ValueError("moves_after_end")
            origin,target=(e[2],16-e[3]),(e[4],16-e[5])
            actor,contact=archived_move(board,e)
            ply+=1
            edges[actor].record(origin,target,ply,not contact and target not in HQ)
            timing.record(edges[actor],actor,ply,quiet)
            quiet=0 if contact else quiet+1
            continue
        if e[0]!=245:
            raise ValueError("event_type")
        kind,code,cause=e[1:4]
        if kind==5:
            ends.append(cause)
        elif kind==2:
            interruptions+=1
            edges=[Edge() for _ in range(4)]
            quiet=0
        elif kind in (3,4) and code in seats:
            actor=seats.index(code)
            if actor in active:
                active.remove(actor)
                board={point:piece for point,piece in board.items() if piece[0]!=actor}
                edges=[Edge() for _ in range(4)]
    if len(ends)!=1 or ends[0] not in FORMAL_ENDS:
        raise ValueError("unresolved_terminal")
    record=vars(timing).copy()
    record.pop("warning_seat")
    return dict(record, plies=ply, end=ends[0], quiet_tail=quiet, interruptions=interruptions)


def collect():
    paths=json.loads(INPUT.read_text(encoding="utf-8-sig"))
    streams=(public_records(Path(paths["corpus"])), local_records(Path(paths["local"])), new_sources())
    scope,rows,seen={},{},set()
    for name,records in zip(COHORTS,streams,strict=True):
        counts,results=Counter(),[]
        for raw in records:
            counts["read"]+=1
            try:
                key,layouts,seats,events=decode(raw)
                if key in seen:
                    counts["duplicate"]+=1
                    continue
                seen.add(key)
                item=replay(layouts,seats,events)
            except ValueError as e:
                counts[str(e)]+=1
                continue
            results.append(dict(item,key=key,layout=layout_group(layouts).hex()))
        counts["kept"]=len(results)
        scope[name],rows[name]=dict(counts),results
    return scope, rows


def metrics(rows):
    complete=[r for r in rows if r["quiet_at"]]
    threshold=[r for r in rows if r["plies"]>=150]
    predictions=Counter(str(r["prediction"]) for r in rows if r["warning"])
    return dict(games=len(rows),four_moves=sum(r["confirmed"]>0 for r in rows),
                quiet_four=len(complete),
                start_median=median([r["quiet_start"] for r in complete]) if complete else None,
                confirmation_median=median([r["quiet_at"] for r in complete]) if complete else None,
                earliest=min(r["quiet_at"] for r in complete) if complete else None,
                predictions=dict(predictions),
                prefix_150=[sum(0<r["quiet_at"]<=150 for r in threshold),len(threshold)])


def summarize(rows,sim):
    profiles={}
    for source,records in rows.items():
        by_layout={}
        for r in sorted(records,key=lambda item:item["key"]):
            by_layout.setdefault(r["layout"],r)
        profiles[source]=dict(**metrics(records),layout_balanced=metrics(list(by_layout.values())),
                              draw_code=metrics([r for r in records if r["end"]==2]),
                              quiet_71_draws=metrics([r for r in records if r["end"]==2 and r["quiet_tail"]==71]),
                              decided_code=metrics([r for r in records if r["end"]==4]))
    for t,name in enumerate(NAMES):
        records=[dict(q,plies=g["plies"]) for g,q in zip(sim["games"],sim["local"],strict=True) if g["table"]==t]
        profiles["sim_"+name]=metrics([dict(r,confirmed=r["established"],quiet_start=r["quiet_first"]) for r in records])
    return profiles


def audit(rescan=True):
    data=json.loads(RESULT.read_text(encoding="utf8"))
    sim=json.loads(SIM.read_text(encoding="utf8"))
    if rescan:
        scope,rows=collect()
        assert (scope,rows)==(data["scope"],data["rows"])
    all_rows=[r for seq in data["rows"].values() for r in seq]
    assert len({r["key"] for r in all_rows})==len(all_rows)
    for r in all_rows:
        assert not r["confirmed"] or 0<r["first"]<r["confirmed"]<=r["plies"]
        assert not r["quiet_at"] or (
            0<r["quiet_start"]<r["quiet_at"]<=r["plies"] and
            r["confirmed"]<=r["quiet_at"] and 0<r["quiet_at"]-r["quiet_start"]<=r["quiet_clock"] and
            0<r["warning"]<r["quiet_at"])
        assert (r["prediction"]==-1)==(r["checked"]==0)
        assert not r["checked"] or r["warning"]<r["checked"]<=r["plies"]
    assert data["summary"]==summarize(data["rows"],sim)
    print("PASS source audit and chronological replay",len(all_rows))
    print(json.dumps({k:[v["games"],v["quiet_four"],v["prefix_150"]] for k,v in data["summary"].items()},sort_keys=True))


def run():
    start=time.perf_counter()
    scope,rows=collect()
    sim=json.loads(SIM.read_text(encoding="utf8"))
    write_json(RESULT,dict(scope=scope,rows=rows,summary=summarize(rows,sim),
                           elapsed_seconds=time.perf_counter()-start))
    audit(False)


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
