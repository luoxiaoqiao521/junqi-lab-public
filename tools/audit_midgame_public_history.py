"""Audit the fixed takeover experiment with restored public history and matched timing."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

from audit_qq_archive import INPUT, decode, local_records, public_records
from compare_archive_cycle_exit_values import interval
from pilot_relay_candidate_budget import ROOTS, file_sha, sha
from probe_replay_midgame_takeover import CONFIG, RULES, ROOT, compile_native, parse, prepare, read

PREVIOUS = ROOT / "build/leading-archive-skip/support-leading-archive-skip-fa842e847af984cd79fa"
HISTORY_FIELDS = "moved restricted singletons legal human_mass_ppm higher ties relays aliases human_relay".split()
HORIZONS = (16, 40, 80, 240, 480, 960)


def require(condition, text):
    if not condition:
        raise ValueError(text)


def distribution(values):
    return dict(count=len(values), minimum=min(values), mean=mean(values),
                median=median(values), maximum=max(values))


def game_lengths(records):
    games = {}
    excluded = Counter()
    for raw in records:
        try:
            key, _, _, events = decode(raw)
        except ValueError as error:
            excluded[str(error)]+=1
            continue
        if key in games:
            excluded["duplicate"]+=1
            continue
        if (events[-1][:2]!=bytes((245,5)) or
                any(event[0] not in (95,245) or event[:2]==bytes((245,5)) for event in events[:-1])):
            excluded["missing_or_inconsistent_terminal"]+=1
            continue
        moves = sum(event[0]==95 for event in events)
        if moves == 0:
            excluded["no_moves"]+=1
            continue
        games[key]=moves
    report = distribution(list(games.values()))
    report.update(shortest=min((n,key[:16]) for key,n in games.items())[1],
                  longest=min(key[:16] for key,n in games.items() if n==report["maximum"]),
                  excluded=dict(excluded))
    return report


def archive_lengths():
    locator = Path(read(INPUT)["corpus"])
    return dict(public=game_lengths(public_records(locator)),
                local=game_lengths(raw for directory in ROOTS for raw in local_records(directory)),
                public_source_sha256=file_sha(locator))


def fields(output, old_rows):
    prior, history, tempo = [], {}, {}
    for line in output.splitlines(keepends=True):
        tag, key, *cells = line.split()
        if tag=="P":
            identity=(key,int(cells[0]))
            require(identity not in history,"duplicate public history root")
            history[identity]=dict(zip(HISTORY_FIELDS,map(int,cells[1:]),strict=True))
        elif tag=="Q":
            ply,table,rep,mode,contacts,quiet=map(int,cells)
            identity=(key,ply,table,rep,mode)
            require(identity not in tempo,"duplicate measured continuation")
            tempo[identity]=(contacts,quiet)
        else:
            prior.append(line)
    require("".join(prior)==old_rows,"restoring public history changed the frozen paired trajectories")
    return history,tempo


def history_summary(history,roots):
    require(history.keys()==roots.keys(),"one or more takeover roots did not restore public history")
    for key,data in history.items():
        require(data["legal"]==roots[key]["legal"] and data["ties"]>=1 and data["restricted"]>=data["singletons"],
                "legal candidates or public constraints did not survive replay")
    rows=list(history.values())
    contact=[row for key,row in history.items() if roots[key]["human_contact"]]
    return dict(roots=len(rows), every_human_move_legal=True,
                history_restricted_dark=distribution([r["restricted"] for r in rows]),
                publicly_moved_dark=distribution([r["moved"] for r in rows]),
                history_singleton_roots=sum(r["singletons"]>0 for r in rows),
                human_move_strictly_below_top3=sum(r["higher"]>=3 for r in rows),
                human_contact_below_top3=sum(r["higher"]>=3 for r in contact),
                human_contact_mass_median=median(r["human_mass_ppm"] for r in contact)/1000000,
                roots_with_public_relay=sum(r["relays"]>0 for r in rows),
                roots_with_same_features_but_different_relay=sum(r["aliases"]>0 for r in rows))


def tempo_summary(rows,measurements):
    total=sum(row["plies"] for row in rows)
    outcomes=Counter(row["result"] for row in rows)
    result=dict(steps=distribution([r["plies"] for r in rows]),
                known_win_steps=distribution([r["plies"] for r in rows if r["result"]==2]),
                unknown=outcomes[-1], contact_per100=sum(c for c,_ in measurements)*100/total,
                longest_quiet_median=median(q for _,q in measurements))
    for name in ("contact","enemy_exit","win"):
        result[name+"_by"]=[sum((r["result"]==2 and r["plies"]<=n) if name=="win" else 0<r[name]<=n
                                for r in rows) for n in HORIZONS]
    return result


def by_table(trials,tempo,human):
    expected=set(trials)|{(key,ply,-1,0,0) for key,ply in human}
    require(tempo.keys()==expected,"one or more timing rows missing")
    report={"human":tempo_summary(list(human.values()),[tempo[k+(-1,0,0)] for k in human])}
    for table in (0,1):
        report[str(table)]={}
        for mode,policy in enumerate(("parent","trained","human_first")):
            keys=[k for k in trials if k[2]==table and k[4]==mode]
            report[str(table)][policy]=tempo_summary([trials[k] for k in keys],[tempo[k] for k in keys])
    return report


def score_limits(old,new):
    lower=lambda value: max(0,value)/2
    upper=lambda value: 1 if value<0 else value/2
    return lower(new)-upper(old),upper(new)-lower(old)


def paired(trials,table,old,new):
    blocks=defaultdict(list)
    wins,transitions=Counter(),Counter()
    wins_by={n:0 for n in HORIZONS}
    same=0
    for (key,ply,t,rep,mode),a in trials.items():
        if t!=table or mode!=old:
            continue
        b=trials[key,ply,t,rep,new]
        blocks[key].append(score_limits(a["result"],b["result"]))
        transitions[f"{a['result']}->{b['result']}"]+=1
        if a["result"]==b["result"]==2:
            wins["faster" if b["plies"]<a["plies"] else "slower" if b["plies"]>a["plies"] else "equal"]+=1
        for n in HORIZONS:
            wins_by[n]+=int(b["result"]==2 and b["plies"]<=n)-int(a["result"]==2 and a["plies"]<=n)
        if old==1 and (a["first_from"],a["first_to"])==(b["first_from"],b["first_to"]):
            require(dict(a,mode=new)==b,"same first action diverged despite matching rollout policy")
            same+=1
    bounds=list(zip(*[tuple(map(mean,zip(*rows))) for rows in blocks.values()],strict=True))
    return dict(pairs=sum(transitions.values()), source_families=len(blocks), team_score=[mean(x) for x in bounds],
                source_resample_range=[interval(bounds[0])[0],interval(bounds[1])[1]], results=dict(transitions),
                both_won_speed=dict(wins), additional_wins_by=list(wins_by.values()), identical_first=same)


def replay(out):
    old=read(PREVIOUS/"summary.json")["evidence"]
    old_rows=(PREVIOUS/"rows.txt").read_text(encoding="utf8")
    feed, manifest=prepare()
    require(manifest==old["manifest"],"weights, archive or rules do not match the frozen run")
    executable=out/"takeover.exe"
    run=subprocess.run([executable,CONFIG,RULES,"960","public-history"],input=feed,text=True,
                       capture_output=True,encoding="utf8",errors="replace",timeout=160,check=False)
    require(run.returncode==0 and not run.stderr,f"public history replay failed: {run.stderr.strip()}")
    history,tempo=fields(run.stdout,old_rows)
    roots,human,trials,_=parse(old_rows,manifest["sources"],960)
    scores={str(table):dict(trained_minus_parent=paired(trials,table,0,1),
                            human_first_minus_trained=paired(trials,table,1,2)) for table in (0,1)}
    result=dict(history=history_summary(history,roots), tempo=by_table(trials,tempo,human),
                paired=scores, horizons=HORIZONS, lengths=archive_lengths(), frozen_feed_sha256=manifest["feed_sha256"],
                engine_sha256=file_sha(executable), rows_sha256=sha(run.stdout.encode("utf8")))
    return json.loads(json.dumps(result)),run.stdout


def main(audit):
    key=os.environ.get("JUNQI_SERIAL_KEY","manual")
    require(key and all(c.isalnum() or c=="-" for c in key),"invalid serial task key")
    out=ROOT/"build/midgame-public-history"/key
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        compile_native(out)
    result,raw=replay(out)
    path=out/"summary.json"
    if audit:
        require(read(path)==result and (out/"rows.txt").read_text(encoding="utf8")==raw,
                "timing or public history changed between runs")
    else:
        (out/"rows.txt").write_text(raw,encoding="utf8")
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf8")
        tmp.replace(path)
    print(json.dumps(result,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
