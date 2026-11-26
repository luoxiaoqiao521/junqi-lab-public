"""Probe one public, safety-screened approach per opening with frozen natural replies."""
from collections import Counter
import json
import os
from statistics import mean, median
import subprocess
import sys

from measure_deadline_opening import EARLY_LOSS_BUDGET, LABEL, ROOT, RULES, SCORE, SCORE_BUDGET
from measure_deadline_opening import build, file_sha, interval, model_roster, require
from serial_worker import write_receipt

START, COUNT = 1203400, 32
KEY = ("seed", "seat", "partner", "table")
OFFER = ("prefix", "ordinary_move", "approach_move", "target", "target_seat", "proved_flag")
FOLLOW = ("next_turn", "piece_present", "legal_attack", "played_attack", "target_exit", "target_reason", "second_enemy")
END = ("plies", "outcome", "ending", "enemy_exit", "friendly_exit", "trace")


def decode(raw, controls):
    rows = []
    for line in raw.splitlines():
        values = list(map(int,line.split()))
        row = dict(zip(KEY,values[:4],strict=True), at=values[4])
        if row["at"]:
            row.update(zip(OFFER,values[5:11],strict=True))
            first,second = values[11:24],values[24:]
            row["base"] = dict(zip((*FOLLOW,*END),first,strict=True))
            row["trial"] = dict(zip((*FOLLOW,*END),second,strict=True))
            require(1<=row["at"]<=140 and row["ordinary_move"]!=row["approach_move"] and row["proved_flag"] in (0,1),
                    "bad public intervention")
            for arm in (row["base"],row["trial"]):
                require(0<=arm["played_attack"]<=arm["legal_attack"]<=int(arm["next_turn"]!=0) and
                        int(arm["target_exit"]!=0)==int(arm["target_reason"]!=0) and
                        all(t==0 or row["at"]<=t<=arm["plies"] for t in
                            (arm["next_turn"],arm["enemy_exit"],arm["target_exit"],arm["second_enemy"])),
                        "bad approach follow-up or retirement event")
        else:
            row["base"] = row["trial"] = dict(zip(END,values[5:],strict=True))
        for arm in (row["base"],row["trial"]):
            require(0<arm["plies"]<=3000 and arm["outcome"] in LABEL and arm["ending"] and
                    all(0<=arm[k]<=arm["plies"] for k in ("friendly_exit","enemy_exit")),"bad ending")
        rows.append(row)
    expected = [(s,j,p,t) for s in range(START,START+COUNT) for j in range(4) for p in (1,3) for t in (0,1)]
    require([tuple(r[k] for k in KEY) for r in rows]==expected,"incomplete opening grid")
    if controls is not None:
        require(all({k:r["base"][k] for k in END}==c["base"] and c["at"]==0
                    for r,c in zip(rows,controls,strict=True)),"base trajectory differs from uninstrumented control")
    return rows


def zero_for_late(value, horizon):
    return value if 0<value<=horizon else 0


def exits(arm, horizon):
    a,b = (zero_for_late(arm[k],horizon) for k in ("enemy_exit","friendly_exit"))
    return "neither" if not a and not b else "tie" if a==b else "enemy_first" if a and (not b or a<b) else "friendly_first"


def stat(rows):
    triggered = [r for r in rows if r["at"]]
    summaries = {}
    for side in ("base","trial"):
        all_games = [r[side] for r in rows]
        touched = [r[side] for r in triggered]
        has_two = [a["second_enemy"]-a["enemy_exit"] for a in touched if a["second_enemy"] and a["enemy_exit"]]
        summaries[side] = dict(
            terminal=dict(Counter(LABEL[a["outcome"]] for a in all_games)),
            touched_terminal=dict(Counter(LABEL[a["outcome"]] for a in touched)),
            observed_length=mean(a["plies"] for a in all_games),
            second_enemy_median=median(has_two) if has_two else None, second_enemy_samples=len(has_two),
            follow={k:sum(bool(a[k]) for a in touched) for k in FOLLOW if k!="target_reason"},
            target_reasons=dict(Counter(str(a["target_reason"]) for a in touched if a["target_exit"])),
            horizons={str(h):dict(results=dict(Counter(LABEL[a["outcome"]] if a["plies"]<=h else "ongoing" for a in all_games)),
                order=dict(Counter(exits(a,h) for a in all_games)),
                target_retired=sum(bool(zero_for_late(a["target_exit"],h)) for a in touched)) for h in (100,140)})
    paired = [r for r in triggered if all(r[k]["outcome"]==2 for k in ("base","trial"))]
    both_won = [r["trial"]["plies"]-r["base"]["plies"] for r in paired]
    paired_exits = [r["trial"]["enemy_exit"]-r["base"]["enemy_exit"] for r in triggered
                    if all(r[k]["enemy_exit"] for k in ("base","trial"))]
    return dict(sources=len({r["seed"] for r in rows}),games=len(rows),triggers=len(triggered),
                trigger_sources=len({r["seed"] for r in triggered}),proved=sum(r["proved_flag"] for r in triggered),
                arms=summaries,both_won=len(both_won),median_win_ply_delta=median(both_won) if both_won else None,
                paired_enemy_exits=len(paired_exits),median_enemy_exit_ply_delta=median(paired_exits) if paired_exits else None,
                transitions=dict(Counter(f'{LABEL[r["base"]["outcome"]]}->{LABEL[r["trial"]["outcome"]]}' for r in triggered)))


def summarize(rows):
    def score_bounds(row):
        a,b = (SCORE[row[k]["outcome"]] for k in ("base","trial"))
        if not row["at"]:
            return 0,0
        return (0 if b is None else b)-(1 if a is None else a), (1 if b is None else b)-(0 if a is None else a)

    def bands(sample):
        groups = [[r for r in sample if r["seed"]==s] for s in sorted({r["seed"] for r in sample})]
        if not groups:
            return None
        low,high = zip(*(tuple(mean(a) for a in zip(*(score_bounds(r) for r in g))) for g in groups))
        return dict(mean_bounds=[mean(low),mean(high)],grouped_95=[interval(low)[0],interval(high)[1]])

    def early_loss_delta(sample):
        groups = {r["seed"]:[] for r in sample}
        for r in sample:
            a,b = (int(r[k]["outcome"]==0 and r[k]["plies"]<=140) for k in ("base","trial"))
            groups[r["seed"]].append(b-a)
        blocks=[mean(g) for g in groups.values()]
        return dict(delta=mean(blocks),grouped_95=interval(blocks))

    return dict(overall=stat(rows),panels={f"{p}/{t}":stat([r for r in rows if r["partner"]==p and r["table"]==t])
                                        for p in (1,3) for t in (0,1)},
                team_score_all=bands(rows),team_score_triggered=bands([r for r in rows if r["at"]]),
                loss_by_140=early_loss_delta(rows),risk_limits=dict(score_loss=SCORE_BUDGET,early_loss=EARLY_LOSS_BUDGET))


def main(audit):
    path = ROOT/"build/first-exit-approach"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe = path/"deadline_opening.exe"
    roster = model_roster()
    models = [roster[name] for name in ("candidate","historical")]
    payload = " ".join(map(str,(START,COUNT,*(v for model in models for v in model["weights"]))))
    def run(mode):
        result = subprocess.run([exe,RULES,mode],cwd=ROOT,input=payload,encoding="utf8",capture_output=True,timeout=110)
        require(result.returncode==0,f"{mode} failed: {result.stderr.strip()}")
        return result.stdout
    unchanged = run("first-exit-approach-control")
    paired = run("first-exit-approach")
    rows = decode(paired,decode(unchanged,None))
    result = dict(engine=file_sha(exe),rules=file_sha(RULES),models=models,summary=summarize(rows),rows=rows)
    for name,raw in (("control.txt",unchanged),("paired.txt",paired)):
        if audit:
            require(raw==(path/name).read_text(encoding="utf8"),f"{name} did not replay")
        else:
            (path/name).write_text(raw,encoding="utf8")
    if audit:
        require(result==json.loads((path/"paired.json").read_text(encoding="utf8")),"derived results did not replay")
    else:
        write_receipt(path/"paired.json",result)
    print(json.dumps(result["summary"],indent=2))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
