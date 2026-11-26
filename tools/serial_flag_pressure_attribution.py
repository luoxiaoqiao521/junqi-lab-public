"""Attribute possible flag replies on frozen trajectories without emitting training labels."""
from collections import Counter
import json
import os
import sys
from time import perf_counter

from serial_public_flag_reply import (PARENT, PRESSURE, ROOT, build, game_rows, model_wire,
                                      native, require)
from serial_worker import read, write_receipt

PRIOR = ROOT/"build/public-flag-reply/test-publicly-verifiable-flag-defense-nomination-84956a50b86ceaa03734/confirmation.json"
FRESH = [(34129100+i, i%4, table) for i in range(15) for table in range(2)]
ROOT_FIELDS = ("seed desk ply actor chosen exact legal listed outside public_risk public_unknown "
               "public_clear draw native_unknown first_referee_clear first_public_clear").split()
REPLY_FIELDS = ("available", "chosen", "target_removed")
ATTACK_FIELDS = ("exact_offers", "captured", "public_offers", "public_captured")
NATIVE_LABELS = ("clear", "threat", "win", "draw", "loss", "unknown")


def parse(lines, grid):
    probes, games, follow, attack = [], [], {}, {}
    for line in lines:
        tag, *fields = line.split()
        if tag == "R":
            row = dict(zip(ROOT_FIELDS, map(int, fields), strict=True))
            safe = row["listed"]+row["outside"]
            require(0 <= row["exact"] < len(NATIVE_LABELS) and
                    safe == row["public_clear"]+row["public_risk"]+row["public_unknown"] and
                    row["legal"] >= safe+row["draw"]+row["native_unknown"] and
                    bool(safe) == (row["first_referee_clear"] >= 0) and
                    bool(row["public_clear"]) == (row["first_public_clear"] >= 0),
                    "invalid exact/public matrix")
            probes.append(row)
        elif tag == "P":
            seed, table, *values = map(int, fields)
            require((seed,table) not in follow and len(values) == 10, "duplicate or incomplete follow-through")
            follow[seed,table] = [dict(zip(REPLY_FIELDS, values[i:i+3], strict=True)) for i in (0,3)]
            exact, taken, public, known = values[6:]
            require(0 <= known <= min(taken, public) and max(taken,public) <= exact,
                    "attack opportunity accounting differs")
            attack[seed,table] = dict(zip(ATTACK_FIELDS, values[6:], strict=True))
            require(all(0 <= r["target_removed"] <= r["chosen"] <= r["available"] for r in follow[seed,table]),
                    "follow-through opportunity accounting differs")
        else:
            games.append(line)
    rules_hash, rows = game_rows(games, grid, (0,))
    by_game = {(r["seed"],r["desk"]):r for r in rows}
    found = Counter((r["seed"],r["desk"]) for r in probes)
    actual = Counter((r["seed"],r["desk"]) for r in probes if r["exact"] == 1)
    require(set(follow) == set(by_game) and all(
        found[k] == r["danger"] and follow[k][0]["chosen"] == r["ours"]["partner_replies"] and
        follow[k][1]["chosen"] == r["opponent"]["partner_replies"] for k,r in by_game.items()) and
        all((r["seed"],r["desk"]) in by_game and 0 < r["ply"] <= by_game[r["seed"],r["desk"]]["plies"]
            for r in probes) and len({(r["seed"],r["desk"],r["ply"]) for r in probes}) == len(probes),
        "missing risk root or follow-through event")
    require(all(0 <= actual[k]-attack[k]["exact_offers"] <= (r["plies"] == 720)
                for k,r in by_game.items()), "the next enemy opportunity is missing")
    return dict(rules_hash=rules_hash,games=rows,roots=probes,
                follow=[follow[k] for k in by_game],attack=[attack[k] for k in by_game])


def summarize(data):
    out = {}
    for desk in (0,1):
        probes = [r for r in data["roots"] if r["desk"] == desk]
        adverse = [r for r in probes if r["exact"] == 1]
        group = dict(Counter(NATIVE_LABELS[r["exact"]] for r in probes))
        follow = [f for g,f in zip(data["games"],data["follow"],strict=True) if g["desk"] == desk]
        attack = [a for g,a in zip(data["games"],data["attack"],strict=True) if g["desk"] == desk]
        out[str(desk)] = dict(public_risk=len(probes), referee=group, distinct_threat_layouts=len({
            r["seed"] for r in adverse}), within_shortlist=sum(bool(r["listed"]) for r in adverse),
            outside_only=sum(r["outside"]>0 and not r["listed"] for r in adverse),
            without_continuing_or_winning_reply=sum(not (r["outside"] or r["listed"]) for r in adverse),
            draw_available=sum(r["draw"]>0 for r in adverse),
            native_safe_without_public_proof=sum((r["outside"] or r["listed"])>0 and
                                                  not r["public_clear"] for r in adverse),
            public_safe=sum(bool(r["public_clear"]) for r in adverse),
            winning_or_continuing_actions={name:sum(r[name] for r in adverse) for name in
                                          ("listed","outside","public_clear","public_unknown","public_risk")},
            follow=[{k:sum(f[side][k] for f in follow) for k in REPLY_FIELDS} for side in (0,1)],
            counterattack={k:sum(r[k] for r in attack) for k in ATTACK_FIELDS},
            outcomes=dict(Counter(r["result"] for r in data["games"] if r["desk"] == desk)))
    return out


def main(audit):
    start = perf_counter()
    run_dir = ROOT/"build/flag-pressure-attribution"/os.environ["JUNQI_SERIAL_KEY"]
    run_dir.mkdir(parents=True,exist_ok=True)
    prior = read(PRIOR)
    grid = FRESH if audit else PRESSURE
    if not audit:
        build(run_dir)
        require(native(run_dir,"controls") == ["CONTROL public-flag-reply"], "public reply controls changed")
    model = model_wire(read(PARENT))
    wire = model+"\n"+"\n".join(" ".join(map(str,row)) for row in grid)+"\n"
    data = parse(native(run_dir,"attribute",wire),grid)
    expected = [r for r in prior["games"] if r["arm"] == 0]
    require(data["rules_hash"] == prior["rules_hash"], "stress rules changed")
    if not audit:
        require(data["games"] == expected, "diagnostics changed a previously frozen action or outcome")
    else:
        require(not {r["layout"] for r in data["games"]} & {r["layout"] for r in expected},
                "fresh study shares a development layout")
        repeat = parse(native(run_dir,"attribute",model+"\n"+" ".join(map(str,grid[0]))+"\n"),grid[:1])
        require(repeat["games"] == data["games"][:1] and
                repeat["roots"] == [r for r in data["roots"] if (r["seed"],r["desk"]) == (grid[0][0],grid[0][2])] and
                repeat["follow"] == data["follow"][:1] and repeat["attack"] == data["attack"][:1],
                "fresh risk and follow-through did not reproduce")
    data["summary"] = summarize(data)
    data["seconds"] = round(perf_counter()-start,3)
    write_receipt(run_dir/("fresh.json" if audit else "reproduced.json"),data)
    print(json.dumps(data["summary"],sort_keys=True))
    if audit:
        supported = sum(r["distinct_threat_layouts"] for r in data["summary"].values()) >= 3
        print("JUNQI_RESEARCH " + ("supported" if supported else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    main(sys.argv[1:] == ["audit"])
