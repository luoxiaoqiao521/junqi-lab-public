"""Replay pinned upstream referee fixtures and inspect already recorded arena games."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from arena import (DEST, LAYOUT, ORIGIN, Match, SOURCE, action_inputs, baseline_score,
                   binding, checked_source, cpu_models, replay, student_score, teacher_policy)


def encode(origin, dest, actor):
    if actor == 2:
        origin, dest = 99 - origin, 99 - dest
    candidates = np.flatnonzero((ORIGIN == origin) & (DEST == dest))
    action, = candidates
    return int(action)


def two_square(api):
    layout = LAYOUT.copy()
    scout = int(np.flatnonzero(layout == 1)[0])
    layout[scout], layout[39] = layout[39], layout[scout]
    tape = [(39,49), (60,50), (49,39), (50,60), (39,49), (60,50)]
    with Match(api, layout, layout) as match:
        for i,(src,dst) in enumerate(tape):
            actor,_,_,mask = match.snapshot()
            assert actor == i % 2 + 1 and mask[encode(src,dst,actor)]
            assert match.play(encode(src,dst,actor)) == 0
        actor,_,_,mask = match.snapshot()
        assert not mask[encode(49,39,actor)]
        assert mask[encode(49,59,actor)] and mask[encode(49,48,actor)]
        assert match.play(encode(49,39,actor)) == -1
    return len(tape)


def chase(api):
    folder = SOURCE / "tests" / "continuous_chase_games_new"
    chars = "CDEFGHIJKLMB"
    blues = "OPQRSTUVWXYN"
    total = 0
    for path in sorted(folder.glob("*.txt"))[:10]:
        board,*actions = checked_source(path.relative_to(SOURCE).as_posix()).splitlines()
        red = np.array([chars.index(c) for c in board[:40]], dtype=np.uint8)
        blue = np.array([blues.index(c) for c in board[60:][::-1]], dtype=np.uint8)
        tape = list(map(int, actions))
        with Match(api, red, blue) as match:
            for i,action in enumerate(tape[:-1],1):
                if match.play(action) != 0:
                    raise RuntimeError(f"{path.name} upstream chase fixture diverges at {i}")
            actor,_,_,mask = match.snapshot()
            if mask[tape[-1]] or match.play(tape[-1]) != -1:
                raise RuntimeError(f"{path.name} upstream forbidden chase accepted")
        total += len(tape)-1
    return total


def action_string(action):
    return [int(ORIGIN[action]),int(DEST[action])]


def read_run(directory):
    rows = [json.loads(line) for line in (directory / "games.jsonl").read_text().splitlines()]
    layouts = {int(key): tuple(np.asarray(x,dtype=np.uint8) for x in pair) for key,pair in
               json.loads((directory / "layouts.json").read_text()).items()}
    return rows, layouts


def audit(directory, api):
    rows, layouts = read_run(directory)
    network,_ = cpu_models()
    student = json.loads((directory / "student.json").read_text())
    torch.set_num_threads(3)
    count = top4 = learned = 0
    examples, aliases, finishing = [], [], []
    for row in rows:
        replay(api, layouts[row["seed"]], row["actions"], row["result"])
        if row["phase"] != "heldout":
            continue
        if row["opponent"] == "student" and row["result"] in (1,2) and row["result"] != row["teacher_seat"]:
            end_actor = (len(row["actions"])-1)%2+1
            end_action = row["actions"][-1]
            flag = int(np.flatnonzero(layouts[row["seed"]][2-end_actor] == 10)[0])
            finishing.append(dict(seed=row["seed"], plies=row["plies"], final_actor=end_actor,
                                  action=action_string(end_action), captured_flag=bool(DEST[end_action] == 99-flag)))
        if row["opponent"] != "base":
            continue
        with Match(api,*layouts[row["seed"]]) as match:
            teacher_turns = 0
            for ply,chosen in enumerate(row["actions"],1):
                actor, planes, ids, mask = match.snapshot()
                if actor == row["teacher_seat"]:
                    prob,_ = teacher_policy(network,planes,ids,mask)
                    assert int(np.argmax(prob)) == chosen
                    actions = np.flatnonzero(mask)
                    x = action_inputs(planes,actions)
                    route = baseline_score(x)
                    idx = np.argsort(-route,kind="stable")[:4]
                    pred = student_score(student,x)
                    in_four = chosen in actions[idx]
                    top4 += in_four
                    learned += int(actions[np.argmax(pred)] == chosen)
                    count += 1
                    teacher_turns += 1
                    if not in_four and prob[chosen] > 0.75 and DEST[chosen]//10 <= ORIGIN[chosen]//10 and len(examples) < 10:
                        if np.all(DEST[actions[idx]]//10 > ORIGIN[actions[idx]]//10):
                            examples.append(dict(seed=row["seed"],ply=ply,seat=actor,teacher=action_string(chosen),
                                                 prob=float(prob[chosen]),candidate_picks=[action_string(a) for a in actions[idx]]))
                    _, inv, multiplicity = np.unique(x,axis=0,return_inverse=True,return_counts=True)
                    for group in np.flatnonzero(multiplicity>1):
                        candidates = actions[inv==group]
                        hi,lo = int(candidates[np.argmax(prob[candidates])]),int(candidates[np.argmin(prob[candidates])])
                        if prob[hi]-prob[lo] > .2 and len(aliases) < 10:
                            aliases.append(dict(seed=row["seed"],ply=ply,seat=actor,high=action_string(hi),
                                                low=action_string(lo),probs=[float(prob[hi]),float(prob[lo])],
                                                features=x[np.flatnonzero(inv==group)[0]].tolist()))
                match.play(chosen)
                if teacher_turns == 20:
                    break
    result = dict(replayed_games=len(rows), heldout_teacher_decisions=count, route_top4_contains=top4,
                  student_all_legal_matches=learned, backwards_or_sideways=examples,
                  feature_aliases=aliases, student_wins=finishing)
    (directory / "decision-audit.json").write_text(json.dumps(result, indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path)
    args = parser.parse_args()
    api = binding()
    if args.run:
        audit(args.run,api)
    else:
        result = {"two_square_prefix":two_square(api), "upstream_chase_prefix_plies":chase(api)}
        print(json.dumps(result))
