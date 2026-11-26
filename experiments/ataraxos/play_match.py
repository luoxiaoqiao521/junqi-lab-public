"""Play the saved Junqi-style student against the frozen teacher, once per color."""
import argparse
import json
from pathlib import Path
import time

import torch

from arena import BUILD, RULESET, binding, cpu_models, digest, replay, setup_board, trace_game


def play(seed, weight_file, output):
    model = json.loads(weight_file.read_text(encoding="utf-8"))
    if model.get("encoding") != "stratego-action-summary-teacher-probability-v1":
        raise ValueError("Expected an arena student model")
    torch.set_num_threads(3)
    teacher,setup = cpu_models()
    api = binding()
    layouts = tuple(setup_board(setup, seed * 2 + i) for i in range(2))
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(profile=RULESET, seed=seed, student=digest(weight_file),
                   referee=digest(BUILD / "official_referee.dll"),
                   layouts=[board.tolist() for board in layouts], games=[])
    for seat in (1,2):
        started = time.perf_counter()
        result,actions,_,_ = trace_game(api, teacher, layouts, seat, "student", model)
        replay(api, layouts, actions, result)
        row = dict(teacher_seat=seat,result=result,plies=len(actions),elapsed=round(time.perf_counter()-started,3),
                   actions=actions)
        receipt["games"].append(row)
        print({k:v for k,v in row.items() if k != "actions"},flush=True)
    (output / "match.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed", type=int)
    cli.add_argument("student", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    play(args.seed, args.student, args.output)
