"""Offline CPU arena using the published Ataraxos policy and referee kernels."""
import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from build_referee import BUILD, ROOT, SOURCE, PIN, checked_source

WEIGHTS = SOURCE / "pretrained" / "final_run"
RULESET = "ataraxos-classic-upstream-kernels-cpu-v1"
COUNTS = np.array([1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6], dtype=np.uint8)
ACTIONS = np.arange(1800)
ORIGIN = ACTIONS % 100
VERTICAL = ACTIONS < 900
ROW, COLUMN = ORIGIN // 10, ORIGIN % 10
COORD = ACTIONS // 100 % 9
DEST = np.where(VERTICAL, 10 * (COORD + (COORD >= ROW)) + COLUMN,
                10 * ROW + COORD + (COORD >= COLUMN))
LAYOUT = np.repeat(np.arange(12, dtype=np.uint8), COUNTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_models():
    # The original tensor modules only read constants from the CUDA binding.
    package = ModuleType("pyengine.utils")
    package.__path__ = [str(SOURCE / "pyengine" / "utils")]
    package.get_pystratego = lambda: SimpleNamespace(NUM_BOARD_STATE_CHANNELS=355, NUM_PIECE_TYPES=14)
    sys.path.insert(0, str(SOURCE))
    sys.modules[package.__name__] = package
    modules = (
        ("pyengine.networks.legacy_rl", "pyengine/networks/legacy_rl.py"),
        ("pyengine.networks.legacy_init", "pyengine/networks/legacy_init.py"),
        ("pyengine.utils.pretrained_loading", "pyengine/utils/pretrained_loading.py"),
    )
    for name, source_path in modules:
        path = SOURCE / source_path
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        code = checked_source(source_path).replace("SDPBackend.EFFICIENT_ATTENTION", "SDPBackend.MATH")
        code = code.replace('f"cuda:{rank}"', '"cpu"').replace('"cuda"', '"cpu"')
        exec(compile(code, str(path), "exec"), module.__dict__)
    loaders = sys.modules["pyengine.utils.pretrained_loading"]
    arrangement_pickle = WEIGHTS / "ema_arrangements42400.pkl"
    if arrangement_pickle.exists():
        raise RuntimeError("Use the verified tensor weights without loading the optional pickle")
    policy_path = WEIGHTS / "model42400.pthm"
    init_path = WEIGHTS / "init_model42400.pthm"
    for path in (policy_path, init_path):
        checked_blob(path)
    policy, _ = loaders.load_pretrained_rl_model(str(policy_path))
    setup = loaders.load_pretrained_arrangement_model(str(init_path))
    return policy.eval(), setup.eval()


def checked_blob(path):
    import subprocess
    expected = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse",
                                        f"{PIN}:{path.relative_to(SOURCE).as_posix()}"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "hash-object", str(path)], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Checkpoint changed: {path.name}")


def binding():
    library = ctypes.CDLL(str(BUILD / "official_referee.dll"))
    ptr = ctypes.c_void_p
    library.make_match.argtypes, library.make_match.restype = [ptr, ptr], ptr
    library.free_match.argtypes, library.free_match.restype = [ptr], None
    library.query.argtypes, library.query.restype = [ptr, ptr, ptr, ptr], ctypes.c_int
    library.play.argtypes, library.play.restype = [ptr, ctypes.c_int32], ctypes.c_int
    return library


class Match:
    def __init__(self, api, red, blue):
        self.api = api
        self.handle = api.make_match(red.ctypes.data, blue.ctypes.data)
        if not self.handle:
            raise ValueError("Invalid 40-piece deployment")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.api.free_match(self.handle)
        self.handle = None

    def snapshot(self):
        planes = np.empty((387, 10, 10), np.float32)
        ids = np.empty((10, 10), np.uint8)
        mask = np.empty(1800, np.uint8)
        player = self.api.query(self.handle, planes.ctypes.data, ids.ctypes.data, mask.ctypes.data)
        if player not in (1, 2) or not mask.any() or not np.isfinite(planes).all():
            raise RuntimeError(f"Invalid snapshot for player {player}")
        return player, planes, ids, mask.astype(bool)

    def play(self, action):
        return self.api.play(self.handle, int(action))


@torch.inference_mode()
def setup_board(setup, seed):
    rng = torch.Generator().manual_seed(seed)
    seq = torch.zeros(1, 0, 14)
    for _ in range(40):
        probs = setup(seq)["logits"][0, -1].softmax(-1)
        choice = torch.multinomial(probs, 1, generator=rng)
        seq = torch.cat((seq, torch.nn.functional.one_hot(choice, 14).float().reshape(1, 1, 14)), 1)
    ranks = seq.argmax(-1)[0].numpy().astype(np.uint8)
    if sorted(ranks) != sorted(LAYOUT):
        raise RuntimeError("Arrangement model violated the piece inventory")
    return ranks


@torch.inference_mode()
def teacher_policy(network, planes, ids, mask):
    logits, value = network.forward_main(*(torch.from_numpy(x[None]) for x in (planes, ids, mask)))
    if not torch.isfinite(logits[0, mask]).all() or not torch.isneginf(logits[0, ~mask]).all():
        raise RuntimeError("Teacher logits disagree with the referee mask")
    return logits[0].softmax(-1).numpy(), value[0].exp().numpy()


def action_inputs(planes, actions):
    """Adapt the Junqi action-summary idea to the new rule and geometry contract."""
    p = planes.reshape(387, 100)
    origin, dest = ORIGIN[actions], DEST[actions]
    r1, c1, r2, c2 = origin // 10, origin % 10, dest // 10, dest % 10
    own = p[:12, origin].argmax(axis=0)
    enemy_dist = p[12:24, dest].T
    contact = enemy_dist.sum(axis=1) > 0.01
    flag, = np.flatnonzero(p[10] > 0.5)
    distance = lambda r, c: (np.abs(r - flag // 10) + np.abs(c - flag % 10)) / 18
    return np.column_stack((p[41, origin], p[42, origin], own / 11,
                            r1 / 9, r2 / 9, np.abs(c1 - 4.5) / 4.5, np.abs(c2 - 4.5) / 4.5,
                            (r2 - r1) / 9, (np.abs(r2 - r1) + np.abs(c2 - c1)) / 9,
                            distance(r1, c1), distance(r2, c2), contact, p[37, dest],
                            p[40, dest], enemy_dist))


def baseline_score(x):
    return 3 * (x[:, 7] * 9) + x[:, 11] * (3 - 0.5 * x[:, 2] * 11)


def student_score(model, x):
    z = np.clip((x - model["mean"]) / model["scale"], -6, 6)
    net = model["network"]
    hidden = np.tanh(z @ net["weights"][0] + net["bias"][0])
    logits = hidden @ net["weights"][1] + net["bias"][1]
    e = np.exp(logits[:, :3] - logits[:, :3].max(axis=1, keepdims=True))
    return e[:, 2] / e.sum(axis=1)


def trace_game(api, teacher, layouts, seat, rival, model=None, collect=False, forced_prefix=()):
    actions, data = [], []
    shortlist_seen, teacher_moves = 0, 0
    with Match(api, *layouts) as match:
        for step in range(481):
            if step == 480:
                return 4, actions, data, (shortlist_seen, teacher_moves)
            actor, planes, ids, mask = match.snapshot()
            legal = np.flatnonzero(mask)
            if step < len(forced_prefix):
                action = int(forced_prefix[step])
            elif actor == seat:
                prob, _ = teacher_policy(teacher, planes, ids, mask)
                action = int(np.argmax(prob))
                if collect:
                    x = action_inputs(planes, legal)
                    shortlisted = np.argsort(-baseline_score(x), kind="stable")[:4]
                    shortlist_seen += int(action in legal[shortlisted])
                    teacher_moves += 1
                    if teacher_moves % 3 == 1:
                        data.append((x, prob[legal]))
            else:
                x = action_inputs(planes, legal)
                values = baseline_score(x)
                if rival in ("student", "student_four"):
                    allowed = np.argsort(-values, kind="stable")[:4] if rival == "student_four" else np.arange(len(legal))
                    action = legal[allowed[np.argmax(student_score(model, x[allowed]))]]
                else:
                    action = legal[np.argmax(values)]
            result = match.play(action)
            if result < 0:
                raise RuntimeError("Referee rejected the selected move")
            actions.append(int(action))
            if result:
                return result, actions, data, (shortlist_seen, teacher_moves)
    raise AssertionError("Unreachable game loop")


def replay(api, layouts, moves, expected):
    with Match(api, *layouts) as match:
        result = 0
        for move in moves:
            if result:
                raise RuntimeError("Extra action after the recorded terminal")
            result = match.play(move)
            if result < 0:
                raise RuntimeError("Replay encountered an illegal move")
        if result != (0 if expected == 4 else expected):
            raise RuntimeError("Replay changed the outcome")


def isolation(api, layouts):
    first, other = layouts
    edited = other.copy()
    pos = np.flatnonzero(edited != edited[0])[0]
    edited[0], edited[pos] = edited[pos], edited[0]
    with Match(api, first, other) as original, Match(api, first, edited) as swapped:
        a, b = original.snapshot(), swapped.snapshot()
        assert a[0] == b[0] == 1 and all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))
        assert original.play(-1) == -1 and original.play(1800) == -1
        assert all(np.array_equal(x, y) for x, y in zip(a[1:], original.snapshot()[1:]))
        red_action = int(np.flatnonzero(a[3])[0])
        assert original.play(red_action) == swapped.play(red_action) == 0
        second, planes, _, _ = original.snapshot()
        assert second == swapped.snapshot()[0] == 2
        hist = planes[-1].reshape(-1)
        assert hist[99 - ORIGIN[red_action]] == -1 and hist[99 - DEST[red_action]] == 1
    return {"swapped_unseen_enemy_ranks": True, "rejected_illegal_without_mutation": True,
            "rotated_last_move_history": True}


def train(data):
    sys.path.insert(0, str(ROOT.parents[1] / "tools"))
    from compare_book_learning import fit
    raw = np.vstack([x for x, _ in data])
    preference = np.concatenate([p for _, p in data])
    mean, scale = raw.mean(axis=0), raw.std(axis=0).clip(min=1e-6)
    normalized = np.clip((raw - mean) / scale, -6, 6)
    labels = np.full((len(raw), 7), np.nan)
    labels[:, :3] = np.column_stack((1 - preference, np.zeros_like(preference), preference))
    network = fit(normalized, labels, seed=1125, width=24, updates=48)
    control = fit(normalized, labels, seed=1125, width=24, updates=0)
    if network == control:
        raise RuntimeError("Training did not update the student")
    return dict(mean=mean.tolist(), scale=scale.tolist(), network=network, rows=len(raw),
                positions=len(data), encoding="stratego-action-summary-teacher-probability-v1", updates=48)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    teacher, setup = cpu_models()
    api = binding()
    started = time.perf_counter()
    layouts = {s: tuple(setup_board(setup, s * 2 + i) for i in range(2))
               for s in (2101, 2102, 2103, 9101, 9102, 9103)}
    if len({board.tobytes() for pair in layouts.values() for board in pair}) != 12:
        raise RuntimeError("A deployment was reused across layout families")
    (output / "layouts.json").write_text(json.dumps(
        {seed: [x.tolist() for x in pair] for seed, pair in layouts.items()}), encoding="utf-8")
    controls = isolation(api, layouts[2101])
    with Match(api, *layouts[2101]) as match:
        actor, planes, ids, mask = match.snapshot()
        probs, value = teacher_policy(teacher, planes, ids, mask)
        controls.update(first_action=int(np.argmax(probs)), legal_count=int(mask.sum()),
                        first_policy_probability=float(probs.max()), first_value=value.tolist())
    print("first decision", controls, flush=True)
    rows, examples = [], []
    for phase, seeds, arms in (("train", (2101, 2102, 2103), ("base",)),
                               ("heldout", (9101, 9102, 9103), ("base", "student", "student_four"))):
        if phase == "heldout":
            path = output / "student.json"
            path.write_text(json.dumps(train(examples), indent=2), encoding="utf-8")
            student = json.loads(path.read_text(encoding="utf-8"))
        for seed in seeds:
            for seat in (1, 2):
                for arm in arms:
                    begin = time.perf_counter()
                    result, tape, items, coverage = trace_game(api, teacher, layouts[seed], seat, arm,
                                                               student if phase == "heldout" else None,
                                                               phase == "train")
                    replay(api, layouts[seed], tape, result)
                    if phase == "train" and result != 4:
                        examples.extend(items)
                    row = dict(phase=phase, seed=seed, teacher_seat=seat, opponent=arm, result=result,
                               plies=len(tape), actions=tape, teacher_in_baseline_top_four=coverage[0],
                               teacher_turns_measured=coverage[1], seconds=round(time.perf_counter()-begin, 3))
                    rows.append(row)
                    print({k:v for k,v in row.items() if k != "actions"}, flush=True)
                    with (output / "games.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row)+"\n")
    counts = {}
    for arm in ("base", "student", "student_four"):
        counts[arm] = Counter(("unresolved" if r["result"] == 4 else "draw" if r["result"] == 3 else
                              "teacher_win" if r["result"] == r["teacher_seat"] else "learner_win")
                              for r in rows if r["phase"] == "heldout" and r["opponent"] == arm)
    metadata = dict(version=RULESET, commit=PIN, weights={p.name:digest(p) for p in WEIGHTS.glob("*.pthm")},
                    engine=digest(BUILD/"official_referee.dll"), torch=str(torch.__version__), controls=controls,
                    outcomes=counts, seconds=round(time.perf_counter()-started, 2),
                    trainer_rows=student["rows"], training_positions=student["positions"],
                    verified_replays=len(rows), max_experiment_plies=480, teacher_choice="greedy",
                    rule_limits={"max_num_moves":4000,"max_num_moves_between_attacks":200,
                                 "two_square_rule":True,"continuous_chasing_rule":True})
    (output / "receipt.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)


