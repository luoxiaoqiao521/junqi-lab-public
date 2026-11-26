"""Export player-visible questions and replay their frozen supporting and opposing continuations."""
import html
import json
import math
import shutil
import subprocess
import sys
import time

from confirm_parent_boundaries import ARCHIVE, EXE, PAIRS, REPORT as CONFIRM, ROOT, cases, inputs, read, sha
from parent_update_audit import validate_trace

TICKET = "export-public-decision-contrast"
RUN = ROOT / "build" / (TICKET + "-v2")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")
QUESTIONS = REPORT.with_name("public-decision-exercises.html")
ANSWER = TICKET + ".md"
WITNESSES = ((0, 1, 0), (0, 0, 0), (1, 1, 16), (1, 0, 16))
SEATS = ("南", "东", "北", "西")
RANKS = ("军旗", "地雷", "炸弹", "工兵", "排长", "连长", "营长", "团长", "旅长", "师长", "军长", "司令")
OUTCOMES = ("移动", "进攻方存活", "防守方存活", "双方移除")


def witness(source, old, roster, weights, panel, sample):
    partner = roster["random" if source["context"] == "normal" else "historical"]
    args = [RUN / "junqi.exe", "parent-candidate-exercise", RUN / "ruleset.json",
            source["seed"], source["seat"], source["anchor"], partner["policy"],
            *roster["candidate"]["weights"], *weights[source["child"]], *partner["weights"],
            old["B"]["picked"], panel * PAIRS + sample]
    result = subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True,
                            timeout=20, creationflags=0x40 | 0x08000000)
    row = json.loads(result.stdout)
    validate_trace(source, row)
    for name in ("ply", "legal", "choices", "worlds"):
        if row["point"][name] != old[name]:
            raise ValueError("exercise changed the original legal shortlist")
    return row["point"]["exercise"]


def validate_question(question, source, old):
    if (set(question) != {"seat", "ply", "choices", "nodes", "edges", "prefix"}
            or question["seat"] != source["seat"] or question["ply"] != old["ply"]
            or len(question["nodes"]) != 129 or len(question["prefix"]) != old["ply"]
            or question["choices"] != [old["choices"][i][:2] for i in (0, old["B"]["picked"])]):
        raise ValueError("unexpected visible question schema")
    for row, col, kind, seat, rank in question["nodes"]:
        if (not 0 <= row < 17 or not 0 <= col < 17 or kind not in range(4)
                or seat not in range(-1, 4) or rank not in range(-1, 12)
                or seat == -1 and rank != -1 or seat == source["seat"] and rank == -1
                or seat not in (-1, source["seat"]) and rank not in (-1, 0)):
            raise ValueError("invalid terrain or hidden identity in question")
    for seat, start, end, outcome, out, revealed in question["prefix"]:
        if (seat not in range(4) or start not in range(129) or end not in range(129)
                or outcome not in range(4) or out not in range(16) or revealed not in range(16)):
            raise ValueError("unexpected public event")


def collect():
    _, roster, weights = inputs()
    confirmation = read(CONFIRM)
    declared = list(cases())
    questions, traces = {}, []
    for index, panel, sample in WITNESSES:
        source, old, _ = declared[index]
        exercise = witness(source, old, roster, weights, panel, sample)
        question, branches = exercise["question"], exercise["witness"]
        validate_question(question, source, old)
        if questions.setdefault(index, question) != question:
            raise ValueError("the future response changed the public exercise")
        expected = confirmation["rows"][index]["point"]["confirm"][panel][sample]
        if [branch["score"] for branch in branches] != expected:
            raise ValueError("exercise diverged from the frozen paired result")
        for move, branch in zip(question["choices"], branches, strict=True):
            if (branch["first"][:3] != [source["seat"], *move]
                    or (branch["edited"] and (panel % 2 != 1 or
                                               branch["eligible"] != branch["next"][1:3]))):
                raise ValueError("exercise diverged from the recorded first move or forced response")
        traces.append(dict(seed=source["seed"], panel=panel, sample=sample, branches=branches))
    return list(questions.values()), traces


def board(question):
    nodes = question["nodes"]
    highlighted = {n for move in question["choices"] for n in move}
    for start, end, rail in question["edges"]:
        r, c, *_ = nodes[start]
        r2, c2, *_ = nodes[end]
        width = math.hypot(r2 - r, c2 - c) * 5.5
        angle = math.degrees(math.atan2(r2 - r, c2 - c))
        yield (f'<i class="edge {"rail" if rail else ""}" style="left:{6+c*5.5}%;top:{6+r*5.5}%;'
               f'width:{width:.3f}%;transform:rotate({angle:.3f}deg)" aria-hidden="true"></i>')
    for node, (r, c, kind, seat, rank) in enumerate(nodes):
        mark = ("营" if kind == 1 else "本" if kind == 2 else "·") if seat < 0 else (
            SEATS[seat] + ("暗" if rank < 0 else RANKS[rank][0]))
        desc = f"{node} " + (mark if rank < 0 else SEATS[seat] + RANKS[rank])
        classes = f"point seat{seat} " + ("hq " if kind == 2 else "") + ("highlight" if node in highlighted else "")
        yield (f'<span class="{classes}" style="left:{6+c*5.5}%;top:{6+r*5.5}%" '
               f'title="{html.escape(desc)}">{mark}<small>{node}</small></span>')


def write_question(index, question):
    seat = SEATS[question["seat"]]
    moves = " 或 ".join(f"{a}→{b}" for a, b in question["choices"])
    yield f"<section><h2>局面 {index}</h2><p>{seat}席走。候选 {moves}。先选一手，并记下什么变化会让你改选。</p>"
    yield '<div class="board">'
    yield from board(question)
    yield "</div><details><summary>查看落子前的公开历史</summary><ol>"
    for mover, start, end, outcome, out, flags in question["prefix"]:
        eliminated = "".join(name for i, name in enumerate(SEATS) if out & (1 << i))
        revealed = "".join(name for i, name in enumerate(SEATS) if flags & (1 << i))
        effects = (f"，{eliminated}退出" if eliminated else "") + (f"，{revealed}军旗公开" if revealed else "")
        yield f"<li>{SEATS[mover]} {start}→{end}，{OUTCOMES[outcome]}{effects}</li>"
    yield "</ol></details></section>"


def render(questions):
    css = """
body{margin:2rem auto;padding:0 1rem;max-width:56rem;background:#f8f6ef;color:#1d2a25;font:16px system-ui,sans-serif}
.board{position:relative;aspect-ratio:1;background:#ede7d5;border-radius:.6rem;min-width:300px;max-width:680px}
.point,.edge{position:absolute}.edge{height:1px;background:#a59575;transform-origin:left}
.rail{height:3px;background:#6d6049}.point{display:grid;place-items:center;transform:translate(-50%,-50%);
background:#f9f7ec;border:1px solid #a59575;border-radius:4px;width:5%;height:4.85%;font-size:clamp(7px,1.8vw,13px);line-height:1}
.point small{font-size:clamp(6px,1.45vw,10px)}.highlight{outline:2px solid #101f1b}
.hq{box-shadow:inset 0 0 0 2px #6d6049}
.seat0{background:#f9d3cb}.seat1{background:#cbdcf9}.seat2{background:#c9e8d9}.seat3{background:#f9e4a7}
details{padding:1rem 0}li{margin:.2rem 0}
"""
    head = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    body = ["<title>军棋落子练习</title>", f"<style>{css}</style>",
            "<h1>军棋落子练习</h1><p>深线是铁路，内框是大本营，外框标出候选走法。暗子只标所属方。</p>"]
    for index, question in enumerate(questions, 1):
        body.extend(write_question(index, question))
    body.append(f'<p><a href="{ANSWER}">选好后查看续局与复盘</a></p></html>')
    return head + "\n".join(body) + "\n"


def frozen():
    return dict(confirmation=sha(CONFIRM), engine=sha(RUN / "junqi.exe"),
                rules=sha(RUN / "ruleset.json"))


def run():
    if RUN.exists() or REPORT.exists() or QUESTIONS.exists():
        raise FileExistsError("exercise exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE, RUN / "junqi.exe")
    shutil.copyfile(ARCHIVE / "ruleset.json", RUN / "ruleset.json")
    started = time.monotonic()
    questions, traces = collect()
    data = dict(frozen=frozen(), traces=traces, seconds=time.monotonic() - started)
    outputs = [(REPORT, json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"),
               (QUESTIONS, render(questions))]
    for path, content in outputs:
        payload = content.encode("utf8")
        if len(payload) > 256 * 1024:
            raise ValueError("exercise output exceeded byte cap")
        with path.open("xb") as output:
            output.write(payload)
    print("Recorded two visible exercises and four frozen paired replays")


def audit():
    data = read(REPORT)
    questions, traces = collect()
    if (data["frozen"] != frozen() or data["traces"] != traces
            or QUESTIONS.read_bytes() != render(questions).encode("utf8")):
        raise ValueError("exercise replay or visible-only sheet changed")
    print("PASS frozen witnesses, visible-only question schema and identical question after a changed future")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: parent_decision_exercise.py [audit]")
