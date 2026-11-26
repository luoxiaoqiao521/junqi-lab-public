"""Test predeclared book-based identity guesses against strictly replayed JGS games."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from time import perf_counter

from archive_action_screen import layout_group
from audit_human_timing import sanitized_tape
from audit_qq_archive import INPUT, LIMIT, ROOT, decode
from compare_qq_rules import VC

DEFAULT_OUT = ROOT / "build/book-identity-priors/study-1"
ARCHIVE_CAP = 768
FIELDS = ("ply", "actor", "from", "to", "verified_rank", "allowed_mask", "context", "stratum")
REDACTED_FIELDS = ("seat", "square", "verified_rank")
RANKS = ("flag", "mine", "bomb", "engineer", "platoon", "company", "battalion",
         "regiment", "brigade", "division", "corps", "marshal")
FLAG, MINE, BOMB, ENGINEER, PLATOON, COMPANY, BATTALION, REGIMENT, BRIGADE, DIVISION, CORPS, MARSHAL = range(12)


def choose(candidates, stats):
    families = {}
    for raw, name, source in candidates:
        stats["files"] += 1
        try:
            if len(raw) > LIMIT:
                raise ValueError("oversize")
            _, layouts, _, events = decode(raw)
        except ValueError:
            stats["unsupported_format"] += 1
            continue
        tape = sanitized_tape(raw, events)
        key = sha256(bytes.fromhex(tape)).hexdigest()
        family = layout_group(layouts).hex()
        row = dict(key=key, family=family, source=source, archive_record=name)
        existing = families.get(family)
        if existing is not None:
            stats["same_layout_family"] += 1
        if existing is None or key < existing[0]["key"]:
            families[family] = (row, tape)
    return families


def local_inputs(inputs):
    files = {path.resolve() for key in ("local", "annotated")
             for path in Path(inputs[key]).rglob("*") if path.suffix.lower() == ".jgs" and path.is_file()}
    files.add(Path(inputs["supplemental"]).resolve())
    for path in sorted(files):
        yield path.read_bytes(), None, "local_double_tag" if "双明" in str(path.parent) else "local"


def archive_inputs(path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = Path(member.name)
            if member.isfile() and name.suffix.lower() == ".jgs" and member.size <= LIMIT:
                record = name.stem if name.stem.isdigit() else None
                yield archive.extractfile(member).read(), record, "archive"


def sources(inputs):
    counts = {"local": Counter(), "archive": Counter()}
    local = choose(local_inputs(inputs), counts["local"])
    archive = choose(archive_inputs(inputs["corpus"]), counts["archive"])
    independent = sorted(archive.keys()-local.keys())
    counts["archive"]["overlaps_local_family"] = len(archive)-len(independent)
    selected = [local[key] for key in sorted(local)] + [archive[key] for key in independent[:ARCHIVE_CAP]]
    return dict((meta["key"], meta) for meta, _ in selected), selected, counts


def build(directory):
    sources = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
               "tests/book_piece_priors.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{source}"' for source in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'{files} /Fe:book_piece_priors.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=directory, check=True, timeout=90)


def replay(directory, indexed, selected):
    with (directory/"input.log").open("w+", encoding="ascii") as feed:
        for meta, tape in selected:
            print(meta["key"], tape, file=feed)
        feed.seek(0)
        run = subprocess.run([directory/"book_piece_priors.exe"], stdin=feed, cwd=ROOT,
                             capture_output=True, encoding="ascii", timeout=230)
    if run.returncode:
        raise RuntimeError(f"Native audit failed: {run.stderr.strip()}")
    evidence, accepted, rejected = [], {}, set()
    for line in run.stdout.splitlines():
        kind, key, *fields = line.split()
        meta = indexed[key]
        if kind == "G":
            accepted[key] = None
        elif kind == "P":
            accepted[key] = int(fields[0])
        elif kind == "R":
            rejected.add(key)
        else:
            keys = REDACTED_FIELDS if kind in ("H", "F") else FIELDS
            row = dict(zip(keys, map(int, fields), strict=True))
            if not 0 <= row["verified_rank"] < len(RANKS):
                raise ValueError("invalid native label")
            evidence.append(meta | {"kind": kind} | row)
    if accepted.keys() | rejected != indexed.keys() or any(value is None for value in accepted.values()):
        raise ValueError("the archive audit omitted games or unfinished evidence")
    return evidence, accepted, rejected


def estimate(rows, wanted):
    by_game = defaultdict(lambda: [0, 0])
    for row in rows:
        totals = by_game[row["key"]]
        totals[0] += wanted(row["verified_rank"])
        totals[1] += 1
    counts = list(by_game.values())
    hits, total = map(sum, zip(*counts)) if counts else (0, 0)
    result = dict(hits=hits, total=total, games=len(counts))
    if counts:
        result["rate"] = round(hits/total, 6)
    if len(counts) >= 5 and 0 < hits < total:
        rng = random.Random(221122)
        rates = []
        for _ in range(800):
            sampled = rng.choices(counts, k=len(counts))
            a, b = map(sum, zip(*sampled))
            rates.append(a/b)
        rates.sort()
        result["game_cluster_interval"] = [round(rates[20], 6), round(rates[779], 6)]
    return result


def camp_strata(rows):
    strata = defaultdict(Counter)
    for row in rows:
        x = strata[row["stratum"]]
        side = row["context"]
        x[side] += 1
        x[side+2] += row["verified_rank"] == BOMB
    sets = [(s, min(s[0], s[1])) for s in strata.values() if s[0] and s[1]]
    count = sum(weight for _, weight in sets)
    if count:
        return dict(common_strata=len(sets), per_group_weight=count,
                    non_camp=sum(s[2]/s[0]*w for s,w in sets)/count,
                    camp=sum(s[3]/s[1]*w for s,w in sets)/count)
    return {}


def describe(rows):
    indexed = {kind: [r for r in rows if r["kind"] == kind] for kind in "HFDCE"}
    choices = {}
    is_bomb = lambda rank: rank == BOMB
    for phase, early in (("early", True), ("later", False)):
        division = [r for r in indexed["D"] if (r["ply"] <= 40) == early]
        for name, direction in (("was_attacked", 0), ("attacked", 1)):
            local = [r for r in division if r["context"] == direction]
            ambiguous = [r for r in local if r["allowed_mask"] == (1<<CORPS)|(1<<MARSHAL)]
            choices[f"division_{phase}_{name}_ambiguous"] = estimate(ambiguous,lambda rank: rank==MARSHAL)
            choices[f"division_{phase}_{name}_all"] = estimate(local,lambda rank: rank==MARSHAL)
            if direction:
                choices[f"division_{phase}_{name}_mine"] = estimate(local,lambda rank: rank==MINE)
    engineers = [r for r in indexed["E"] if r["allowed_mask"] == (1<<BOMB)|(1<<ENGINEER)]
    for name, direction in (("attacked", 0), ("was_attacked", 1)):
        local = [r for r in engineers if r["context"] == direction]
        choices[f"engineer_{name}_ambiguous"] = estimate(local,is_bomb)
        choices[f"engineer_{name}_rail_ambiguous"] = estimate([r for r in local if r["stratum"]],is_bomb)
    for value, label in enumerate(("comparable_road","camp")):
        choices[f"opening_{label}_bomb"] = estimate([r for r in indexed["C"] if r["context"]==value],is_bomb)
    choices["camp_position_adjusted"] = camp_strata(indexed["C"])
    for name, pred in (("mine",lambda r:r==MINE),("bomb",is_bomb),("regiment_up",lambda r:r>=REGIMENT)):
        choices["false_hq_"+name] = estimate(indexed["H"],pred)
    choices["front_company_or_lower"] = estimate(indexed["F"],lambda r:ENGINEER<=r<=COMPANY)
    choices["front_division_or_higher"] = estimate(indexed["F"],lambda r:r>=DIVISION)
    return choices


def save(path, contents):
    with path.open("x", encoding="utf8", newline="\n") as handle:
        handle.write(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose an unused audit output directory.")
    start = perf_counter()
    inputs = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    inputs["annotated"] = str(Path(inputs["local"]).parent/"联众赌局复盘-军棋的艺术赏析")
    indexed, selected, counts = sources(inputs)
    with TemporaryDirectory(prefix="junqi-book-priors-") as directory:
        directory = Path(directory)
        build(directory)
        rows, accepted, rejected = replay(directory,indexed,selected)
    panels = {}
    for name, predicate in (("local",lambda source:source.startswith("local")),
                            ("archive",lambda source:source=="archive"),
                            ("local_double_tag",lambda source:source=="local_double_tag")):
        panels[name] = dict(games=sum(predicate(indexed[key]["source"]) for key in accepted),
                            plies=sum(ply for key,ply in accepted.items() if predicate(indexed[key]["source"])),
                            rejected=sum(predicate(indexed[key]["source"]) for key in rejected),
                            hypotheses=describe([row for row in rows if predicate(row["source"])]))
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/"cases.jsonl","".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
    receipt = dict(version=1, rank_order=RANKS, input_counts=counts, panels=panels,
                   cases=len(rows), public_family_cap=ARCHIVE_CAP,
                   evidence_sha256=sha256((args.output/"cases.jsonl").read_bytes()).hexdigest())
    save(args.output/"summary.json",json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(f"PASS book identity audit {len(accepted)} games {len(rows)} rows in {perf_counter()-start:.2f}s")


if __name__ == "__main__":
    main()
