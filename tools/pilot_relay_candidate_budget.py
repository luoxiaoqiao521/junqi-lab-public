"""Freeze a local development-only JGS pilot and audit the native three-slot comparison."""
import os
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from compare_qq_rules import VC

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path(os.environ.get("JUNQI_LOCAL_ARCHIVE_ROOT", REPO / "data/local-archive"))
ROOTS = (ARCHIVE_ROOT / "annotated", ARCHIVE_ROOT / "replays")
PLIES = (48, 96, 144)
JGS_HEADER = 412
JGS_ROW = 10
LIMIT = 65536


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def setup(repo):
    sys.path.insert(0, str(repo / "tools"))
    from archive_action_screen import layout_group
    from audit_qq_archive import SEAT_BLOCKS, valid_grid
    from partner_crossplay import model_roster
    return layout_group, SEAT_BLOCKS, valid_grid, model_roster


def header_fields(header, size, blocks, valid_grid):
    if not JGS_HEADER <= size <= LIMIT or len(header) != JGS_HEADER or not header.startswith(b"QQGame JQS"):
        raise ValueError("signature_or_size")
    count = int.from_bytes(header[26:28], "little")
    if count < 2 or JGS_HEADER + JGS_ROW * count > size:
        raise ValueError("event_length")
    layouts = [header[offset:offset+30] for offset in blocks]
    if not all(valid_grid(layout) for layout in layouts):
        raise ValueError("incomplete_layout")
    if {header[offset-28] for offset in blocks} != set(range(4)):
        raise ValueError("seat_codes")
    return layouts, count


def corpus_families(corpus, layout_group, blocks, valid_grid):
    families = set()
    with tarfile.open(corpus, "r:gz") as archive:
        for member in archive:
            if (not member.isfile() or not JGS_HEADER <= member.size <= LIMIT
                    or member.name.rsplit("/", 1)[-1].startswith("._")):
                continue
            with archive.extractfile(member) as reader:
                header = reader.read(JGS_HEADER)
            try:
                layouts, _ = header_fields(header, member.size, blocks, valid_grid)
            except ValueError:
                continue
            families.add(layout_group(layouts))
    if not families:
        raise ValueError("historical corpus contains no complete JGS layouts")
    return families


def development_paths(roots, excluded, layout_group, blocks, valid_grid):
    chosen = {}
    screen = Counter()
    for source, root in enumerate(roots):
        if not root.is_dir():
            raise ValueError(f"source {source} directory is missing")
        paths = sorted(root.rglob("*"), key=lambda p: (p.as_posix().casefold(), p.as_posix()))
        for path in paths:
            if path.suffix.lower() != ".jgs" or not path.is_file():
                continue
            screen[f"{source}/seen"] += 1
            try:
                with path.open("rb", buffering=0) as reader:
                    header = reader.read(JGS_HEADER)
                layouts, count = header_fields(header, path.stat().st_size, blocks, valid_grid)
            except (OSError, ValueError) as error:
                reason = error if isinstance(error, ValueError) else "unreadable"
                screen[f"{source}/{reason}"] += 1
                continue
            family = layout_group(layouts)
            if family in excluded:
                screen[f"{source}/corpus_overlap"] += 1
                continue
            if int.from_bytes(family[:4], "big") % 4 == 0:
                screen[f"{source}/holdout_header_only"] += 1
                continue
            if family in chosen:
                screen[f"{source}/development_duplicate_family"] += 1
                continue
            chosen[family] = (source, path, header, count)
            screen[f"{source}/development_distinct"] += 1
    return chosen, dict(screen)


def bounded_tape(path, frozen_header, count, blocks):
    # Keep structural bytes only. After 144 actual moves, retain control records
    # to the next move boundary, and read only that next row's type byte.
    with path.open("rb") as stream:
        original = stream.read(JGS_HEADER)
        if original != frozen_header:
            raise ValueError("selected source header changed during preparation")
        header = bytearray(JGS_HEADER)
        header[:10] = original[:10]
        for offset in blocks:
            header[offset-28] = original[offset-28]
            header[offset:offset+30] = original[offset:offset+30]
        rows = []
        moves = 0
        for _ in range(count):
            tag = stream.read(1)
            if tag == b"\x5f" and moves == PLIES[-1]:
                break
            row = tag + stream.read(JGS_ROW - 1)
            if len(row) != JGS_ROW:
                raise ValueError("selected source no longer contains its declared event rows")
            moves += tag == b"\x5f"
            rows.append(row if tag == b"\x5f" else row[:5] + bytes(5))
        if len(rows) < 2:
            raise ValueError("native tape requires at least two structural rows")
        header[26:28] = len(rows).to_bytes(2, "little")
    return bytes(header) + b"".join(rows), moves, len(rows)


def prepare(repo, roots, limit):
    layout_group, blocks, valid_grid, model_roster = setup(repo)
    model = model_roster()["candidate"]
    locator = repo / "build" / "jgs-evidence-input.json"
    corpus = Path(json.loads(locator.read_text(encoding="utf-8-sig"))["corpus"])
    excluded = corpus_families(corpus, layout_group, blocks, valid_grid)
    development, screen = development_paths(roots, excluded, layout_group, blocks, valid_grid)
    if not development:
        raise ValueError("no eligible historical development layouts")
    weights = model["weights"]
    if len(weights) != 5:
        raise ValueError("the native probe expects five frozen short-v1 weights")
    rows = [" ".join(map(str, weights))]
    inputs = {}
    for family in sorted(development)[:limit]:
        source, path, header, count = development[family]
        tape, moves, events = bounded_tape(path, header, count, blocks)
        key = family.hex()
        rows.append(f"{key} {tape.hex()}")
        inputs[key] = dict(source=source, prefix_sha256=sha(tape), moves=moves, records=events)
    feed = "\n".join(rows) + "\n"
    evidence = dict(model_sha256=model["artifact_sha256"],
                    locator_sha256=file_sha(locator), corpus_sha256=file_sha(corpus),
                    historical_families=len(excluded), filtered=screen,
                    requested_plies=list(PLIES), inputs_sha256=sha(feed.encode("ascii")),
                    families=inputs, holdout_payload_read=False)
    return feed, evidence


def native_rows(output, keys):
    expected = {(key, ply) for key in keys for ply in PLIES}
    reports, missing = {}, {}
    for line in output.splitlines():
        tag, key, ply_text, *rest = line.split()
        identity = (key, int(ply_text))
        if identity not in expected or identity in reports or identity in missing:
            raise ValueError("unexpected or duplicate native root")
        if tag == "M" and len(rest) == 1:
            missing[identity] = rest[0]
        elif tag == "R" and len(rest) == 14 and rest[0] == "ok":
            _, admits, edited, changed, a_status, b_status, a_plies, b_plies, a_calls, b_calls, scanned, scan_ns, time_a, time_b = rest
            values = tuple(map(int, (admits, edited, changed)))
            plies = tuple(map(int, (a_plies, b_plies)))
            calls = tuple(map(int, (a_calls, b_calls)))
            if (any(x not in (0, 1) for x in values) or min(plies) < 0
                    or calls != (3, 3) or (values[2] and not values[1])):
                raise ValueError("native comparison failed binary or equal-budget invariants")
            reports[identity] = dict(zip(("top3_admits", "edited", "selected_changed"), values),
                                     a_status=a_status, b_status=b_status, a_plies=plies[0],
                                     b_plies=plies[1], evaluator_calls_a=calls[0], evaluator_calls_b=calls[1],
                                    nomination_calls=int(scanned), time_ns=list(map(int,(scan_ns,time_a,time_b))))
        else:
            raise ValueError("unexpected native output format")
    if expected != reports.keys() | missing.keys():
        raise ValueError("native output omitted requested roots")
    return reports, missing


def summarize(reports, missing):
    settled = list(reports.values())
    transitions = Counter(f"{row['a_status']}->{row['b_status']}" for row in settled)
    changes = [row for row in settled if row["selected_changed"]]
    return dict(requested=len(reports) + len(missing), compared=len(reports),
                paired_families=len({key for key, _ in reports}),
                missing=dict(Counter(missing.values())),
                top3_admits=sum(row["top3_admits"] for row in settled),
                edited=sum(row["edited"] for row in settled),
                selected_changed=len(changes), status_transitions=dict(transitions),
                changed_status_transitions=dict(Counter(f"{r['a_status']}->{r['b_status']}" for r in changes)),
                changed_families=len({key for (key, _), row in reports.items() if row["selected_changed"]}),
                nomination_calls=sum(row["nomination_calls"] for row in settled), evaluator_calls=[sum(row[name] for row in settled)
                                 for name in ("evaluator_calls_a", "evaluator_calls_b")])


def compile_native(out):
    sources=("core/board.cpp","core/layout.cpp","core/rules.cpp","core/state.cpp",
             "tests/relay_candidate_budget.cpp")
    files=" ".join(f'"{REPO.as_posix()}/{source}"' for source in sources)
    command=(f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX /utf-8 '
             f'/I"{REPO.as_posix()}/core" {files} /Fe:relay_budget.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"',cwd=out,check=True,timeout=95)


def sample(out):
    started=time.perf_counter()
    feed,evidence=prepare(REPO,ROOTS,64)
    if evidence["inputs_sha256"]!="cbe0ffd369379d1ae6e60030de4feb394a3e54b4fdfedfe0493cd6e899af58aa":
        raise ValueError("frozen development prefixes or weights changed")
    native=out / "relay_budget.exe"
    process=subprocess.run([native,REPO / "config/ruleset.json"],input=feed,capture_output=True,
                           text=True,encoding="utf-8",errors="replace",timeout=130,check=False)
    if process.returncode:
        raise ValueError(f"native pilot exit {process.returncode}: {process.stderr.strip()}")
    reports,missing=native_rows(process.stdout,evidence["families"])
    timing=[sum(row["time_ns"][i] for row in reports.values())/1e6 for i in range(3)]
    for row in reports.values():
        del row["time_ns"]
    summary=summarize(reports,missing)
    decision=(summary["changed_families"]>=8 and
              any(value and key.split("->")[1]=="available" and key.split("->")[0]!="available"
                  for key,value in summary["changed_status_transitions"].items()))
    evidence.update(native_sha256=file_sha(native), summary=summary,
                    pass_to_abc_coverage=decision,
                    reports=[dict(key=k,ply=p,**row) for (k,p),row in sorted(reports.items())],
                    missing=[dict(key=k,ply=p,reason=reason) for (k,p),reason in sorted(missing.items())])
    runtime=dict(phase_ms=timing,end_to_end_seconds=round(time.perf_counter()-started,3))
    if process.stderr:
        (out / "prefix-rejections.log").write_text(process.stderr,encoding="utf-8")
    return evidence,runtime


def main(audit):
    key=os.environ.get("JUNQI_SERIAL_KEY","manual")
    if not key or not all(c.isalnum() or c=="-" for c in key):
        raise ValueError("invalid local serial key")
    out=REPO / "build/relay-candidate-budget" / key
    receipt=out / "pilot.json"
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        compile_native(out)
    evidence,runtime=sample(out)
    if audit:
        if json.loads(receipt.read_text(encoding="utf-8"))["evidence"]!=evidence:
            raise ValueError("development replay pilot did not reproduce")
    else:
        temporary=receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(evidence=evidence,runtime=runtime),indent=2)+"\n",encoding="utf-8")
        temporary.replace(receipt)
    print(json.dumps(dict(summary=evidence["summary"],pass_to_abc_coverage=evidence["pass_to_abc_coverage"],
                          runtime=runtime),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
