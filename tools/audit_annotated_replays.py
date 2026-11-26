"""Audit selected human replays, layout collections, and the recorder's source offline."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from archive_action_screen import layout_group
from audit_public_attack_windows import attack_outcome, write_json
from audit_qq_archive import LIMIT, ROOT, archived_move, decode, decode_rank_grid, initial_board, replay_facts
from compare_qq_rules import EXE, compile_probe, tape
from local_archive_inputs import archive_value, selected_replay

DOWNLOADS = Path.home() / "Downloads"
COLLECTIONS = ("诸葛阵", "一万个经典军棋布局")
RESULT = ROOT / "build" / "qq-evidence" / "annotated-source-audit.json"


def mirror(layout):
    return b"".join(layout[i:i+5][::-1] for i in range(0, 30, 5))


def layout_audit(raws):
    total = rejected = 0
    layouts = set()
    for raw in raws:
        total += 1
        layout = decode_rank_grid(raw)
        if layout is None:
            rejected += 1
        else:
            layouts.add(layout)
    families = {min(item, mirror(item)) for item in layouts}
    profiles = Counter()
    for layout in families:
        hq = 28 if layout[26] == 2 else 26
        profiles["bomb_in_non_flag_hq"] += layout[hq] == 4
        profiles["mine_in_non_flag_hq"] += layout[hq] == 3
        profiles["mine_ahead_of_flag"] += layout[layout.index(2)-5] == 3
    return dict(files=total, rejected=rejected, unique=len(layouts), mirror_groups=len(families),
                profiles=dict(profiles)), families


def layout_files(folder):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".jql":
            # Wrong file size is still counted, without loading arbitrary content.
            yield path.read_bytes() if path.stat().st_size == 50 else b""


def nested_sources(folder):
    zip_path, = folder.rglob("*.zip")
    rar_path, = folder.rglob("*.rar")
    with ZipFile(zip_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        extensions = Counter(Path(item.filename).suffix.lower() for item in members)
        layouts = [archive.read(item) if item.file_size == 50 else b""
                   for item in members if Path(item.filename).suffix.lower() == ".jql"]
    if rar_path.stat().st_size > 1_000_000:
        raise ValueError("unexpected RAR size")
    listing = subprocess.check_output(["tar.exe", "-tf", str(rar_path)], timeout=10)
    entries = listing.lower().count(b".jgs")
    # Read archive contents to stdout, never extract or execute a member.
    with tempfile.TemporaryFile(dir=RESULT.parent) as sink:
        subprocess.run(["tar.exe", "-xOf", str(rar_path)], stdout=sink, check=True, timeout=10)
        if sink.tell() > 1_000_000:
            raise ValueError("unexpected RAR payload")
        sink.seek(0)
        payload = sink.read()
    offsets = [m.start() for m in re.finditer(rb"QQGame JQS", payload)] + [len(payload)]
    records = [payload[a:b] for a, b in zip(offsets, offsets[1:])]
    if offsets[0] != 0 or entries != len(records):
        raise ValueError("RAR stream framing does not match its directory")
    return dict(extensions), layouts, records


def recorded_milestones(layouts, seats, events):
    board = initial_board(layouts)
    active = set(range(4))
    flags, exits, majors = [], [], []
    ply = 0
    for e in events:
        if e[0] == 245:
            kind, code, cause = e[1:4]
            if kind in (3, 4) and code in seats:
                seat = seats.index(code)
                if seat in active:
                    exits.append([ply, seat, kind, cause])
                active.discard(seat)
                board = {point: piece for point, piece in board.items() if piece[0] != seat}
            continue
        ply += 1
        actor = board[e[2], 16-e[3]]
        target = board.get((e[4], 16-e[5]))
        outcome = e[1] & 3
        if target and target[1] == 2 and outcome in (1, 3):
            flags.append([ply, actor[0], target[0], len(active)])
        losses = ([actor] if outcome in (2, 3) else []) + ([target] if outcome in (1, 3) else [])
        majors.extend([ply, seat, rank] for seat, rank in losses if rank in (5, 6))
        archived_move(board, e)
    sides = {seat % 2 for seat in active}
    surviving_team = next(iter(sides)) if len(sides) == 1 else None
    solo_entry = next((ply for ply, seat, _, _ in exits
                       if seat % 2 == surviving_team and
                       not any(p <= ply and s % 2 != seat % 2 for p, s, _, _ in exits)), None)
    return dict(flags=flags, exits=exits, major_losses=majors,
                surviving_team=surviving_team, recorded_solo_vs_two_from=solo_entry)


def native_facts(layouts, seats, events):
    entries, ccw = tape(layouts, seats, events)
    origins = [(x, y, rank) for (x, y), (_, rank) in initial_board(layouts).items()]
    payload = "\n".join(" ".join(map(str, row)) for row in [(1,), (0, len(entries), ccw), *origins, *entries])
    lines = subprocess.check_output([EXE], input=payload+"\n", text=True, timeout=8).splitlines()
    names = ("games", "plies", "contacts", "bad_legal", "bad_contact", "no_moves", "no_move_mismatches")
    return dict(zip(names, map(int, lines[0].split()[:len(names)]), strict=True))


def annotated_replays():
    names = archive_value("annotated_replays")
    if not isinstance(names, list) or len(names) != 11:
        raise ValueError("Unexpected local annotated selection")
    return [selected_replay(DOWNLOADS, name, LIMIT) for name in names]


def replay_audit(folder, compressed):
    groups = [("selected", annotated_replays())]
    groups.append(("layout-folder", [p.read_bytes() for p in sorted(folder.rglob("*.jgs"))]))
    groups.append(("rar", compressed))
    seen, layout_sets, data = set(), set(), []
    compile_probe()
    for group, raws in groups:
        for index, raw in enumerate(raws, 1):
            label = f"{group}-{index:02}"
            try:
                decoded = decode(raw)
                key, layouts, seats, events = decoded
                stats, ends = replay_facts(layouts, seats, events)
                native = native_facts(layouts, seats, events)
                milestones = recorded_milestones(layouts, seats, events)
            except (ValueError, KeyError) as error:
                data.append(dict(label=label, rejected=type(error).__name__))
                continue
            shape = layout_group(layouts)
            shape_new = shape not in layout_sets
            layout_sets.add(shape)
            window = attack_outcome(decoded)
            data.append(dict(label=label, key=key[:16], duplicate=key in seen, new_layout_group=shape_new,
                             moves=stats["moves"], contacts=stats["contacts"], ends=ends, native=native,
                             milestones=milestones, flag_window=window if not window else
                             dict(start=window["start"], outcome=window["outcome"], enemies=window["enemies_left"])))
            seen.add(key)
    return data


def memory_audit():
    folder = DOWNLOADS / "军棋子力记忆器"
    form = (folder / "jiqi.frm").read_text(encoding="gb18030")
    setup = (folder / "yanse.frm").read_text(encoding="gb18030")
    checks = {
        "colour_picker": all(f"Command{i}_Click" in setup for i in range(1, 5)),
        "four_24_slot_lists_excluding_flag": all(x in form for x in ("w(1 To 96)", "Mod 24", 'bing$ = "弹"'))
            and 'bing$ = "旗"' not in form,
        "one_click_hide": re.search(r"If b = 1 Then\s+z\(h\)\.Visible = False", form) is not None,
        "restore_slot": re.search(r"zz_Click.*?z\(h\)\.Visible = True", form, re.S) is not None,
        "right_click_highlight": re.search(r"If b = 2 Then\s+If w\(h\)", form) is not None,
        "side_reset": "For i = 24 * s - 23 To 24 * s" in form,
        "looks_for_live_input_or_persistence": re.search(
            r"(?im)\b(?:ReadProcessMemory|FindWindow|SetWindowsHookEx|GetPixel|FileSystemObject|"
            r"GetSetting|SaveSetting)\b|^\s*Open\s+.+\s+For\s+", form + "\n" + setup) is not None,
    }
    return dict(features=checks, files=dict(Counter(path.suffix.lower() for path in folder.rglob("*")
                                                    if path.is_file())))


def run():
    zhuge, collection = (DOWNLOADS / name for name in COLLECTIONS)
    zip_types, zipped_layouts, rar_replays = nested_sources(collection)
    first, families_a = layout_audit(layout_files(zhuge))
    second, families_b = layout_audit(layout_files(collection))
    third, families_c = layout_audit(zipped_layouts)
    result = dict(layouts=[first, second, third],
                  layout_overlap=[len(families_a & families_b), len(families_c & families_b)],
                  layout_group_union=len(families_a | families_b | families_c),
                  collection_formats=dict(Counter(path.suffix.lower() for path in collection.rglob("*")
                                                  if path.is_file())),
                  zip_types=zip_types, replays=replay_audit(collection, rar_replays), memory=memory_audit())
    write_json(RESULT, result)
    audit()


def audit():
    r = json.loads(RESULT.read_text(encoding="utf-8"))
    assert [data["files"] for data in r["layouts"]] == [22, 11146, 177], "layout batch changed"
    assert r["zip_types"] == {".jql": 177, ".jqh": 38}, "nested format coverage changed"
    features = r["memory"]["features"]
    assert all(value for key, value in features.items() if key != "looks_for_live_input_or_persistence")
    assert not features["looks_for_live_input_or_persistence"], "manual recorder contract changed"
    rows = r["replays"]
    assert Counter(row["label"].rsplit("-", 1)[0] for row in rows) == {
        "selected": 11, "layout-folder": 2, "rar": 8}
    aligned = [row for row in rows if "native" in row]
    assert len(aligned) >= 11 and all(row["native"]["games"] == 1 and
                                     row["native"]["plies"] == row["moves"] and
                                     row["native"]["contacts"] == row["contacts"] for row in aligned)
    strict = [row for row in aligned if not any(row["native"][name] for name in ("bad_legal", "bad_contact"))]
    print("PASS source accounting", len(rows), "decoded", len(aligned), "native_move_contact_aligned",
          len(strict), "layout_mirror_groups", r["layout_group_union"])


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
