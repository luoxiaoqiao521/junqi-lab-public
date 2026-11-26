"""Development comparison: recorded quiet exits versus a still-legal return."""
from collections import Counter, defaultdict
import hashlib
import json
from random import Random
from statistics import mean
import subprocess
import sys
import tempfile
import time

from audit_public_attack_windows import write_json
from audit_qq_archive import initial_board
from compare_qq_rules import tape
from partner_crossplay import model_roster
from verify_archive_cycle_exit_options import EXE, RESULT as PRIOR, FIELDS, compile_probe, selected

RESULT = PRIOR.with_name("archive-cycle-exit-values.json")


def selection():
    prior = json.loads(PRIOR.read_text(encoding="utf8"))
    eligible = [r for r in prior["rows"] if r["source"] == "public" and r["native"] and
                r["native"]["return_quiet"] and r["departure"]["kind"].startswith("quiet")]
    chosen = []
    for alias in (0, 1):
        pool = sorted([r for r in eligible if r["native"]["alias"] == alias], key=lambda r: r["key"])
        chosen.extend(Random(88401+alias).sample(pool, 32))
    assert len({r["layout"] for r in chosen}) == 64
    by_key = {m["key"]: record for m, record in selected()}
    return [(r, by_key[r["key"]]) for r in chosen]


def probe_output(cases, weights, executable, mode):
    assert len(weights) == 5
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=executable.parent) as feed:
        print(len(cases), file=feed)
        print(*weights, 0.0, file=feed)
        for meta, (_, layouts, seats, events) in cases:
            entries, ccw = tape(layouts, seats, events[:meta["departure"]["index"]+1])
            print(meta["key"], len(entries), ccw, *meta["back"], file=feed)
            for (x, y), (_, rank) in initial_board(layouts).items():
                print(x, y, rank, file=feed)
            for entry in entries:
                print(*entry, file=feed)
        feed.seek(0)
        text = subprocess.run([executable, mode], stdin=feed, capture_output=True, text=True,
                              check=True, timeout=130).stdout
    return text.splitlines()


def execute(cases, weights, executable=EXE, mode="pairs"):
    decisions, pairs = {}, defaultdict(list)
    for line in probe_output(cases, weights, executable, mode):
        tag, key, *values = line.split()
        if tag == "D":
            assert key not in decisions and len(values) == len(FIELDS)
            decisions[key] = dict(zip(FIELDS, [int(v) for v in values[:6]]+[float(v) for v in values[6:]], strict=True))
        elif tag == "P":
            pairs[key].append(list(map(int, values)))
        else:
            raise ValueError("prevalidated source was rejected")
    assert set(decisions) == set(pairs) == {m["key"] for m, _ in cases}
    return decisions, dict(pairs)


def interval(blocks):
    rng = Random(88457)
    samples = sorted(mean(rng.choices(blocks, k=len(blocks))) for _ in range(2000))
    return [samples[50], samples[1949]]


def panel(keys, pairs, variant):
    blocks, lowers, uppers = [], [], []
    transitions = Counter()
    outcomes = {"human": Counter(), "return": Counter()}
    for key in keys:
        group = [row for row in pairs[key] if row[0] == variant]
        lower, upper = [], []
        for _, _, a, b in group:
            transitions[f"{b}>{a}"] += 1
            outcomes["human"][str(a)] += 1
            outcomes["return"][str(b)] += 1
            lower.append((a/2 if a >= 0 else 0)-(b/2 if b >= 0 else 1))
            upper.append((a/2 if a >= 0 else 1)-(b/2 if b >= 0 else 0))
        lowers.append(mean(lower))
        uppers.append(mean(upper))
        blocks.append(mean(lower))
    complete = all(l == u for l, u in zip(lowers, uppers, strict=True))
    return dict(layouts=len(keys), pairs=sum(transitions.values()), outcomes=outcomes,
                transitions=dict(transitions), unknown_result_bounds=[mean(lowers), mean(uppers)],
                mean=mean(blocks) if complete else None, block_bootstrap_95=interval(blocks) if complete else None)


def summarized(cases, pairs):
    groups = {"all": [m["key"] for m, _ in cases]}
    groups.update({f"alias_{alias}": [m["key"] for m, _ in cases if m["native"]["alias"] == alias]
                   for alias in (0, 1)})
    return {name: {str(v): panel(keys, pairs, v) for v in (0, 2)} for name, keys in groups.items()}


def audit(cases=None, rescan=True):
    data = json.loads(RESULT.read_text(encoding="utf8"))
    assert data["prior_sha256"] == hashlib.sha256(PRIOR.read_bytes()).hexdigest()
    frozen = model_roster()["candidate"]
    assert data["weights_sha256"] == frozen["artifact_sha256"]
    cases = selection() if cases is None else cases
    assert data["selected"] == [m["key"] for m, _ in cases]
    assert data["decisions"] == {m["key"]: m["native"] for m, _ in cases}
    for rows in data["pairs"].values():
        assert [(v, n) for v, n, _, _ in rows] == [(v, n) for v in (0, 2) for n in range(12)]
        assert all(a in (-1, 0, 1, 2) and b in (-1, 0, 1, 2) for _, _, a, b in rows)
    assert set(data["pairs"]) == set(data["selected"])
    if rescan:
        assert execute(cases, frozen["weights"]) == (data["decisions"], data["pairs"])
    assert data["summary"] == summarized(cases, data["pairs"])
    print("PASS 64 fixed layout blocks, 1536 complete paired cells, two frozen rosters and deterministic replay")
    print(json.dumps(data["summary"]["all"], sort_keys=True))


def run():
    start = time.perf_counter()
    cases = selection()
    compile_probe()
    frozen = model_roster()["candidate"]
    decisions, pairs = execute(cases, frozen["weights"])
    write_json(RESULT, dict(prior_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
        weights_sha256=frozen["artifact_sha256"], selected=[m["key"] for m, _ in cases],
        decisions=decisions, pairs=pairs, summary=summarized(cases, pairs),
        rules="archive-prefix: ccw from record, involved no-move, reveal flags, quiet71, cap3000",
        elapsed_seconds=time.perf_counter()-start))
    audit(cases, False)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
