"""Replicate the predeclared flag-pressure comparison on unused layouts."""
import json
import sys
import time

from camp_control import collect
from flag_interception import REPORT as DISCOVERY, ROOT, RUN, source_keys, summarize
from partner_crossplay import model_roster, sha

REPORT = ROOT/"docs/REPORTS/public-flag-interception-replication.json"
START, COUNT = 287200, 128
MIN_LAYOUTS = 12


def frozen_inputs():
    original = json.loads(DISCOVERY.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    frozen = dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"), parent=parent)
    if original["frozen"] != frozen:
        raise ValueError("discovery engine, rules or parent failed the frozen-data check")
    if {r["seed"] for r in original["sources"]} & set(range(START, START+COUNT)):
        raise ValueError("discovery layouts overlap the replication cohort")
    return parent, dict(discovery=sha(DISCOVERY), **frozen)


def verdict(summary):
    own = summary["groups"][0]
    if own["compatible_layouts"] < MIN_LAYOUTS:
        return "limited-coverage"
    estimates = own["comparisons"]
    if not all(estimates[f"{world}/1"]["exploratory_interval"][0] > 0 for world in (0, 1)):
        return "pressure-benefit-unresolved"
    if any(estimates[f"{world}/0"]["exploratory_interval"][1] < 0 for world in (0, 1)):
        return "ordinary-response-regression"
    return "conditional-pressure-benefit"


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    parent, frozen = frozen_inputs()
    summary = summarize(report["sources"], START, COUNT)
    if report["frozen"] != frozen or report["summary"] != summary or report["verdict"] != verdict(summary):
        raise ValueError("frozen replication data, outcome or classification changed")
    rows = report["sources"]
    first_hit = next((r for r in rows if r["points"]), rows[0])
    for row in (first_hit, rows[-1]):
        if row != collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, row["seed"], row["seat"],
                          row["variant"], mode="flag-interception"):
            raise ValueError("replication trajectory side check failed")
    print("PASS disjoint seeds, identical executable, threshold and matched replay")


def run():
    if REPORT.exists():
        raise FileExistsError("replication report exists; use audit")
    parent, frozen = frozen_inputs()
    started = time.monotonic()
    rows = [collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, seed, seat, variant, mode="flag-interception")
            for seed, seat, variant in source_keys(START, COUNT)]
    summary = summarize(rows, START, COUNT)
    report = dict(sources=rows, summary=summary, verdict=verdict(summary),
                  frozen=frozen, seconds=time.monotonic()-started)
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 2*1024**2:
        raise ValueError("replication report exceeded byte budget")
    with REPORT.open("xb") as out:
        out.write(data)
    print(f"Replicated {len(rows)} sources: {report['verdict']}")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
