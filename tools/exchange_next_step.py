"""Choose the next bounded experiment from audited exchange evidence."""
import json
from pathlib import Path
import sys

from evidence_model import sha
from exchange_response_confirm import REPORT, summarize

OUTPUT = Path(__file__).resolve().parents[1]/"docs/REPORTS/public-exchange-next.json"


def choose(summary):
    comparisons = summary["exchange"]["pressure_minus_other"]
    rep = summary["representation"]
    context_support = all(
        (value := comparisons.get(f"{world}/effect_of_enemy")) is not None
        and value["exploratory_interval"][0] > 0 for world in (0, 1)
    )
    adequate = all(group["1"] >= 4 and group["-1"] >= 4 for group in rep["direction_layouts"])
    if not context_support:
        route, why = "compare-response-coverage", "enemy response gain lacked cross-world replication"
    elif not adequate:
        route, why = "sample-exchange-counterexamples", "too few independently sourced stable attack and guard contrasts"
    elif rep["five_feature_conflicts"]:
        if rep["repeated_interception_discriminators"]:
            route, why = "test-public-interception-feature", "one public cue resolved replicated feature conflicts"
        else:
            route, why = "find-public-exchange-discriminator", "five-feature conflicts reproduced without a validated cue"
    else:
        route, why = "check-exchange-ranking", "no exact conflict found; current representation still needs a fit/holdout check"
    return dict(next=route, reason=why, confirmation_eligible=False,
                revisit_independent_layouts=summary["revisits"]["changed_pair_layouts"])


def shadow_checks():
    measured = {"exploratory_interval": [.01, .03]}
    report = {"exchange": {"pressure_minus_other": {}}, "revisits": {"changed_pair_layouts": 0},
              "representation": {"direction_layouts": [{"1": 4, "-1": 3}, {"1": 4, "-1": 4}],
                                 "five_feature_conflicts": [], "repeated_interception_discriminators": []}}
    checks = [(None, "compare-response-coverage"),
              (("exchange", "pressure_minus_other", {f"{w}/effect_of_enemy": measured for w in (0, 1)}),
               "sample-exchange-counterexamples"),
              (("representation", "direction_layouts", [{"1": 4, "-1": 4}]*2), "check-exchange-ranking"),
              (("representation", "five_feature_conflicts", [dict(delta=[0, 0, 0, 1, 0])]),
               "find-public-exchange-discriminator"),
              (("representation", "repeated_interception_discriminators", [[0, 0, 0, 1, 0]]),
               "test-public-interception-feature")]
    for update, expected in checks:
        if update:
            section, key, value = update
            report[section][key] = value
        if choose(report)["next"] != expected:
            raise ValueError(f"route check failed for {expected}")


def main():
    shadow_checks()
    data = json.loads(REPORT.read_text(encoding="utf8"))
    summary = summarize(data["sources"])
    if summary != data["summary"]:
        raise ValueError("routing inputs differ from the raw evidence")
    result = dict(report_sha256=sha(REPORT), decision=choose(summary))
    if sys.argv[1:] == ["audit"]:
        if json.loads(OUTPUT.read_text(encoding="utf8")) != result:
            raise ValueError("recorded research route changed on audit")
    else:
        with OUTPUT.open("x", encoding="utf8") as output:
            output.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["decision"]))


if __name__ == "__main__":
    main()
