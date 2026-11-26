"""Verify the recorded run and replay its first frozen block from a fresh shell."""
from pathlib import Path
import sys
import tempfile

from audit_cycle_scoring_model import PROBE, RECEIPT, ROOT, inputs, read, recover_cases
from compare_archive_cycle_exit_values import execute
from evidence_model import sha

MANIFEST = ROOT / "ops/cycle-run-manifest.json"


def require_asset(path, expected, recovery):
    if not path.is_file():
        raise ValueError(f"missing {path.name}: {recovery}")
    if sha(path) != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}: {recovery}")


def asset_path(asset):
    if "locator" in asset:
        locator = asset["locator"]
        try:
            return Path(read(ROOT / locator["path"])[locator["key"]])
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"cannot resolve {locator['key']} in {locator['path']}: {asset['recover']}") from error
    return ROOT / asset["path"]


def check_failures(model, rules):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for asset, label in ((model, "missing"), (rules, "SHA256 mismatch")):
            fixture = root / Path(asset["path"]).name
            if label != "missing":
                fixture.write_bytes(b'{"wrong_rules":true}')
            try:
                require_asset(fixture, asset["sha256"], asset["recover"])
            except ValueError as error:
                if label not in str(error) or asset["recover"] not in str(error):
                    raise
            else:
                raise ValueError(f"{label} fixture reached replay")


def verify(manifest):
    assets = {asset["role"]: asset for asset in manifest["assets"]}
    if len(assets) != len(manifest["assets"]):
        raise ValueError("asset roles must be unique")
    for asset in assets.values():
        require_asset(asset_path(asset), asset["sha256"], asset["recover"])
    check_failures(assets["model"], assets["model_rules"])
    for stage in manifest["stages"]:
        receipt = read(ROOT / stage["receipt"])
        if (receipt["status"] != "passed" or receipt["ticket"] != manifest["run"]["ticket"] or
                receipt["baseline"] != manifest["run"]["dispatch_head"] or receipt["index"] != stage["index"]):
            raise ValueError(f"invalid stage receipt: {stage['receipt']}")
    old, weights, corpus, _ = inputs()
    recorded = read(RECEIPT)
    keys = old["selected"]
    if keys != manifest["pairing"]["ordered_source_keys"]:
        raise ValueError("source order changed")
    key = keys[0]
    case = recover_cases(keys, corpus)[:1]
    for mode, expected in (("pairs", old["pairs"]), ("collision-pairs", recorded["corrected_pairs"])):
        decisions, pairs = execute(case, weights, PROBE, mode)
        if decisions != {key: old["decisions"][key]} or pairs != {key: expected[key]}:
            raise ValueError(f"{mode} changed the first frozen block")
    print(f"PASS handoff: {len(assets)} assets, 3 stage receipts, first frozen block in both modes")
    print("PASS handoff failures: missing model and wrong model rules rejected with recovery instructions")


if __name__ == "__main__":
    try:
        verify(read(MANIFEST))
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"BLOCKED handoff: {error}")
