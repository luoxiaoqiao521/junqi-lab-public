"""Verify public relay screening on native development fixtures."""
import json
import os
import subprocess
import sys

from verify_relay_continuity import ROOT, compile_probe, sha

EXPECTED = (
    "PASS public relay screen: two hidden worlds, new, blocked, earlier detour, engineer only, unknown mobility",
    "PASS public relay replies: identical opener, one still available, one blocked, one target withdrew",
)


def main(audit):
    key = os.environ.get("JUNQI_SERIAL_KEY", "manual")
    if not key or not all(char.isalnum() or char == "-" for char in key):
        raise ValueError("invalid local serial key")
    out = ROOT / "build" / "relay-public-screen" / key
    if not audit:
        out.mkdir(parents=True, exist_ok=True)
        compile_probe(out)
    executable = out / "relay.exe"
    result = subprocess.run([executable, "--public-relay-screen-fixtures"], capture_output=True,
                            text=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError(f"native fixture exit {result.returncode}: {result.stderr.strip()}")
    observed = tuple(result.stdout.splitlines())
    if observed != EXPECTED:
        raise ValueError(f"public relay output differs: {observed!r}")
    sources = ("sim/public_relay_screen.h", "tests/public_relay_screen_cases.h",
               "tests/core_smoke.cpp", "tools/verify_public_relay_screen.py")
    receipt = {"executable": sha(executable), "sources": {path: sha(ROOT / path) for path in sources},
               "verified": list(observed), "holdout_opened": False, "promotion_allowed": False}
    output = out / "fixtures.json"
    if audit:
        if json.loads(output.read_text(encoding="utf-8")) != receipt:
            raise ValueError("public relay binary or receipt changed")
    else:
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        temporary.replace(output)
    print(*observed, sep="\n")
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
