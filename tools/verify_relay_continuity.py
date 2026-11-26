"""Compile and verify the continuous paired relay mechanics without using a holdout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_qq_rules import VC

ROOT = Path(__file__).resolve().parents[1]
COMPILE = ("core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp", "tests/core_smoke.cpp")
CONTRACT = (*COMPILE, "sim/relay_continuity.h", "tests/relay_continuity_cases.h", "tools/verify_relay_continuity.py")
EXPECTED = (
    "PASS relay paired trajectories: 24 no-trigger, 24 same-fallback, 24 receiver overrides",
    "PASS relay actual turn: skip, opener acts again, receiver regains choice",
    "PASS relay settlement: receiver retires and team wins; early draw and cutoff do not call selector",
    "PASS relay invalid override rejected",
    "PASS relay hidden-world boundary: same public history and ordered actions",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_probe(out):
    files = " ".join(f'"{ROOT.as_posix()}/{file}"' for file in COMPILE)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
               f'/I"{ROOT.as_posix()}/core" {files} /Fe:relay.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=out, check=True, timeout=100)


def score(out):
    executable = out / "relay.exe"
    result = subprocess.run([executable, "--relay-continuity-fixtures"], capture_output=True,
                            text=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError(f"native fixture exit {result.returncode}: {result.stderr.strip()}")
    observed = tuple(result.stdout.splitlines())
    if observed != EXPECTED:
        raise ValueError(f"relay acceptance output differs: {observed!r}")
    return dict(sources={name: sha(ROOT / name) for name in CONTRACT}, executable=sha(executable),
                verified=list(observed), holdout_opened=False, promotion_allowed=False)


def main(audit):
    key = os.environ.get("JUNQI_SERIAL_KEY", "manual")
    if not all(c.isalnum() or c == "-" for c in key):
        raise ValueError("invalid local serial key")
    out = ROOT / "build/relay-continuity" / key
    result = out / "fixtures.json"
    if not audit:
        out.mkdir(parents=True, exist_ok=True)
        compile_probe(out)
    receipt = score(out)
    if audit:
        if json.loads(result.read_text(encoding="utf-8")) != receipt:
            raise ValueError("relay binary or acceptance receipt changed")
    else:
        temporary = result.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        temporary.replace(result)
    print(*receipt["verified"], sep="\n")
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
