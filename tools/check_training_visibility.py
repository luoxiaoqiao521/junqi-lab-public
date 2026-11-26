"""Check the monitor against a completed archive run and a real stopped launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PWSH = os.environ.get("JUNQI_PWSH", "pwsh")
COMPLETE = ROOT / "build/archive-pressure-balanced"


def snapshot(directory):
    result = subprocess.run([PWSH, "-NoProfile", "-File", str(ROOT / "tools/training_monitor.ps1"),
                             "-Snapshot", "-Directory", str(directory), "-Hours", "1"],
                            text=True, encoding="utf8", capture_output=True, check=True, timeout=25)
    return json.loads(result.stdout)


def main():
    done = snapshot(COMPLETE)
    assert done["Status"] == "complete" and done["Completed"] == done["Total"] == 2
    with tempfile.TemporaryDirectory(prefix="archive-progress-", dir=ROOT / "build") as folder:
        directory = Path(folder)
        (directory / "STOP").touch()
        outcome = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools/bounded_training.py"),
                                  "--profile", "archive-pressure", "--directory", str(directory), "--hours", "1"],
                                 text=True, encoding="utf8", capture_output=True, timeout=45)
        stopped = snapshot(directory)
        assert outcome.returncode != 0 and "stopped before launch" in outcome.stderr
        assert stopped["Status"] == "stopped" and stopped["Completed"] == 0 and stopped["Total"] == 2
        assert not (directory / "ordinary.json").exists() and not (directory / "summary.json").exists()
    print("PASS archive trainer monitor: completed 2/2 and stopped before any training or summary")


if __name__ == "__main__":
    main()
